"""Shared fakes for the referee, the worker and the orchestrator tests."""

from __future__ import annotations

import json
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from autoresearch import incumbent
from autoresearch.boxes.fake_box import FakeBox, ok
from autoresearch.boxes.protocol import CommandResult
from autoresearch.referee.referee import PATCHED_TREE
from autoresearch.types import Prediction, StopReason, Usage, WorkerOutput
from autoresearch.worker.protocol import WorkerInput

DIFF_TEMPLATE = (
    "diff --git a/networkx/algorithms/cluster.py b/networkx/algorithms/cluster.py\n"
    "--- a/networkx/algorithms/cluster.py\n"
    "+++ b/networkx/algorithms/cluster.py\n"
    "@@ -1 +1 @@\n"
    "-x = 1\n"
    "+x = {value}\n"
)


def diff_for(value: str) -> str:
    return DIFF_TEMPLATE.format(value=value)


def arg(command: str, flag: str) -> str:
    """Value of ``--flag value`` in a shell quoted command."""
    m = re.search(rf"{flag} (?:'([^']*)'|(\S+))", command)
    assert m, f"{flag} not in {command}"
    return m.group(1) if m.group(1) is not None else m.group(2)


def referee_box(
    box: FakeBox | None = None,
    *,
    speedup: float | Callable[[str], float] = 1.05,
    apply_ok: bool = True,
    module_ok: bool = True,
    full_ok: bool = True,
    fp_match: bool = True,
    contaminate_pairs: bool = False,
    ir_ok: bool = True,
    cleanup_ok: bool = True,
    tree_oracle: Callable[[str], CommandResult] | None = None,
) -> FakeBox:
    """A referee box whose patched tree runs ``speedup`` times faster than its incumbent.

    ``speedup`` may be a function of the patch text written to the box, so one
    box can answer differently per attempt.
    """
    box = box if box is not None else FakeBox()

    box.on("git rev-parse HEAD", ok("boxhead\n"))
    if tree_oracle is not None:
        box.on("git checkout -q --detach", tree_oracle, first=True)

    def apply_patch(cmd: str) -> CommandResult:
        if "--remove" in cmd:
            return ok(json.dumps({"kind": "worktree_removed", "ok": cleanup_ok, "error": ""}))
        if "--patch" in cmd and not apply_ok:
            return ok(
                json.dumps({"kind": "worktree", "ok": False, "error": "patch does not apply"})
            )
        return ok(
            json.dumps({"kind": "worktree", "ok": True, "applied": "--patch" in cmd, "error": ""})
        )

    box.on("apply_patch.py", apply_patch)

    def run_tests(cmd: str) -> CommandResult:
        scope = arg(cmd, "--scope")
        good = module_ok if scope == "module" else full_ok
        return ok(
            json.dumps(
                {
                    "kind": "tests",
                    "scope": scope,
                    "ok": good,
                    "passed": 10 if good else 9,
                    "failed": 0 if good else 1,
                    "errors": 0,
                    "duration_s": 1.5,
                }
            )
        )

    box.on("run_tests.py", run_tests)

    def current_speedup() -> float:
        if callable(speedup):
            patch = box.files.get("/workspace/work/attempt.diff", b"").decode()
            return speedup(patch)
        return speedup

    def time_target(cmd: str) -> CommandResult:
        root = arg(cmd, "--root")
        patched = root == PATCHED_TREE
        if "--verify" in cmd:
            fp = "fp_same" if (fp_match or not patched) else "fp_other"
            return ok(json.dumps({"kind": "verify", "hot_executed": True, "result_fp": fp}))
        t = 1.0 / current_speedup() if patched else 1.0
        if contaminate_pairs:
            samples = [{"t": t, "contaminated": True, "reasons": ["steal"]}]
            return ok(
                json.dumps({"kind": "run", "min_clean": None, "min_all": t, "samples": samples})
            )
        return ok(json.dumps({"kind": "run", "min_clean": t, "min_all": t, "samples": []}))

    box.on("time_target.py", time_target)
    box.on("canary.py", ok(json.dumps({"kind": "canary", "min_s": 0.5})))

    def count_ir(cmd: str) -> CommandResult:
        if not ir_ok:
            return ok(json.dumps({"kind": "ir", "ir": None, "error": "valgrind missing"}))
        root = arg(cmd, "--root")
        calls = int(arg(cmd, "--calls"))
        body = 800 if root == PATCHED_TREE else 1000
        return ok(json.dumps({"kind": "ir", "ir": 5000 + calls * body, "error": ""}))

    box.on("count_ir.py", count_ir)
    return box


def tree_oracle(box: FakeBox, origin: Path, base_sha: str) -> Callable[[str], CommandResult]:
    """Answer the incumbent sync the way a real box would: apply the stack diff the
    orchestrator wrote to the box onto a local clone of the same origin and report
    that clone's tree hash. The orchestrator then sees the hash it expects."""

    def handler(cmd: str) -> CommandResult:
        stack = box.files.get("/workspace/work/incumbent.diff", b"").decode()
        with tempfile.TemporaryDirectory() as tmp:
            clone = Path(tmp) / "clone"
            incumbent.clone_at(str(origin), base_sha, clone)
            if stack.strip():
                incumbent.apply_and_commit(clone, stack, "stack")
            tree = incumbent.tree_hash(clone)
        return ok(f"boxhead_{tree[:6]}\n{tree}\n")

    return handler


@dataclass
class FakeWorker:
    """Returns scripted outputs in order, or the same output each time if only one is given."""

    outputs: list[WorkerOutput] = field(default_factory=list)
    inputs: list[WorkerInput] = field(default_factory=list)

    def attempt(self, inp: WorkerInput) -> WorkerOutput:
        self.inputs.append(inp)
        if len(self.outputs) > 1:
            return self.outputs.pop(0)
        return self.outputs[0]


def submitted(patch: str | None, speedup: float = 1.1, turns: int = 5) -> WorkerOutput:
    return WorkerOutput(
        patch=patch,
        prediction=Prediction(speedup) if patch else None,
        rationale="did a thing" if patch else "",
        stop_reason=StopReason.SUBMITTED if patch else StopReason.MAX_TURNS,
        turns=turns,
        usage=Usage(1000, 600, 100, 40),
        wall_s=12.0,
        transcript='{"kind": "end"}\n',
        box_id="sb_worker",
    )
