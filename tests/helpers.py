"""Shared fakes for the referee, the worker and the orchestrator tests."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from autoresearch.boxes.fake_box import FakeBox, ok
from autoresearch.boxes.protocol import CommandResult
from autoresearch.referee.referee import PATCHED_TREE
from autoresearch.types import AttemptRef, Prediction, StopReason, Usage, WorkerOutput
from autoresearch.worker.protocol import WorkerInput

BASE_SHA = "c94928ed94899033126c9d47f797a1f698584b20"

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
    per_input: dict[str, float] | None = None,
    head: str = BASE_SHA,
    apply_ok: bool = True,
    module_ok: bool = True,
    full_ok: bool = True,
    fp_match: bool = True,
    patched_verify_fails: bool = False,
    verify_fails_for: tuple[str, ...] = (),
    contaminate_pairs: bool | int = False,
    cleanup_ok: bool = True,
) -> FakeBox:
    """A referee box whose patched tree runs ``speedup`` times faster than the base.

    ``speedup`` may be a function of the patch text written to the box, so one
    box can answer differently per attempt. ``per_input`` overrides it for named
    inputs, which is how a patch that trades one input for another is faked.
    ``head`` is what the box reports after checking out the base; the referee
    refuses anything but the base sha. ``contaminate_pairs`` as True taints every
    timing sample; as an integer it taints that many timing launches per input
    (a pass of six pairs is twelve launches) and then lets the rest through,
    which is how a retry that succeeds is faked.
    """
    box = box if box is not None else FakeBox()
    timing_launches: dict[str, int] = {}

    box.on("git checkout -q --detach", ok(f"{head}\n"))

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
        # An explicit lookup, not an else: an unexpected label should fail the
        # test rather than quietly resolve to the full suite's answer.
        good = {"module": module_ok, "full": full_ok}[scope]
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
        if "--profile" in cmd:
            return ok(
                json.dumps(
                    {
                        "kind": "profile",
                        "call_s": 0.25,
                        "hot_executed": True,
                        "hot_s": 0.2,
                        "total_s": 0.25,
                        "hot_tottime_share": 0.8,
                        "flat": f"flat view of {arg(cmd, '--setup')}\n",
                        "callers": "callers view\n",
                        "result_fp": "fp_same",
                    }
                )
            )
        if "--verify" in cmd:
            # A verify launch carries no label, so an input is identified here
            # by its setup statements, which are the only thing that names it.
            setup = arg(cmd, "--setup")
            fails = patched_verify_fails or setup in verify_fails_for
            if patched and fails:
                return ok(json.dumps({"error": "RecursionError in patched tree"}))
            fp = "fp_same" if (fp_match or not patched) else "fp_other"
            return ok(
                json.dumps(
                    {
                        "kind": "verify",
                        "hot_executed": True,
                        "hot_tottime_share": 0.9,
                        "call_s": 0.5,
                        "fixed_s": 0.3,
                        "python": [3, 12, 4],
                        "result_fp": fp,
                    }
                )
            )
        # The referee labels a timing launch "<input name>/<tree>", which is the
        # only thing in the command that says which input is being timed.
        name = arg(cmd, "--label").split("/")[0]
        ratio = (per_input or {}).get(name, current_speedup())
        t = 1.0 / ratio if patched else 1.0
        fixed = 0.31 if patched else 0.29
        launch = timing_launches.get(name, 0)
        timing_launches[name] = launch + 1
        tainted = (
            contaminate_pairs if isinstance(contaminate_pairs, bool) else launch < contaminate_pairs
        )
        if tainted:
            samples = [{"t": t, "contaminated": True, "reasons": ["steal"]}]
            return ok(
                json.dumps(
                    {
                        "kind": "run",
                        "min_clean": None,
                        "min_all": t,
                        "samples": samples,
                        "fixed_s": fixed,
                    }
                )
            )
        return ok(
            json.dumps(
                {"kind": "run", "min_clean": t, "min_all": t, "samples": [], "fixed_s": fixed}
            )
        )

    box.on("time_target.py", time_target)
    box.on("canary.py", ok(json.dumps({"kind": "canary", "min_s": 0.5})))
    return box


@dataclass
class FakeAttemptTrace:
    """Records every call as a tuple, so tests assert on order as well as content."""

    calls: list[tuple[str, Any]] = field(default_factory=list)
    raises: bool = False

    def _add(self, name: str, payload: Any) -> None:
        if self.raises:
            raise RuntimeError(f"tracing is down: {name}")
        self.calls.append((name, payload))

    def box(self, box_id: str) -> None:
        self._add("box", box_id)

    def model_turn(self, turn: int, messages: list[Any], response: Any) -> None:
        self._add("model_turn", (turn, len(messages), response.finish_reason))

    def tool(self, turn: int, name: str, args: dict[str, Any], result: str) -> None:
        self._add("tool", (turn, name))

    def end(
        self,
        stop: StopReason,
        *,
        patch: str | None = None,
        prediction: Prediction | None = None,
        error: str = "",
    ) -> None:
        self._add("end", str(stop))

    @property
    def kinds(self) -> list[str]:
        return [k for k, _ in self.calls]


@dataclass
class FakeTracer:
    attempts: list[FakeAttemptTrace] = field(default_factory=list)
    raises: bool = False

    def attempt(self, ref: AttemptRef, run_id: str, base_sha: str) -> FakeAttemptTrace:
        trace = FakeAttemptTrace(raises=self.raises)
        self.attempts.append(trace)
        return trace

    @property
    def only(self) -> FakeAttemptTrace:
        assert len(self.attempts) == 1, f"expected one attempt, got {len(self.attempts)}"
        return self.attempts[0]


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
