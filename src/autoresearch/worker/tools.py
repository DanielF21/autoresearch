"""The worker's tools: a schema the model sees and an executor that runs in its box.

Four tools, deliberately. The agent gets a shell and a description of the
filesystem, and reads, searches and edits with it however it likes. The other
three exist because each one encodes something the agent would otherwise have to
rebuild correctly on every attempt: which tests are the module's, how a fair
back to back timing is ordered, and how an attempt ends.

Every executor takes the same context and returns a ToolResult. ``ended`` is
True only for ``submit``. The shell can do anything; the referee, not the tool
set, is what keeps a patch honest.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from autoresearch.boxes.image import BASE_DIR, REPO_DIR
from autoresearch.boxes.protocol import Box
from autoresearch.config import TargetSpec
from autoresearch.referee.referee import PIN_CORE, guest_command, target_args

MAX_OUTPUT = 12_000
SHELL_TIMEOUT_MAX = 900


class ToolError(ValueError):
    """Bad arguments. Reported to the model as the tool's result, never raised further."""


@dataclass(frozen=True)
class ToolContext:
    box: Box
    target: TargetSpec
    repo: str = REPO_DIR
    base: str = BASE_DIR


@dataclass(frozen=True)
class ToolResult:
    text: str
    ended: bool = False
    prediction: float | None = None
    rationale: str = ""


Executor = Callable[[ToolContext, dict[str, Any]], ToolResult]


def _clip(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return (
        text[: limit // 2]
        + f"\n... [{len(text) - limit} characters omitted] ...\n"
        + text[-limit // 2 :]
    )


def _int(args: dict[str, Any], key: str, default: int | None) -> int | None:
    v = args.get(key, default)
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError) as e:
        raise ToolError(f"{key} must be an integer") from e


# ----- executors ---------------------------------------------------------------------


def _guest(
    ctx: ToolContext, script: str, *args: str, timeout: int, pin: int | None = None
) -> dict[str, Any]:
    r = ctx.box.run(guest_command(script, *args, pin=pin), timeout=timeout)
    line = r.last_json_line()
    if line is None:
        return {"error": f"{script} failed (rc {r.exit_code}): {r.stderr[-800:]}"}
    return dict(json.loads(line))


MODULE_TESTS_TIMEOUT = 300


def run_tests(ctx: ToolContext, _args: dict[str, Any]) -> ToolResult:
    """The hot module's own tests, the only suite the worker runs.

    The whole suite is the referee's job. It runs on every submission and takes
    about a minute, so a worker running it too only spends its own budget to
    learn what the referee will establish anyway.
    """
    rec = _guest(
        ctx,
        "run_tests.py",
        "--root",
        ctx.repo,
        "--package-root",
        ctx.target.package_root,
        "--target",
        ctx.target.tests.module,
        "--scope",
        "module",
        "--workers",
        "1",
        "--log",
        "/workspace/worker_tests.log",
        "--timeout",
        str(MODULE_TESTS_TIMEOUT - 30),
        timeout=MODULE_TESTS_TIMEOUT,
    )
    if "error" in rec:
        return ToolResult(f"error: {rec['error']}")
    head = (
        f"module tests: {'PASSED' if rec.get('ok') else 'FAILED'}: "
        f"{rec.get('passed', 0)} passed, {rec.get('failed', 0)} failed, {rec.get('errors', 0)} errors "
        f"in {float(rec.get('duration_s', 0)):.1f}s"
    )
    tail = "" if rec.get("ok") else "\n" + _clip(str(rec.get("tail", "")), 6000)
    return ToolResult(head + tail)


def run_benchmark(ctx: ToolContext, _args: dict[str, Any]) -> ToolResult:
    """Two back to back pairs per input, working tree against the untouched base.

    Every input, not just the first, because a patch can be enormous on one and
    slower on another and the worker has no other way to see that. Noisy by
    design: it says whether the tree is warm or cold. The referee's own pairs,
    more of them per input, are the measurement that goes into history.

    Pinned to the same core the referee pins to, so a change that spreads work
    across cores looks the same here as it will there.
    """
    t = ctx.target
    lines: list[str] = []
    ratios: list[float] = []
    for spec in t.inputs:
        pair_ratios: list[float] = []
        failed = ""
        for order in (("base", "working"), ("working", "base")):
            times: dict[str, float] = {}
            for which in order:
                root = ctx.base if which == "base" else ctx.repo
                rec = _guest(
                    ctx,
                    "time_target.py",
                    *target_args(t, root, spec),
                    "--repeats",
                    "3",
                    "--no-counters",
                    timeout=600,
                    pin=PIN_CORE,
                )
                if rec.get("error"):
                    failed = str(rec["error"])
                    break
                times[which] = float(rec["min_all"])
            if failed:
                break
            pair_ratios.append(times["base"] / times["working"])
        if failed:
            lines.append(f"  {spec.name:<14} error: {_clip(failed, 200)}")
            continue
        mean = sum(pair_ratios) / len(pair_ratios)
        ratios.append(mean)
        flag = "  <-- SLOWER" if mean < 1.0 else ""
        lines.append(
            f"  {spec.name:<14} {mean:>8.3f}x   (pairs "
            f"{pair_ratios[0]:.3f}, {pair_ratios[1]:.3f}){flag}"
        )
    head = "indicative speedup per input, 2 pairs each on a shared machine:\n"
    if not ratios:
        return ToolResult(head + "\n".join(lines) + "\nNothing timed.")
    geo = math.exp(sum(math.log(r) for r in ratios if r > 0) / len(ratios))
    worst = min(ratios)
    tail = (
        f"\ngeometric mean {geo:.3f}x, worst {worst:.3f}x. The referee scores the geometric "
        "mean over all inputs and records a patch as a real speedup only if no input is "
        "slower and at least one clears its own noise floor. Being slower anywhere "
        "disqualifies a patch however fast it is elsewhere."
    )
    return ToolResult(head + "\n".join(lines) + tail)


def shell(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    cmd = args.get("cmd")
    if not isinstance(cmd, str) or not cmd.strip():
        raise ToolError("cmd must be a non empty string")
    timeout = min(_int(args, "timeout", 120) or 120, SHELL_TIMEOUT_MAX)
    r = ctx.box.run(cmd, timeout=timeout, cwd=ctx.repo)
    out = r.stdout
    if r.stderr:
        out += ("\n" if out else "") + "[stderr]\n" + r.stderr
    if r.timed_out:
        out += f"\n[timed out after {timeout}s]"
    return ToolResult(_clip(out) + f"\n[exit {r.exit_code}]")


def submit(_ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    try:
        speedup = float(args.get("predicted_speedup"))  # type: ignore[arg-type]
    except (TypeError, ValueError) as e:
        raise ToolError("predicted_speedup must be a number, for example 1.15") from e
    rationale = args.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ToolError("rationale must be a non empty string")
    return ToolResult("submitted", ended=True, prediction=speedup, rationale=rationale)


# ----- the tool set ------------------------------------------------------------------


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    run: Executor

    @property
    def spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _params(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required}


TOOLS: tuple[Tool, ...] = (
    Tool(
        "run_tests",
        "Run the hot module's own tests on your working tree. Takes seconds. The whole suite "
        "is not available here: the referee runs it on every submission and requires it to "
        "pass, so use this while you work and let the referee do the rest.",
        _params({}, []),
        run_tests,
    ),
    Tool(
        "run_benchmark",
        "Time the target on your working tree against the untouched base commit, on every "
        "benchmark input, two back to back pairs each. Reports a ratio per input plus their "
        "geometric mean. Indicative only: noisy, and not the referee's measurement. Run it "
        "before you submit: a patch that is slower on any input cannot count as a speedup, "
        "and this is the only way to see that coming.",
        _params({}, []),
        run_benchmark,
    ),
    Tool(
        "shell",
        "Run a shell command. This is how you read, search and edit. Commands run in "
        f"{REPO_DIR} unless you cd elsewhere. Output is truncated in the middle past "
        f"{MAX_OUTPUT} characters, so page through large files rather than printing them "
        f"whole. Default timeout 120 seconds, maximum {SHELL_TIMEOUT_MAX}.",
        _params({"cmd": {"type": "string"}, "timeout": {"type": "integer"}}, ["cmd"]),
        shell,
    ),
    Tool(
        "submit",
        "Finish the attempt. The harness takes git diff of your working tree as the patch. "
        "State the speedup you predict the referee will record, which is the geometric mean "
        "over every input, as a ratio, for example 1.15 for 15 percent faster, and a "
        "rationale a maintainer could read.",
        _params(
            {"predicted_speedup": {"type": "number"}, "rationale": {"type": "string"}},
            ["predicted_speedup", "rationale"],
        ),
        submit,
    ),
)

TOOL_BY_NAME = {t.name: t for t in TOOLS}
TOOL_SPECS = [t.spec for t in TOOLS]


def execute(ctx: ToolContext, name: str, args: dict[str, Any]) -> ToolResult:
    """Run one tool. Argument errors come back as text so the model can correct itself."""
    tool = TOOL_BY_NAME.get(name)
    if tool is None:
        return ToolResult(f"error: unknown tool {name!r}; available: {', '.join(TOOL_BY_NAME)}")
    try:
        return tool.run(ctx, args)
    except ToolError as e:
        return ToolResult(f"error: {e}")
