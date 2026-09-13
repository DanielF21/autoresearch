"""The agent loop worker: a fresh box, a conversation with tools, one diff out.

Per attempt: create a box from the image, check its repo out at the run's base
commit, add a read only worktree of the same commit for the benchmark to compare
against, upload the guest programs and the history, then loop: ask the model, run
the tools it asks for, append the results, until it submits or a cap trips. The
box is terminated on every exit path. The patch is ``git diff`` of the repo,
never model text.

Caps and kill rules, each recorded as the stop reason:
- max_turns and max_seconds from the config. The worker is never told either.
  When the next turn is the last one the cap allows, the loop says so in a user
  message and offers only the submit tool; a submit then is recorded as
  last_turn, and anything else ends the attempt with the cap as the reason and
  the diff as it stands. The time cap is announced with a margin of a few
  median turns, since the next turn's length is unknown.
- max_input_tokens from the config, a hard stop
- repeated_tool_call: the same tool with the same arguments three times running
- no_progress: two consecutive turns with no tool call, after one nudge
- model_error and box_error: the platform failed after the client's retry
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autoresearch import observe
from autoresearch.boxes.image import BASE_DIR, GUEST_DIR, HISTORY_DIR, REPO_DIR
from autoresearch.boxes.protocol import Box, BoxError, BoxFactory
from autoresearch.config import RunConfig
from autoresearch.model.protocol import ChatModel, Message, ModelError, ToolCall
from autoresearch.observe import NullTracer, Tracer
from autoresearch.types import Attempt, Prediction, StopReason, Usage, WorkerOutput
from autoresearch.worker import prompt, tools
from autoresearch.worker.protocol import WorkerInput

GUEST_SOURCE = Path(__file__).parent.parent / "guest"
SETUP_TIMEOUT = 600
REPEAT_LIMIT = 3
NO_PROGRESS_LIMIT = 2
TRANSCRIPT_CLIP = 4000
TIME_MARGIN_TURNS = 2.0  # median turns kept in hand before the time cap


@dataclass
class _Transcript:
    lines: list[str] = field(default_factory=list)

    def add(self, **record: Any) -> None:
        record["t"] = time.time()
        self.lines.append(json.dumps(record, sort_keys=True))

    def text(self) -> str:
        return "\n".join(self.lines) + ("\n" if self.lines else "")


def _clip(s: str, n: int = TRANSCRIPT_CLIP) -> str:
    return s if len(s) <= n else s[:n] + f"... [{len(s) - n} more]"


def _release(box: Box) -> None:
    """Terminate a worker box. The one place a BoxError is reported rather than raised.

    This runs after the attempt's result is built, so raising would discard a
    finished attempt, patch and all, and take the round down with it. A box that
    will not terminate is still a leak that never sleeps, so it is said on
    stderr, which the run's launch log keeps, with the command that finds it.
    """
    try:
        box.terminate()
    except BoxError as e:
        print(
            f"worker box {box.name} ({box.box_id}) may still be running: {e}. "
            "autoresearch reap lists every live box.",
            file=sys.stderr,
            flush=True,
        )


def _call_signature(call: ToolCall) -> str:
    return call.name + ":" + json.dumps(call.arguments, sort_keys=True)


def _time_margin(turn_walls: list[float], factor: float = TIME_MARGIN_TURNS) -> float:
    """Seconds to keep in hand so the last turn can be announced before the cap.

    The loop cannot know how long the next turn takes, so it keeps a multiple of
    the median turn so far: width 64 turns ran about four times slower than
    width 16 on the same model. With no turn behind it there is nothing to
    reserve, and a run whose cap is already spent gets its last turn at once.
    """
    if not turn_walls:
        return 0.0
    ordered = sorted(turn_walls)
    return factor * ordered[len(ordered) // 2]


class AgentLoopWorker:
    def __init__(
        self,
        model: ChatModel,
        boxes: BoxFactory,
        config: RunConfig,
        tracer: Tracer | None = None,
    ) -> None:
        self._model = model
        self._boxes = boxes
        self._config = config
        self._tracer: Tracer = tracer if tracer is not None else NullTracer()

    # ----- box setup -----------------------------------------------------------------

    def _prepare_box(self, box: Box, inp: WorkerInput) -> str:
        """Set the box up and return the Python version it runs, as the prompt states it."""
        box.upload_dir(GUEST_SOURCE, GUEST_DIR)
        # The image cloned the repo at the base commit. Check it out explicitly and
        # confirm HEAD, so a stale image or a wrong config cannot go unnoticed.
        #
        # Both trees are byte compiled before the base worktree is made read only.
        # Order matters twice over. Python writes __pycache__ into the tree it
        # imports, so a base tree made read only while cold would recompile on
        # every launch and look slower than the working tree, which biases
        # run_benchmark in the agent's favour. Compiling both also removes the
        # cold start that used to fall on whichever tree ran first.
        #
        # The Python version is printed by the same command, so telling the
        # worker what it has costs no extra round trip and is never guessed.
        r = box.run(
            f"cd {REPO_DIR} && git checkout -q --detach {inp.base_sha}"
            f" && git rev-parse HEAD"
            f" && rm -rf {BASE_DIR} && git worktree prune"
            f" && git worktree add -q --detach {BASE_DIR} HEAD"
            f" && python3 -m compileall -q {REPO_DIR} {BASE_DIR} > /dev/null"
            f" && chmod -R a-w {BASE_DIR}"
            f" && python3 -c 'import sys; print(\"python\", sys.version.split()[0])'",
            timeout=SETUP_TIMEOUT,
        )
        if not r.ok:
            raise BoxError(f"worker box setup failed: {r.stderr[-800:]}")
        head = r.stdout.split()[0] if r.stdout.split() else ""
        if head != inp.base_sha:
            raise BoxError(f"worker box is at {head}, not the base {inp.base_sha}")
        versions = [ln.split()[1] for ln in r.stdout.splitlines() if ln.startswith("python ")]
        if not versions:
            raise BoxError("worker box did not report its Python version")
        self._upload_history(box, inp.history)
        return versions[-1]

    @staticmethod
    def _history_files(history: tuple[Attempt, ...]) -> dict[str, bytes]:
        """Raw artifacts only, keyed by path relative to the history directory.

        The diff, the measurement and the earlier worker's own words. No harness
        written summary: the prompt carries the index, and a summary here would
        be a third copy of the patch.
        """
        files: dict[str, bytes] = {}
        for a in history:
            d = a.ref.dirname
            if a.patch:
                files[f"{d}/patch.diff"] = a.patch.encode()
            if a.measurement is not None:
                files[f"{d}/measurement.json"] = json.dumps(
                    a.measurement.to_dict(), indent=1
                ).encode()
            if a.rationale.strip() or a.prediction is not None:
                # The worker's own words first, so the file's first line is the
                # mechanism sentence the index shows; the prediction follows.
                predicted = "unknown" if a.prediction is None else f"{a.prediction.speedup:.2f}x"
                files[f"{d}/rationale.md"] = (
                    f"{a.rationale.strip()}\n\nPredicted speedup: {predicted}\n"
                ).encode()
        return files

    def _upload_history(self, box: Box, history: tuple[Attempt, ...]) -> None:
        """One upload, not one call per file.

        A write per file is three round trips per attempt, so a width 16 run at
        round 32 would spend around 1500 of them before the agent's first turn.
        The tree is staged locally and sent in a single upload instead.
        """
        files = self._history_files(history)
        if not files:
            return
        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp)
            for name, data in files.items():
                path = staging / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            box.upload_dir(staging, HISTORY_DIR)

    def _collect_patch(self, box: Box) -> str:
        # No pathspec. The pristine baseline worktree is a sibling of the repo,
        # not a directory inside it, so a diff run here cannot reach it. Git
        # pathspecs are repository relative, so naming it by absolute path is an
        # error rather than a no op.
        r = box.run(f"cd {REPO_DIR} && git add -N . && git diff --binary", timeout=120)
        if not r.ok:
            raise BoxError(f"git diff failed: {r.stderr[-500:]}")
        return r.stdout

    # ----- the loop ------------------------------------------------------------------

    def attempt(self, inp: WorkerInput) -> WorkerOutput:
        cfg = self._config.worker
        t0 = time.perf_counter()
        transcript = _Transcript()
        trace = observe.start_attempt(self._tracer, inp.ref, self._config.run_id, inp.base_sha)
        usage = Usage()
        turns = 0
        box: Box | None = None
        box_id = ""

        def finish(
            stop: StopReason,
            patch: str | None = None,
            prediction: Prediction | None = None,
            rationale: str = "",
            error: str = "",
        ) -> WorkerOutput:
            transcript.add(kind="end", stop=str(stop), turns=turns, error=error)
            trace.end(stop, patch=patch, prediction=prediction, error=error)
            return WorkerOutput(
                patch=patch or None,
                prediction=prediction,
                rationale=rationale,
                stop_reason=stop,
                turns=turns,
                usage=usage,
                wall_s=time.perf_counter() - t0,
                error=error,
                transcript=transcript.text(),
                box_id=box_id,
            )

        try:
            box = self._boxes.create(
                name=f"worker-{self._config.run_id}-{inp.ref.dirname}", role="worker"
            )
            box_id = box.box_id
            transcript.add(kind="box", box_id=box_id)
            trace.box(box_id)
            python = self._prepare_box(box, inp)
        except BoxError as e:
            if box is not None:
                _release(box)
            return finish(StopReason.BOX_ERROR, error=str(e))

        ctx = tools.ToolContext(box=box, target=inp.target)
        messages: list[Message] = [
            {"role": "system", "content": prompt.system_prompt(inp.prompt, python)},
            {
                "role": "user",
                "content": prompt.initial_user_message(
                    inp.target,
                    inp.base_sha,
                    inp.docs,
                    inp.history,
                    inp.ref.number,
                    self._config.referee.pairs,
                ),
            },
        ]
        recent: list[str] = []
        idle_turns = 0
        turn_walls: list[float] = []
        # Set when the turn about to run has been announced as the last one. The
        # worker is never told its budget; it is told when the budget is spent.
        last_turn: StopReason | None = None

        try:
            while True:
                if usage.prompt_tokens > cfg.max_input_tokens:
                    return finish(StopReason.MAX_INPUT_TOKENS, patch=self._collect_patch(box))
                if last_turn is not None:
                    # The announced last turn came back without a submit.
                    return finish(last_turn, patch=self._collect_patch(box))
                remaining_s = cfg.max_seconds - (time.perf_counter() - t0)
                if turns + 1 >= cfg.max_turns:
                    last_turn = StopReason.MAX_TURNS
                elif remaining_s < _time_margin(turn_walls):
                    last_turn = StopReason.MAX_SECONDS
                if last_turn is not None:
                    messages.append({"role": "user", "content": prompt.LAST_TURN})
                    transcript.add(kind="last_turn", turn=turns + 1, cap=str(last_turn))
                specs = tools.TOOL_SPECS if last_turn is None else tools.SUBMIT_ONLY_SPECS

                turns += 1
                turn_t0 = time.perf_counter()
                try:
                    resp = self._model.complete(messages, specs, cache_key=inp.cache_key)
                except ModelError as e:
                    return finish(
                        StopReason.MODEL_ERROR, patch=self._collect_patch(box), error=str(e)
                    )
                usage = usage + resp.usage
                # Before the append, so the traced input is what was sent.
                trace.model_turn(turns, messages, resp)
                messages.append(resp.message)
                transcript.add(
                    kind="model",
                    turn=turns,
                    latency_s=round(resp.latency_s, 3),
                    usage=resp.usage.to_dict(),
                    finish=resp.finish_reason,
                    reasoning=_clip(resp.reasoning),
                    content=_clip(resp.content),
                    tool_calls=[{"name": c.name, "args": c.arguments} for c in resp.tool_calls],
                )

                if last_turn is not None:
                    # Only a submit is honoured. Anything else is noted and not run;
                    # the top of the loop then ends the attempt with the cap.
                    calls = tuple(c for c in resp.tool_calls if c.name == "submit")[:1]
                    for other in resp.tool_calls:
                        if other not in calls:
                            transcript.add(
                                kind="tool",
                                turn=turns,
                                name=other.name,
                                result="not run: last turn",
                            )
                    if not calls:
                        continue
                elif not resp.tool_calls:
                    idle_turns += 1
                    if idle_turns >= NO_PROGRESS_LIMIT:
                        return finish(StopReason.NO_PROGRESS, patch=self._collect_patch(box))
                    messages.append({"role": "user", "content": prompt.NUDGE})
                    continue
                else:
                    calls = resp.tool_calls
                idle_turns = 0

                for call in calls:
                    recent.append(_call_signature(call))
                    recent = recent[-REPEAT_LIMIT:]
                    if len(recent) == REPEAT_LIMIT and len(set(recent)) == 1:
                        return finish(StopReason.REPEATED_TOOL_CALL, patch=self._collect_patch(box))
                    result = tools.execute(ctx, call.name, call.arguments)
                    transcript.add(
                        kind="tool", turn=turns, name=call.name, result=_clip(result.text)
                    )
                    trace.tool(turns, call.name, call.arguments, result.text)
                    messages.append(
                        {"role": "tool", "tool_call_id": call.id, "content": result.text}
                    )
                    if result.ended:
                        patch = self._collect_patch(box)
                        prediction = (
                            None if result.prediction is None else Prediction(result.prediction)
                        )
                        return finish(
                            StopReason.SUBMITTED if last_turn is None else StopReason.LAST_TURN,
                            patch=patch,
                            prediction=prediction,
                            rationale=result.rationale,
                        )
                turn_walls.append(time.perf_counter() - turn_t0)
        except BoxError as e:
            return finish(StopReason.BOX_ERROR, error=str(e))
        finally:
            _release(box)
