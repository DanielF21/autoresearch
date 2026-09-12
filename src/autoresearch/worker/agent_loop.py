"""The agent loop worker: a fresh box, a conversation with tools, one diff out.

Per attempt: create a box from the image, check its repo out at the run's base
commit, make a second worktree of the same commit for the benchmark tool, upload
the guest programs and the history, then loop: ask the model, run the tools it
asks for, append the results, until it submits or a cap trips. The box is
terminated on every exit path. The patch is ``git diff`` of the repo, never
model text.

Caps and kill rules, each recorded as the stop reason:
- max_turns, max_seconds, max_input_tokens from the config
- repeated_tool_call: the same tool with the same arguments three times running
- no_progress: two consecutive turns with no tool call, after one nudge
- model_error and box_error: the platform failed after the client's retry
"""

from __future__ import annotations

import json
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autoresearch.boxes.image import GUEST_DIR, REPO_DIR
from autoresearch.boxes.protocol import Box, BoxError, BoxFactory
from autoresearch.config import RunConfig
from autoresearch.model.protocol import ChatModel, Message, ModelError, ToolCall
from autoresearch.types import Prediction, StopReason, Usage, WorkerOutput
from autoresearch.worker import prompt, tools
from autoresearch.worker.protocol import WorkerInput

GUEST_SOURCE = Path(__file__).parent.parent / "guest"
SETUP_TIMEOUT = 600
REPEAT_LIMIT = 3
NO_PROGRESS_LIMIT = 2
TRANSCRIPT_CLIP = 4000


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


def _call_signature(call: ToolCall) -> str:
    return call.name + ":" + json.dumps(call.arguments, sort_keys=True)


class AgentLoopWorker:
    def __init__(self, model: ChatModel, boxes: BoxFactory, config: RunConfig) -> None:
        self._model = model
        self._boxes = boxes
        self._config = config

    # ----- box setup -----------------------------------------------------------------

    def _prepare_box(self, box: Box, inp: WorkerInput) -> None:
        box.upload_dir(GUEST_SOURCE, GUEST_DIR)
        # The image cloned the repo at the base commit. Check it out explicitly and
        # confirm HEAD, so a stale image or a wrong config cannot go unnoticed.
        r = box.run(
            f"cd {REPO_DIR} && git checkout -q --detach {inp.base_sha}"
            f" && git rev-parse HEAD"
            f" && rm -rf {tools.BASE_DIR} && git worktree prune"
            f" && git worktree add -q --detach {tools.BASE_DIR} HEAD",
            timeout=SETUP_TIMEOUT,
        )
        if not r.ok:
            raise BoxError(f"worker box setup failed: {r.stderr[-800:]}")
        head = r.stdout.split()[0] if r.stdout.split() else ""
        if head != inp.base_sha:
            raise BoxError(f"worker box is at {head}, not the base {inp.base_sha}")
        for a in inp.history:
            d = f"{tools.HISTORY_DIR}/{a.ref.dirname}"
            box.write(f"{d}/summary.md", prompt.render_attempt(a).encode())
            if a.patch:
                box.write(f"{d}/patch.diff", a.patch.encode())
            if a.measurement is not None:
                box.write(
                    f"{d}/measurement.json", json.dumps(a.measurement.to_dict(), indent=1).encode()
                )

    def _collect_patch(self, box: Box) -> str:
        r = box.run(
            f"cd {REPO_DIR} && git add -N . && git diff --binary -- . ':(exclude){shlex.quote(tools.BASE_DIR)}'",
            timeout=120,
        )
        if not r.ok:
            raise BoxError(f"git diff failed: {r.stderr[-500:]}")
        return r.stdout

    # ----- the loop ------------------------------------------------------------------

    def attempt(self, inp: WorkerInput) -> WorkerOutput:
        cfg = self._config.worker
        t0 = time.perf_counter()
        transcript = _Transcript()
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
            self._prepare_box(box, inp)
        except BoxError as e:
            if box is not None:
                box.terminate()
            return finish(StopReason.BOX_ERROR, error=str(e))

        ctx = tools.ToolContext(box=box, target=inp.target)
        messages: list[Message] = [
            {"role": "system", "content": prompt.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": prompt.initial_user_message(
                    inp.target,
                    inp.base_sha,
                    self._config.referee.noise_floor,
                    inp.docs,
                    inp.history,
                    inp.ref.number,
                ),
            },
        ]
        recent: list[str] = []
        idle_turns = 0

        try:
            while True:
                if turns >= cfg.max_turns:
                    return finish(StopReason.MAX_TURNS, patch=self._collect_patch(box))
                if time.perf_counter() - t0 > cfg.max_seconds:
                    return finish(StopReason.MAX_SECONDS, patch=self._collect_patch(box))
                if usage.prompt_tokens > cfg.max_input_tokens:
                    return finish(StopReason.MAX_INPUT_TOKENS, patch=self._collect_patch(box))

                turns += 1
                try:
                    resp = self._model.complete(messages, tools.TOOL_SPECS, cache_key=inp.cache_key)
                except ModelError as e:
                    return finish(
                        StopReason.MODEL_ERROR, patch=self._collect_patch(box), error=str(e)
                    )
                usage = usage + resp.usage
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

                if not resp.tool_calls:
                    idle_turns += 1
                    if idle_turns >= NO_PROGRESS_LIMIT:
                        return finish(StopReason.NO_PROGRESS, patch=self._collect_patch(box))
                    messages.append({"role": "user", "content": prompt.NUDGE})
                    continue
                idle_turns = 0

                for call in resp.tool_calls:
                    recent.append(_call_signature(call))
                    recent = recent[-REPEAT_LIMIT:]
                    if len(recent) == REPEAT_LIMIT and len(set(recent)) == 1:
                        return finish(StopReason.REPEATED_TOOL_CALL, patch=self._collect_patch(box))
                    result = tools.execute(ctx, call.name, call.arguments)
                    transcript.add(
                        kind="tool", turn=turns, name=call.name, result=_clip(result.text)
                    )
                    messages.append(
                        {"role": "tool", "tool_call_id": call.id, "content": result.text}
                    )
                    if result.ended:
                        patch = self._collect_patch(box)
                        prediction = (
                            None if result.prediction is None else Prediction(result.prediction)
                        )
                        return finish(
                            StopReason.SUBMITTED,
                            patch=patch,
                            prediction=prediction,
                            rationale=result.rationale,
                        )
        except BoxError as e:
            return finish(StopReason.BOX_ERROR, error=str(e))
        finally:
            box.terminate()
