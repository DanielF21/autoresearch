"""The totals read at the round 20 gate, computed from the run directory alone."""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass

from autoresearch import history
from autoresearch.types import Usage, Verdict


@dataclass(frozen=True)
class RunStatus:
    run_id: str
    rounds_done: int
    rounds_total: int
    attempts: int
    by_verdict: dict[str, int]
    by_stop: dict[str, int]
    accepted: int
    best_ratio: float | None
    cumulative_ratio: float
    worker_wall_median_s: float | None
    referee_wall_median_s: float | None
    round_wall_median_s: float | None
    usage: Usage
    prompt_tokens_per_attempt: float | None
    cached_share: float | None
    cost_usd: float | None
    harness_errors: tuple[str, ...]

    def render(self) -> str:
        lines = [
            f"run {self.run_id}: {self.rounds_done} of {self.rounds_total} rounds, {self.attempts} attempts",
            f"accepted {self.accepted}; best single ratio {self.best_ratio:.4f}"
            if self.best_ratio is not None
            else "accepted 0",
            f"cumulative incumbent speedup {self.cumulative_ratio:.4f}",
            "verdicts: " + ", ".join(f"{k} {v}" for k, v in sorted(self.by_verdict.items())),
            "worker stops: " + ", ".join(f"{k} {v}" for k, v in sorted(self.by_stop.items())),
        ]
        if self.worker_wall_median_s is not None:
            lines.append(
                f"median wall per round: worker {self.worker_wall_median_s:.0f}s, "
                f"referee {self.referee_wall_median_s or 0:.0f}s, "
                f"round {self.round_wall_median_s or 0:.0f}s"
            )
        if self.prompt_tokens_per_attempt is not None:
            lines.append(
                f"tokens per attempt: {self.prompt_tokens_per_attempt:,.0f} prompt "
                f"({100 * (self.cached_share or 0):.0f} percent cached), "
                f"{self.usage.completion_tokens / max(1, self.attempts):,.0f} completion"
            )
        if self.cost_usd is not None:
            lines.append(f"inference cost so far at list price: ${self.cost_usd:.2f}")
        if self.harness_errors:
            lines.append(f"harness errors ({len(self.harness_errors)}):")
            lines += [f"  {e}" for e in self.harness_errors]
        else:
            lines.append("harness errors: none")
        return "\n".join(lines)


# List prices from the SDK's catalog on 2026-09-11, per million tokens, asap window.
PRICES_PER_M: dict[str, tuple[float, float]] = {
    "deepseek/deepseek-v4-pro-0813": (1.32, 3.96),
    "deepseek/deepseek-v4-flash-0731": (0.09, 0.18),
    "zai-org/GLM-5.3": (0.98, 3.08),
    "zai-org/GLM-5.3-Flash": (0.15, 0.50),
    "moonshotai/Kimi-K3": (3.00, 15.00),
}


def compute_status(
    paths: history.RunPaths, rounds_total: int, model: str, run_id: str
) -> RunStatus:
    attempts = history.load_history(paths)
    rounds = history.read_rounds(paths)
    by_verdict = Counter(str(a.verdict) if a.verdict else "pending" for a in attempts)
    by_stop = Counter(str(a.stop_reason) for a in attempts)
    accepted = [a for a in attempts if a.verdict == Verdict.ACCEPTED and a.result is not None]
    ratios = [a.result.median_ratio for a in accepted if a.result and a.result.median_ratio]
    cumulative = 1.0
    for r in ratios:
        cumulative *= r
    usage = Usage()
    for a in attempts:
        usage = usage + a.usage
    n = len(attempts)
    prices = PRICES_PER_M.get(model)
    cost = None
    if prices and n:
        cost = (usage.prompt_tokens * prices[0] + usage.completion_tokens * prices[1]) / 1e6
    errors = tuple(e for r in rounds for e in r.errors)
    return RunStatus(
        run_id=run_id,
        rounds_done=len(rounds),
        rounds_total=rounds_total,
        attempts=n,
        by_verdict=dict(by_verdict),
        by_stop=dict(by_stop),
        accepted=len(accepted),
        best_ratio=max(ratios) if ratios else None,
        cumulative_ratio=cumulative,
        worker_wall_median_s=statistics.median(r.worker_wall_s for r in rounds) if rounds else None,
        referee_wall_median_s=statistics.median(r.referee_wall_s for r in rounds)
        if rounds
        else None,
        round_wall_median_s=(
            statistics.median(r.worker_wall_s + r.referee_wall_s for r in rounds)
            if rounds
            else None
        ),
        usage=usage,
        prompt_tokens_per_attempt=usage.prompt_tokens / n if n else None,
        cached_share=(usage.cached_tokens / usage.prompt_tokens) if usage.prompt_tokens else None,
        cost_usd=cost,
        harness_errors=errors,
    )
