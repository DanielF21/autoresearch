"""The totals read at the round 20 gate, computed from the run directory alone.

Counts are of facts, not decisions: how many attempts were measured, how many
passed the tests, how many cleared the noise floor. Best so far is the highest
median ratio among attempts that cleared it; the best raw ratio over every
timed attempt is shown beside it so the two cannot be confused.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass

from autoresearch import history
from autoresearch.types import Usage


@dataclass(frozen=True)
class RunStatus:
    run_id: str
    rounds_done: int
    rounds_total: int
    attempts: int
    measured: int
    no_patch: int
    duplicates: int
    tests_pass: int
    clears_noise: int
    by_stop: dict[str, int]
    best_ratio: float | None
    best_attempt: int | None
    best_raw_ratio: float | None
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
            f"run {self.run_id}: {self.rounds_done} of {self.rounds_total} rounds, "
            f"{self.attempts} attempts",
            f"measured {self.measured}, no patch {self.no_patch}, duplicates {self.duplicates}, "
            f"tests pass {self.tests_pass}, real speedups {self.clears_noise}",
        ]
        if self.best_ratio is not None:
            lines.append(
                f"best real speedup so far {self.best_ratio:.4f} (attempt {self.best_attempt:04d})"
            )
        else:
            lines.append("best real speedup so far: none")
        if self.best_raw_ratio is not None:
            lines.append(f"best raw ratio over all timed attempts {self.best_raw_ratio:.4f}")
        lines.append(
            "worker stops: " + ", ".join(f"{k} {v}" for k, v in sorted(self.by_stop.items()))
        )
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
    measured = [a for a in attempts if a.measurement is not None]
    timed = [a.measurement.speedup for a in measured if a.measurement and a.measurement.speedup]
    best, which = history.best_ratio(attempts)
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
        measured=len(measured),
        no_patch=sum(1 for a in attempts if a.skipped),
        duplicates=sum(1 for a in attempts if a.duplicate_of),
        tests_pass=sum(1 for a in measured if a.measurement and a.measurement.tests_pass),
        clears_noise=sum(1 for a in measured if a.clears_noise),
        by_stop=dict(Counter(str(a.stop_reason) for a in attempts)),
        best_ratio=best,
        best_attempt=which,
        best_raw_ratio=max(timed) if timed else None,
        worker_wall_median_s=statistics.median(r.worker_wall_s for r in rounds) if rounds else None,
        referee_wall_median_s=(
            statistics.median(r.referee_wall_s for r in rounds) if rounds else None
        ),
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
