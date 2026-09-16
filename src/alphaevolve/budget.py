"""Tokens: the x axis, and the budget that ends a run.

A request's ``prompt_tokens`` already include its ``cached_tokens``, and its
``completion_tokens`` already include its ``reasoning_tokens``, so the primary
measure is prompt plus completion and summing all four would count twice. The
uncached measure is the prompt tokens that were not a cache hit plus the
completion. Both are read from ``rounds.jsonl``, the same record for both arms.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from autoresearch import history
from autoresearch.types import Usage


class BudgetError(RuntimeError):
    pass


def primary(usage: Usage) -> int:
    return usage.prompt_tokens + usage.completion_tokens


def uncached(usage: Usage) -> int:
    return max(0, usage.prompt_tokens - usage.cached_tokens) + usage.completion_tokens


def run_usage(paths: history.RunPaths) -> Usage:
    total = Usage()
    for record in history.read_rounds(paths):
        total = total + record.usage
    return total


def harness_budget(run_dir: Path) -> int:
    """The primary token total of a finished or stopped harness run."""
    paths = history.RunPaths(run_dir)
    if not history.read_rounds(paths):
        raise BudgetError(f"{run_dir} has no completed rounds to take a budget from")
    return primary(run_usage(paths))


def stop_at(budget: int) -> Callable[[history.RunPaths], bool]:
    """The run loop's stop predicate: true once the run has spent ``budget``."""

    def reached(paths: history.RunPaths) -> bool:
        return primary(run_usage(paths)) >= budget

    return reached
