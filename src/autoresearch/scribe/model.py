"""Model construction and token accounting for the Scribe's roles.

``SailChatModel`` is typed to take the worker's config and reads four of its
fields, so a role's settings are copied into one. That is the one place to change
if the worker config does.

Cost is a bracket, as in ``autoresearch status``: Sail does not publish a cached
token rate, so the low end prices cached input at zero and the high end at the
full input rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from autoresearch.model.protocol import ChatModel
from autoresearch.orchestrator.status import PRICES_PER_M
from autoresearch.scribe.config import RoleModel
from autoresearch.types import Usage


def chat_model(role: RoleModel) -> ChatModel:
    from autoresearch.config import WorkerConfig
    from autoresearch.model.sail_model import SailChatModel

    return SailChatModel(
        WorkerConfig(
            model=role.model,
            reasoning_effort=role.reasoning_effort,
            completion_window=role.completion_window,
            max_turns=role.max_turns,
            max_seconds=role.max_seconds,
            max_input_tokens=role.max_input_tokens,
            turn_timeout=role.turn_timeout,
        )
    )


def cost_bracket(usage: Usage, model: str) -> tuple[float, float] | None:
    prices = PRICES_PER_M.get(model)
    if prices is None:
        return None
    completion = usage.completion_tokens * prices[1]
    fresh = max(0, usage.prompt_tokens - usage.cached_tokens)
    return (fresh * prices[0] + completion) / 1e6, (
        usage.prompt_tokens * prices[0] + completion
    ) / 1e6


@dataclass
class Ledger:
    """Tokens by role, summed over every model call the Scribe made."""

    by_role: dict[str, Usage] = field(default_factory=dict)
    calls: dict[str, int] = field(default_factory=dict)

    def add(self, role: str, usage: Usage) -> None:
        self.by_role[role] = self.by_role.get(role, Usage()) + usage
        self.calls[role] = self.calls.get(role, 0) + 1

    def to_dict(self, models: dict[str, RoleModel]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for role, usage in sorted(self.by_role.items()):
            bracket = cost_bracket(usage, models[role].model) if role in models else None
            out[role] = {
                "model": models[role].model if role in models else "",
                "loops": self.calls.get(role, 0),
                "usage": usage.to_dict(),
                "cost_usd_low": None if bracket is None else round(bracket[0], 4),
                "cost_usd_high": None if bracket is None else round(bracket[1], 4),
            }
        return out
