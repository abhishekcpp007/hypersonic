from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from hypersonic.engine.events import POST_TOOL, PRE_TOOL, Hooks
from hypersonic.engine.session import harness_note

WARN_AT = 0.8
WARNING = "About 80% of this session's budget is used. Finish the most important part and wrap up."
OVER = "The session budget is used up, so no more tools will run. Reply now with a short summary of where things stand."
BLOCK_LIMIT = 3
SPENT = "The session budget is used up and the model kept calling tools, so the turn was ended."


class BudgetSpent(Exception):
    pass


@dataclass
class Budget:
    steps: int | None = None
    seconds: int | None = None
    tokens: int | None = None

    @property
    def active(self) -> bool:
        return any(limit is not None for limit in (self.steps, self.seconds, self.tokens))


def register(hooks: Hooks, budget: Budget) -> None:
    if not budget.active:
        return
    started = time.monotonic()
    state = {"warned": False, "blocked": 0}

    def used(agent: Any) -> float:
        spent = {"steps": agent.session.steps, "seconds": time.monotonic() - started, "tokens": agent.session.tokens}
        limits = {"steps": budget.steps, "seconds": budget.seconds, "tokens": budget.tokens}
        ratios = [spent[name] / limit if limit else float("inf") for name, limit in limits.items() if limit is not None]
        return max(ratios) if ratios else 0.0

    async def gate(agent: Any, call: Any) -> str | None:
        if used(agent) < 1:
            return None
        state["blocked"] += 1
        if state["blocked"] > BLOCK_LIMIT:
            raise BudgetSpent(SPENT)
        return OVER

    async def warn(agent: Any, call: Any, block: dict[str, Any]) -> None:
        if not state["warned"] and used(agent) >= WARN_AT:
            state["warned"] = True
            agent.session.queue.append(harness_note(WARNING))

    hooks.on(PRE_TOOL, gate)
    hooks.on(POST_TOOL, warn)
