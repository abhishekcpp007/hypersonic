from __future__ import annotations

import json
from typing import Any

from hypersonic.engine.events import POST_TOOL, TURN_START, Hooks
from hypersonic.engine.session import harness_note

REPEAT_LIMIT = 4
NUDGE = (
    f"You've made the same tool call {REPEAT_LIMIT} times in a row. Stop repeating it. "
    "Try a different approach, or tell the user what's blocking you."
)


def register(hooks: Hooks) -> None:
    state: dict[str, Any] = {"key": None, "count": 0}

    async def reset(agent: Any) -> None:
        state.update(key=None, count=0)

    async def watch(agent: Any, call: Any, block: dict[str, Any]) -> None:
        key = (call.name, json.dumps(call.input, sort_keys=True))
        state["count"] = state["count"] + 1 if key == state["key"] else 1
        state["key"] = key
        if state["count"] % REPEAT_LIMIT == 0:
            agent.session.queue.append(harness_note(NUDGE))

    hooks.on(TURN_START, reset)
    hooks.on(POST_TOOL, watch)
