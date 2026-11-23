from __future__ import annotations

from dataclasses import replace
from typing import Any

from hypersonic.engine.events import POST_TOOL, STOP, TURN_START, Hooks
from hypersonic.policy.rules import read_only, split
from hypersonic.tools import shell

DONE_ROUNDS = 2
FAILED = "The check command `{check}` failed. Fix it before you finish, or say plainly why you can't.\n{output}"


def changes_things(call: Any) -> bool:
    if call.name == "files":
        return call.input.get("command") != "view"
    if call.name != "shell":
        return False
    parsed = split(call.input.get("command", ""))
    return parsed is None or parsed[1] or not all(read_only(words) for words in parsed[0])


def register(hooks: Hooks) -> None:
    state = {"dirty": False, "rounds": 0}

    async def reset(agent: Any) -> None:
        state.update(dirty=False, rounds=0)

    async def watch(agent: Any, call: Any, block: dict[str, Any]) -> None:
        if "is_error" not in block and changes_things(call):
            state["dirty"] = True

    async def check(agent: Any, reply: Any) -> str | None:
        ready = agent.check and state["dirty"] and state["rounds"] < DONE_ROUNDS
        if not ready:
            return None
        # from the workspace root, and on a copy, so the check neither depends on nor moves the agent's folder
        output = await shell.run(replace(agent.ctx, cwd=agent.ctx.workspace), {"command": agent.check})
        state["dirty"] = False
        if not shell.failed(output):
            return None
        state["rounds"] += 1
        return FAILED.format(check=agent.check, output=output)

    hooks.on(TURN_START, reset)
    hooks.on(POST_TOOL, watch)
    hooks.on(STOP, check)
