"""A stand-in model for stress tests. It does what the newest user message orders, so no real model is needed.

Copy this file into <HYPERSONIC_HOME>/providers/ and pick the model `chaos/model`. evals/stress/run.py does that itself.

Orders: echo X | steps N | hold S | sleep | slowtext | shell CMD | files JSON | web URL | garbage | parallel N |
        big N | forever | blank | raise | partial
"""
from __future__ import annotations

import asyncio
import itertools
import json
from typing import Any

from hypersonic.providers.base import STOP_END, STOP_LENGTH, STOP_TOOLS, Reply, ToolCall, Usage, Vendor

HARNESS = ("<harness>", "<env", "<instructions", "<context>", "<handoff>")
COMPACT_ASK = "Write a handoff note"
OVER_THE_LIMIT = 31_000
SLOW_PIECES, SLOW_SECONDS = 200, 0.02
# an odd number, so the test runner can find and clean up exactly these processes and no one else's
LONG_NAP = 613
GARBAGE = [
    ("nope", {}),
    ("shell", {"command": 5}),
    ("shell", {}),
    ("files", {"command": "view"}),
    ("files", {"command": "create", "path": "a\x00b", "file_text": "x"}),
    ("files", {"command": "explode", "path": "x"}),
    ("files", {"command": "create", "path": "~nosuchuser_zz/x", "file_text": "x"}),
    ("files", {"command": "view", "path": "x", "view_range": "all"}),
    ("files", {"command": "insert", "path": "missing.txt", "insert_line": -4, "insert_text": "x"}),
    ("web", {"url": "http://[::1"}),
    ("web", {"url": 7}),
    ("search", {"query": None}),
    ("shell", {"command": "echo ok", "timeout": "soon"}),
    ("files", {"command": "create", "path": "big/" + "d/" * 60 + "f.txt", "file_text": "x"}),
    ("shell", {"command": "echo " + "y" * 200_000}),
]
ids = itertools.count(1)


def say(text: str, usage: Usage | None = None) -> Reply:
    return Reply([{"type": "text", "text": text}], text, [], STOP_END, usage or Usage(input=50, output=5), "chaos")


def call(*calls: tuple[str, Any], usage: Usage | None = None, stop: str = STOP_TOOLS) -> Reply:
    made = [ToolCall(f"chaos_{next(ids)}", name, args) for name, args in calls]
    blocks = [{"type": "tool_use", "id": item.id, "name": item.name, "input": item.input} for item in made]
    return Reply(blocks, "", made, stop, usage or Usage(input=50, cache_read=100, output=5), "chaos")


def blocks_of(message: dict[str, Any]) -> list[dict[str, Any]]:
    content = message["content"]
    return content if isinstance(content, list) else [{"type": "text", "text": content}]


def newest_order(messages: list[dict[str, Any]]) -> tuple[int, str]:
    """Where the newest thing the user typed is, and what it says. Harness notes and tool results don't count."""
    for position in range(len(messages) - 1, -1, -1):
        if messages[position]["role"] != "user":
            continue
        texts = [b["text"] for b in blocks_of(messages[position]) if b.get("type") == "text" and not b["text"].lstrip().startswith(HARNESS)]
        if texts:
            return position, texts[-1]
    return -1, ""


def last_result(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        found = [b for b in blocks_of(message) if b.get("type") == "tool_result"]
        if found:
            return "\n".join(str(b["content"]) for b in found)
    return ""


class Chaos:
    model = "model"

    def __init__(self) -> None:
        self.key: tuple[int, str] = (-2, "")
        self.step = 0

    async def stream(self, system: str, tools: list, messages: list[dict[str, Any]], on_text: Any) -> Reply:
        position, text = newest_order(messages)
        if text.startswith(COMPACT_ASK):
            return say("Goal: finish the stress run. Decisions: none. Current state: running. Files touched: none. Next steps: carry on.")
        # after a compaction the order is gone from history, so the plan in progress simply continues
        if text and (position, text) != self.key and not (self.step and position < self.key[0]):
            self.key, self.step = (position, text), 0
        self.step += 1
        verb, _, rest = self.key[1].partition(" ")
        return await getattr(self, f"do_{verb}", self.do_echo)(rest, messages, on_text)

    async def do_echo(self, rest: str, messages: list, on_text: Any) -> Reply:
        await on_text(rest)
        return say(rest)

    async def do_steps(self, rest: str, messages: list, on_text: Any) -> Reply:
        total = int(rest)
        if self.step > total:
            return say(f"done {total}")
        return call(("shell", {"command": f"echo step-{self.step} >> steps.txt"}))

    async def do_hold(self, rest: str, messages: list, on_text: Any) -> Reply:
        return call(("shell", {"command": f"sleep {rest}"})) if self.step == 1 else say("held")

    async def do_sleep(self, rest: str, messages: list, on_text: Any) -> Reply:
        return call(("shell", {"command": f"sleep {LONG_NAP}"})) if self.step == 1 else say("woke")

    async def do_slowtext(self, rest: str, messages: list, on_text: Any) -> Reply:
        for number in range(SLOW_PIECES):
            await on_text(f"piece {number} ")
            await asyncio.sleep(SLOW_SECONDS)
        return say("slow text finished")

    async def do_shell(self, rest: str, messages: list, on_text: Any) -> Reply:
        return call(("shell", {"command": rest})) if self.step == 1 else say(last_result(messages))

    async def do_files(self, rest: str, messages: list, on_text: Any) -> Reply:
        return call(("files", json.loads(rest))) if self.step == 1 else say(last_result(messages))

    async def do_web(self, rest: str, messages: list, on_text: Any) -> Reply:
        return call(("web", {"url": rest})) if self.step == 1 else say(last_result(messages))

    async def do_garbage(self, rest: str, messages: list, on_text: Any) -> Reply:
        rounds = 3
        return call(*GARBAGE) if self.step <= rounds else say("survived the garbage")

    async def do_parallel(self, rest: str, messages: list, on_text: Any) -> Reply:
        if self.step > 1:
            return say(last_result(messages)[:200])
        return call(*[("files", {"command": "view", "path": "."}) for _ in range(int(rest))])

    async def do_big(self, rest: str, messages: list, on_text: Any) -> Reply:
        if self.step > int(rest):
            return say("big done")
        return call(("shell", {"command": f"echo big-{self.step} >> big.txt"}), usage=Usage(input=OVER_THE_LIMIT, output=5))

    async def do_forever(self, rest: str, messages: list, on_text: Any) -> Reply:
        return call(("shell", {"command": "echo again >> forever.txt"}))

    async def do_blank(self, rest: str, messages: list, on_text: Any) -> Reply:
        return Reply([], "", [], STOP_END, Usage(input=50), "chaos")

    async def do_raise(self, rest: str, messages: list, on_text: Any) -> Reply:
        raise RuntimeError("the stand-in model was told to fail")

    async def do_partial(self, rest: str, messages: list, on_text: Any) -> Reply:
        if self.step == 1:
            return call(("shell", {"command": "touch should-not-exist.txt"}), stop=STOP_LENGTH)
        return say("partial done")


def register(vendors: dict[str, Vendor]) -> None:
    vendors["chaos"] = Vendor(lambda name, effort, home: Chaos(), lambda home: [{"name": "chaos/model", "summary": "stand-in for stress tests"}])
