from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Sequence

from hypersonic.context.compact import COMPACT_AT, RETRY_AFTER_TOKENS, WINDOW_WANTED, compact, due
from hypersonic.context.prefix import opening
from hypersonic.engine.events import POST_TOOL, PRE_TOOL, STOP, TURN_START, Hooks
from hypersonic.engine.session import PREVIEW_CHARS, Session, harness_note
from hypersonic.policy.rules import ALLOW, DENY, Policy
from hypersonic.providers.base import (
    STOP_END,
    STOP_LENGTH,
    STOP_PAUSE,
    STOP_REFUSAL,
    Message,
    Provider,
    ToolCall,
    ToolSpec,
    text_block,
)
from hypersonic.rag.retrieve import attach
from hypersonic.tools.base import Tool, ToolContext, ToolError, invalid

MAX_STEPS = 200
STOP_MAX_STEPS = "max_steps"
STOP_ABORTED = "aborted"
STOP_ERROR = "error"
STOP_COMMAND = "command"
DENIED = "Denied by the user. Don't retry this call unchanged."
SMALL_WINDOW = (
    "{model} is loaded with room for only {window:,} tokens, so it forgets the start of longer chats without saying so. "
    "Quit Ollama and start it again with OLLAMA_CONTEXT_LENGTH={wanted} to fix that."
)
TRUNCATED = "Not run: the reply hit the output limit and this input was cut off. Retry with a smaller input."
DECLINED = "Not run: the model declined this turn."
INTERRUPTED = "Interrupted by the user."
EMPTY_RETRIES = 1
EMPTY_NUDGE = "Your last reply was empty. Answer the user now from what you have, or say what is blocking you."
EMPTY_REPLY = "The model sent back an empty reply, even after a retry. Ask again, or pick another model with /model."
DROPPED = "Stopped. {count} queued message(s) were not sent."

Emit = Callable[[dict[str, Any]], Awaitable[None]]
Ask = Callable[[ToolCall], Awaitable[bool]]


@dataclass
class Agent:
    provider: Provider
    session: Session
    ctx: ToolContext
    tools: dict[str, Tool]
    system: str
    emit: Emit
    ask: Ask
    policy: Policy = field(default_factory=Policy)
    trusted: bool = False
    check: str = ""
    hooks: Hooks = field(default_factory=Hooks)
    model: str = ""
    effort: str = ""
    window_told: int = 0

    @property
    def specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self.tools.values()]


def result(call: ToolCall, content: str, error: bool = False) -> dict[str, Any]:
    block = {"type": "tool_result", "tool_use_id": call.id, "content": content}
    if error:
        block["is_error"] = True
    return block


def results_message(
    results: list[dict[str, Any]], queued: Sequence[str] = (), extra: Sequence[dict[str, Any]] = ()
) -> Message:
    return {"role": "user", "content": results + list(extra) + [text_block(text) for text in queued]}


def close_open_calls(session: Session) -> None:
    last = session.messages[-1] if session.messages else {"role": ""}
    if last["role"] != "assistant":
        return
    open_calls = [
        ToolCall(block["id"], block["name"], block["input"])
        for block in last["content"]
        if block.get("type") == "tool_use"
    ]
    if open_calls:
        session.append(results_message([result(call, INTERRUPTED, error=True) for call in open_calls]))


def malformed(tool: Tool | None, call: ToolCall) -> dict[str, Any] | None:
    if tool is None:
        return result(call, f"Unknown tool {call.name}.", error=True)
    problem = invalid(tool.spec.schema, call.input)
    if problem:
        report = {"INVALID_JSON": json.dumps(call.input), "problem": problem}
        return result(call, json.dumps(report), error=True)
    return None


async def attempt(agent: Agent, tool: Tool, call: ToolCall) -> dict[str, Any]:
    try:
        verdict, reason = agent.policy.decide(call, agent.ctx.workspace, agent.ctx.cwd or agent.ctx.workspace)
    except Exception as error:
        return result(call, f"{type(error).__name__}: {error}", error=True)
    vetoed = await agent.hooks.first(PRE_TOOL, agent, call)
    if vetoed:
        return result(call, vetoed, error=True)
    if verdict == DENY:
        return result(call, reason, error=True)
    if verdict != ALLOW and not await agent.ask(call):
        return result(call, DENIED, error=True)
    try:
        return result(call, await tool.run(agent.ctx, call.input))
    except ToolError as error:
        return result(call, str(error), error=True)
    except Exception as error:
        return result(call, f"{type(error).__name__}: {error}", error=True)


async def run_one(agent: Agent, call: ToolCall) -> dict[str, Any]:
    tool = agent.tools.get(call.name)
    started = time.monotonic()
    shown = call.input if isinstance(call.input, dict) else {}
    await agent.emit({"type": "tool.call", "id": call.id, "name": call.name, "input": shown})
    block = malformed(tool, call) or await attempt(agent, tool, call)
    await agent.hooks.fire(POST_TOOL, agent, call, block)
    took = round(time.monotonic() - started, 1)
    preview = str(block["content"])[:PREVIEW_CHARS]
    await agent.emit({"type": "tool.result", "id": call.id, "ok": "is_error" not in block, "seconds": took, "preview": preview})
    return block


async def run_tools(agent: Agent, calls: list[ToolCall]) -> list[dict[str, Any]]:
    results: list[Any] = [None] * len(calls)

    async def fill(index: int) -> None:
        results[index] = await run_one(agent, calls[index])

    batch: list[Awaitable[None]] = []
    for index, call in enumerate(calls):
        tool = agent.tools.get(call.name)
        if tool and isinstance(call.input, dict) and tool.read_only(call.input):
            batch.append(fill(index))
            continue
        await asyncio.gather(*batch)
        batch = []
        await fill(index)
    await asyncio.gather(*batch)
    return results


def retrieved(agent: Agent, text: str) -> list[dict[str, Any]]:
    ctx = agent.ctx
    if not (ctx.rag_on and ctx.store):
        return []
    found = attach(ctx.store, text, ctx.scopes, ctx.attached, asked=ctx.rag_once)
    return [text_block(found)] if found else []


async def maybe_compact(agent: Agent, resume: bool) -> None:
    session = agent.session
    if not due(session) or session.last.prompt < session.compact_retry_at:
        return
    try:
        done = await compact(agent, resume)
    except Exception as error:
        done = False
        await agent.emit({"type": "error", "message": f"Compaction failed. {type(error).__name__}: {error}"})
    # a failed attempt costs a full-size request, so wait for the context to grow before trying again
    session.compact_retry_at = 0 if done else session.last.prompt + RETRY_AFTER_TOKENS


async def run_turn(agent: Agent, text: str, extra: Sequence[dict[str, Any]] = ()) -> str:
    session = agent.session
    stop, final, blanks, dropped = STOP_MAX_STEPS, "", 0, 0

    async def on_text(delta: str) -> None:
        await agent.emit({"type": "text", "delta": delta})

    await agent.emit({"type": "turn.start"})
    try:
        # a server that died mid-tool leaves a call with no result, which no provider accepts
        close_open_calls(session)
        first = [] if session.messages else opening(agent.ctx.workspace, agent.trusted)
        notes = [text_block(note) for note in session.take_system()]
        content = first + list(extra) + [text_block(text)] + retrieved(agent, text) + notes
        session.append({"role": "user", "content": content})
        await agent.hooks.fire(TURN_START, agent)
        for _ in range(MAX_STEPS):
            reply = await agent.provider.stream(agent.system, agent.specs, session.messages, on_text)
            session.record(reply.usage)
            window = getattr(agent.provider, "window", 0)
            if window and window < COMPACT_AT and window != agent.window_told:
                agent.window_told = window
                told = SMALL_WINDOW.format(model=agent.model, window=window, wanted=WINDOW_WANTED)
                await agent.emit({"type": "notice", "text": told, "error": True})
            totals = {"limit": min(COMPACT_AT, window or COMPACT_AT), "steps": session.steps, "spent": session.tokens}
            await agent.emit({"type": "usage", **asdict(reply.usage), "prompt": reply.usage.prompt, "model": reply.model, **totals})
            said_nothing = not reply.tool_calls and not reply.text.strip()
            if said_nothing and reply.stop == STOP_REFUSAL:
                stop = STOP_REFUSAL
                break
            if said_nothing and reply.stop != STOP_PAUSE:
                blanks += 1
                if blanks > EMPTY_RETRIES:
                    stop = STOP_ERROR
                    await agent.emit({"type": "error", "message": EMPTY_REPLY})
                    break
                session.append({"role": "user", "content": [text_block(harness_note(EMPTY_NUDGE))]})
                continue
            blanks = 0
            if not said_nothing:
                session.append({"role": "assistant", "content": reply.content})
            if reply.stop == STOP_PAUSE:
                continue
            calls = reply.tool_calls
            if reply.stop == STOP_REFUSAL or not calls:
                if calls:
                    session.append(results_message([result(call, DECLINED, error=True) for call in calls]))
                feedback = await agent.hooks.first(STOP, agent, reply) if reply.stop == STOP_END else None
                if feedback:
                    session.append({"role": "user", "content": [text_block(harness_note(feedback))]})
                    continue
                stop, final = reply.stop, reply.text
                break
            if reply.stop == STOP_LENGTH:
                results = [result(call, TRUNCATED, error=True) for call in calls]
            else:
                results = await run_tools(agent, calls)
            session.append(results_message(results, session.drain(), session.drain_extra()))
            await maybe_compact(agent, resume=True)
    except asyncio.CancelledError:
        close_open_calls(session)
        dropped = len(session.drain())
        session.drain_extra()
        stop = STOP_ABORTED
    except Exception as error:
        close_open_calls(session)
        stop = STOP_ERROR
        await agent.emit({"type": "error", "message": f"{type(error).__name__}: {error}"})
    finally:
        agent.ctx.rag_once = False
    if dropped:
        await agent.emit({"type": "notice", "text": DROPPED.format(count=dropped)})
    await agent.emit({"type": "turn.end", "stop": stop, "text": final})
    if stop not in (STOP_ABORTED, STOP_ERROR):
        await maybe_compact(agent, resume=False)
    return final
