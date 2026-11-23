from __future__ import annotations

import asyncio
import json
import secrets
from typing import Any, Callable

from hypersonic.providers.base import (
    STOP_END,
    STOP_LENGTH,
    STOP_TOOLS,
    Message,
    OnText,
    Reply,
    ToolCall,
    ToolSpec,
    Usage,
)

MAX_TOKENS = 8192
ID_BYTES = 6
BETWEEN_BLOCKS = "\n\n"


def result_text(block: dict[str, Any]) -> str:
    content = block.get("content", "")
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)


def to_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.schema}}
        for t in tools
    ]


def to_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Translate the harness's Anthropic-style messages into OpenAI chat messages."""
    out: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
        if role == "system":
            out.append({"role": "system", "content": "".join(b.get("text", "") for b in blocks)})
        elif role == "assistant":
            text = "".join(b["text"] for b in blocks if b.get("type") == "text")
            calls = [
                {"id": b["id"], "type": "function",
                 "function": {"name": b["name"], "arguments": json.dumps(b["input"], ensure_ascii=False)}}
                for b in blocks if b.get("type") == "tool_use"
            ]
            msg: dict[str, Any] = {"role": "assistant", "content": text or (None if calls else "")}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        else:
            for b in blocks:
                if b.get("type") == "tool_result":
                    out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": result_text(b)})
            parts = []
            for b in blocks:
                if b.get("type") == "text":
                    parts.append({"type": "text", "text": b["text"]})
                elif b.get("type") == "image":
                    src = b["source"]
                    parts.append({"type": "image_url",
                                  "image_url": {"url": f"data:{src['media_type']};base64,{src['data']}"}})
            if parts and all(p["type"] == "text" for p in parts):
                out.append({"role": "user", "content": BETWEEN_BLOCKS.join(p["text"] for p in parts)})
            elif parts:
                out.append({"role": "user", "content": parts})
    return out


class OpenAICompatProvider:
    def __init__(
        self, model: str, base_url: str, api_key: str | None, client: Any = None, window: Callable[[], int] | None = None
    ):
        self.model = model
        self.window = 0
        self.ask_window = window
        if client is None:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(base_url=base_url, api_key=api_key or "none")
        self.client = client

    @staticmethod
    def take(call: Any, slots: list[dict[str, Any]], by_key: dict[Any, dict[str, Any]]) -> None:
        """Add one streamed piece of a tool call to the call it belongs to.

        Servers differ: most number the calls, some only give an id, a few give neither after the first piece.
        """
        key = call.index if call.index is not None else call.id
        slot = by_key.get(key) if key is not None else (slots[-1] if slots else None)
        if slot is None or (call.id and slot["id"] and call.id != slot["id"]):
            slot = {"key": key if key is not None else len(slots), "id": "", "name": "", "args": ""}
            slots.append(slot)
            by_key[slot["key"]] = slot
        if call.id:
            slot["id"] = call.id
        if call.function and call.function.name:
            slot["name"] = call.function.name
        if call.function and call.function.arguments:
            slot["args"] += call.function.arguments

    async def stream(
        self, system: str, tools: list[ToolSpec], messages: list[Message], on_text: OnText
    ) -> Reply:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + to_messages(messages),
            "max_tokens": MAX_TOKENS,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            request["tools"] = to_tools(tools)
        stream = await self.client.chat.completions.create(**request)

        text = ""
        slots: list[dict[str, Any]] = []
        by_key: dict[Any, dict[str, Any]] = {}
        finish = None
        usage = None
        answered = ""
        async for chunk in stream:
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            answered = getattr(chunk, "model", None) or answered
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if delta is not None and delta.content:
                text += delta.content
                await on_text(delta.content)
            for call in (delta.tool_calls if delta is not None else None) or []:
                self.take(call, slots, by_key)
            if choice.finish_reason:
                finish = choice.finish_reason
        if not self.window and self.ask_window:
            self.window = await asyncio.to_thread(self.ask_window)

        content: list[dict[str, Any]] = [{"type": "text", "text": text}] if text.strip() else []
        calls: list[ToolCall] = []
        numbered = all(isinstance(slot["key"], int) for slot in slots)
        for slot in sorted(slots, key=lambda slot: slot["key"]) if numbered else slots:
            cid = slot["id"] or f"call_{secrets.token_hex(ID_BYTES)}"
            try:
                args = json.loads(slot["args"] or "{}")
                if not isinstance(args, dict):
                    args = {"value": args}
            except ValueError:
                args = {"_unparsed": slot["args"]}
            content.append({"type": "tool_use", "id": cid, "name": slot["name"], "input": args})
            calls.append(ToolCall(cid, slot["name"], args))

        # length takes precedence: a reply cut off mid tool-call must not run the partial call
        text = text if text.strip() else ""
        stop = STOP_LENGTH if finish == "length" else (STOP_TOOLS if calls else STOP_END)
        return Reply(
            content=content,
            text=text,
            tool_calls=calls,
            stop=stop,
            usage=Usage(
                input=getattr(usage, "prompt_tokens", 0) or 0,
                output=getattr(usage, "completion_tokens", 0) or 0,
            ),
            model=answered,
        )
