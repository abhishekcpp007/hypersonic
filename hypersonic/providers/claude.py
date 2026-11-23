from __future__ import annotations

import anthropic

from hypersonic.providers.base import (
    STOP_END,
    STOP_LENGTH,
    STOP_PAUSE,
    STOP_REFUSAL,
    STOP_TOOLS,
    Message,
    OnText,
    Reply,
    ToolCall,
    ToolSpec,
    Usage,
)

MAX_TOKENS = 64000
MAX_JSON_RETRIES = 2
FALLBACK_BETA = "server-side-fallback-2026-07-01"
EPHEMERAL = {"type": "ephemeral"}
AUTH_HINT = "resolve authentication"
NO_CREDENTIALS = "No Anthropic credentials found. Set ANTHROPIC_API_KEY, then run `hypersonic stop` and try again."
STOPS = {
    "end_turn": STOP_END,
    "stop_sequence": STOP_END,
    "tool_use": STOP_TOOLS,
    "max_tokens": STOP_LENGTH,
    "model_context_window_exceeded": STOP_LENGTH,
    "pause_turn": STOP_PAUSE,
    "refusal": STOP_REFUSAL,
}


class ClaudeProvider:
    def __init__(
        self,
        model: str,
        effort: str,
        client: anthropic.AsyncAnthropic | None = None,
        fallbacks: bool = True,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.model = model
        self.effort = effort
        self.fallbacks = fallbacks
        if client is None:
            options = {name: value for name, value in (("base_url", base_url), ("api_key", api_key)) if value}
            client = anthropic.AsyncAnthropic(**options)
        self.client = client

    def request(self, system: str, tools: list[ToolSpec], messages: list[Message]) -> dict:
        request = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "system": [{"type": "text", "text": system, "cache_control": EPHEMERAL}],
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.schema,
                    "eager_input_streaming": True,
                }
                for tool in tools
            ],
            "messages": messages,
            "cache_control": EPHEMERAL,
            "output_config": {"effort": self.effort},
        }
        if self.fallbacks:
            request["betas"] = [FALLBACK_BETA]
            request["fallbacks"] = "default"
        return request

    def open_stream(self, request: dict):
        endpoint = self.client.beta.messages if self.fallbacks else self.client.messages
        return endpoint.stream(**request)

    async def final_message(self, request: dict, on_text: OnText):
        failures = 0
        while True:
            try:
                async with self.open_stream(request) as stream:
                    async for event in stream:
                        if event.type == "text":
                            await on_text(event.text)
                    return await stream.get_final_message()
            except ValueError:
                failures += 1
                if failures > MAX_JSON_RETRIES:
                    raise
            except TypeError as error:
                if AUTH_HINT in str(error):
                    raise RuntimeError(NO_CREDENTIALS) from error
                raise

    async def stream(
        self, system: str, tools: list[ToolSpec], messages: list[Message], on_text: OnText
    ) -> Reply:
        final = await self.final_message(self.request(system, tools, messages), on_text)
        content = final.to_dict(mode="json")["content"]
        usage = final.usage
        return Reply(
            content=content,
            text="".join(block["text"] for block in content if block["type"] == "text"),
            tool_calls=[
                ToolCall(block["id"], block["name"], block["input"])
                for block in content
                if block["type"] == "tool_use"
            ],
            stop=STOPS.get(final.stop_reason, STOP_LENGTH),
            model=final.model or "",
            usage=Usage(
                input=usage.input_tokens or 0,
                cache_read=usage.cache_read_input_tokens or 0,
                cache_write=usage.cache_creation_input_tokens or 0,
                output=usage.output_tokens or 0,
            ),
        )
