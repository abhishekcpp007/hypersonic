import asyncio
from types import SimpleNamespace

from hypersonic.providers.base import STOP_LENGTH, STOP_TOOLS, ToolSpec
from hypersonic.providers.claude import EPHEMERAL, ClaudeProvider

THINKING = {"type": "thinking", "thinking": "", "signature": "opaque=="}
TEXT = {"type": "text", "text": "Reading it."}
CALL = {"type": "tool_use", "id": "toolu_1", "name": "shell", "input": {"command": "ls"}}


class StubStream:
    def __init__(self, final):
        self.final = final

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def __aiter__(self):
        yield SimpleNamespace(type="text", text="Reading ")
        yield SimpleNamespace(type="input_json", partial_json="{")
        yield SimpleNamespace(type="text", text="it.")

    async def get_final_message(self):
        return self.final


class StubClient:
    def __init__(self, final):
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self.stream))
        self.final = final

    def stream(self, **request):
        self.requests.append(request)
        return StubStream(self.final)


def reply_for(messages):
    final = SimpleNamespace(
        to_dict=lambda mode: {"content": [THINKING, TEXT, CALL]},
        stop_reason="tool_use",
        model="claude-opus-5-5",
        usage=SimpleNamespace(
            input_tokens=12, cache_read_input_tokens=900, cache_creation_input_tokens=40, output_tokens=None
        ),
    )
    client = StubClient(final)
    provider = ClaudeProvider("claude-opus-5-5", "medium", client)
    spec = ToolSpec("shell", "Run a command.", {"type": "object", "properties": {}})
    seen = []

    async def on_text(delta):
        seen.append(delta)

    reply = asyncio.run(provider.stream("SYSTEM", [spec], messages, on_text))
    return reply, client.requests[0], seen


def test_request_shape_keeps_the_prefix_cacheable():
    messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    _, request, _ = reply_for(messages)
    assert request["model"] == "claude-opus-5-5"
    assert request["system"] == [{"type": "text", "text": "SYSTEM", "cache_control": EPHEMERAL}]
    assert request["cache_control"] == EPHEMERAL
    assert request["output_config"] == {"effort": "medium"}
    assert request["fallbacks"] == "default"
    assert request["messages"] is messages
    assert request["tools"] == [
        {
            "name": "shell",
            "description": "Run a command.",
            "input_schema": {"type": "object", "properties": {}},
            "eager_input_streaming": True,
        }
    ]
    assert "thinking" not in request


def test_a_full_context_window_or_an_unknown_stop_never_runs_the_tools(monkeypatch):
    for reason in ("model_context_window_exceeded", "something_new"):
        final = SimpleNamespace(
            to_dict=lambda mode: {"content": [CALL]}, stop_reason=reason, model="claude-opus-5-5",
            usage=SimpleNamespace(input_tokens=1, cache_read_input_tokens=0, cache_creation_input_tokens=0, output_tokens=1),
        )

        async def on_text(delta):
            pass

        provider = ClaudeProvider("claude-opus-5-5", "medium", StubClient(final))
        assert asyncio.run(provider.stream("S", [], [], on_text)).stop == STOP_LENGTH


def test_reply_keeps_raw_blocks_and_maps_usage():
    reply, _, seen = reply_for([])
    assert reply.content == [THINKING, TEXT, CALL]
    assert reply.text == "Reading it."
    assert seen == ["Reading ", "it."]
    assert reply.stop == STOP_TOOLS
    assert reply.model == "claude-opus-5-5"
    assert [(call.id, call.name, call.input) for call in reply.tool_calls] == [("toolu_1", "shell", {"command": "ls"})]
    assert reply.usage.prompt == 952
    assert reply.usage.output == 0
