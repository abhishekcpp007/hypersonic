import asyncio
from types import SimpleNamespace

from hypersonic.providers.base import STOP_END, STOP_LENGTH, STOP_TOOLS, ToolSpec
from hypersonic.providers.openai_compat import OpenAICompatProvider, to_messages, to_tools


def chunk(content=None, tool=None, finish=None, usage=None):
    delta = SimpleNamespace(content=content, tool_calls=None)
    if tool is not None:
        idx, cid, name, args = tool
        fn = SimpleNamespace(name=name, arguments=args)
        delta.tool_calls = [SimpleNamespace(index=idx, id=cid, function=fn)]
    choice = SimpleNamespace(delta=delta, finish_reason=finish)
    return SimpleNamespace(choices=[choice], usage=usage)


class FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        async def gen():
            for c in self._chunks:
                yield c
        return gen()


class FakeClient:
    def __init__(self, chunks):
        self.chunks = chunks
        self.request = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        self.request = kwargs
        return FakeStream(self.chunks)


def run(provider, system, tools, messages):
    seen = []

    async def on_text(t):
        seen.append(t)

    reply = asyncio.run(provider.stream(system, tools, messages, on_text))
    return reply, seen


def test_to_messages_maps_tool_use_and_results():
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}, {"type": "tool_use", "id": "c1", "name": "shell", "input": {"command": "ls"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "a.txt"}]},
    ]
    out = to_messages(msgs)
    assert out[0] == {"role": "user", "content": "hi"}
    assert out[1]["role"] == "assistant" and out[1]["tool_calls"][0]["function"]["name"] == "shell"
    assert out[2] == {"role": "tool", "tool_call_id": "c1", "content": "a.txt"}


def test_image_block_becomes_image_url():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image", "source": {"media_type": "image/png", "data": "AAA"}}]}]
    parts = to_messages(msgs)[0]["content"]
    assert parts[0] == {"type": "text", "text": "look"}
    assert parts[1]["image_url"]["url"] == "data:image/png;base64,AAA"


def test_to_tools_is_openai_function_shape():
    spec = ToolSpec("shell", "run it", {"type": "object", "properties": {"command": {"type": "string"}}})
    assert to_tools([spec]) == [{"type": "function", "function": {"name": "shell", "description": "run it", "parameters": spec.schema}}]


def test_stream_collects_text_tool_calls_and_usage():
    chunks = [
        chunk(content="Let me "),
        chunk(content="check."),
        chunk(tool=(0, "call_1", "shell", '{"command": "l')),
        chunk(tool=(0, None, None, 's -la"}')),
        chunk(finish="tool_calls", usage=SimpleNamespace(prompt_tokens=42, completion_tokens=8)),
    ]
    provider = OpenAICompatProvider("qwen", "http://x/v1", "k", client=FakeClient(chunks))
    reply, seen = run(provider, "SYS", [ToolSpec("shell", "d", {"type": "object", "properties": {}})], [{"role": "user", "content": "go"}])
    assert seen == ["Let me ", "check."]
    assert reply.text == "Let me check."
    assert reply.stop == STOP_TOOLS
    assert len(reply.tool_calls) == 1
    assert reply.tool_calls[0].name == "shell" and reply.tool_calls[0].input == {"command": "ls -la"}
    assert reply.usage.input == 42 and reply.usage.output == 8
    assert provider.client.request["messages"][0] == {"role": "system", "content": "SYS"}
    assert provider.client.request["tools"][0]["function"]["name"] == "shell"


def piece(index, cid, name, args):
    return SimpleNamespace(index=index, id=cid, function=SimpleNamespace(name=name, arguments=args))


def many(*pieces, finish="tool_calls"):
    delta = SimpleNamespace(content=None, tool_calls=list(pieces))
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish)], usage=None)


def test_tool_calls_are_put_together_whatever_the_server_leaves_out():
    by_id = [many(piece(None, "a", "shell", '{"command":')), many(piece(None, "b", "files", '{"command": "view", ')),
             many(piece(None, "a", None, ' "ls"}')), many(piece(None, "b", None, '"path": "x"}'))]
    bare = [many(piece(None, "a", "shell", '{"command":')), many(piece(None, None, None, ' "ls"}')),
            many(piece(None, "b", "shell", '{"command": "pwd"}'))]
    for chunks in (by_id, bare):
        reply, _ = run(OpenAICompatProvider("m", "http://x", "k", client=FakeClient(chunks)), "s", [], [])
        assert [call.id for call in reply.tool_calls] == ["a", "b"]
        assert reply.tool_calls[0].input == {"command": "ls"} and reply.stop == STOP_TOOLS
    assert reply.tool_calls[1].input == {"command": "pwd"}


def test_calls_without_an_id_get_one_that_never_repeats():
    chunks = [many(piece(0, None, "shell", '{"command": "ls"}'))]
    first, _ = run(OpenAICompatProvider("m", "http://x", "k", client=FakeClient(chunks)), "s", [], [])
    second, _ = run(OpenAICompatProvider("m", "http://x", "k", client=FakeClient(chunks)), "s", [], [])
    assert first.tool_calls[0].id != second.tool_calls[0].id
    assert first.content[0]["id"] == first.tool_calls[0].id


def test_a_reply_of_only_blank_lines_counts_as_no_text():
    odd = SimpleNamespace(choices=[SimpleNamespace(delta=None, finish_reason=None)], usage=None)
    reply, _ = run(OpenAICompatProvider("m", "http://x", "k", client=FakeClient([odd, chunk(content="\n\n", finish="stop")])), "s", [], [])
    assert reply.content == [] and reply.text == "" and reply.stop == STOP_END


def test_text_blocks_keep_a_gap_between_them():
    message = {"role": "user", "content": [{"type": "text", "text": "</environment>"}, {"type": "text", "text": "hello"}]}
    assert to_messages([message]) == [{"role": "user", "content": "</environment>\n\nhello"}]


def test_the_real_window_is_asked_for_once_after_the_first_reply():
    asked = []
    provider = OpenAICompatProvider("m", "http://x", "k", client=FakeClient([chunk(content="hi", finish="stop")]), window=lambda: asked.append(1) or 4096)
    run(provider, "s", [], [])
    run(provider, "s", [], [])
    assert provider.window == 4096 and len(asked) == 1


def test_plain_text_reply_has_no_tools():
    provider = OpenAICompatProvider("qwen", "http://x/v1", "k", client=FakeClient([chunk(content="Paris."), chunk(finish="stop")]))
    reply, _ = run(provider, "SYS", [], [{"role": "user", "content": "capital of France?"}])
    assert reply.text == "Paris." and reply.stop == STOP_END and reply.tool_calls == []
    assert "tools" not in provider.client.request


def test_bad_tool_json_is_kept_not_crashed():
    provider = OpenAICompatProvider("q", "http://x/v1", "k", client=FakeClient([chunk(tool=(0, "c", "shell", "{not json")), chunk(finish="tool_calls")]))
    reply, _ = run(provider, "S", [], [{"role": "user", "content": "x"}])
    assert reply.tool_calls[0].input == {"_unparsed": "{not json"}


def test_truncated_tool_call_reports_length_not_tools():
    # a reply cut off mid tool-call must stop as LENGTH so the loop never runs the partial call
    chunks = [chunk(tool=(0, "c", "shell", '{"command": "rm -r')), chunk(finish="length")]
    provider = OpenAICompatProvider("q", "http://x/v1", "k", client=FakeClient(chunks))
    reply, _ = run(provider, "S", [], [{"role": "user", "content": "x"}])
    assert reply.stop == STOP_LENGTH
