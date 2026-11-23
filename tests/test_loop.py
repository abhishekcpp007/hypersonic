import asyncio
import json

from conftest import say, use

from hypersonic.context.compact import COMPACT_AT
from hypersonic.engine.loop import EMPTY_NUDGE, EMPTY_REPLY, DECLINED, DENIED, INTERRUPTED, TRUNCATED, run_turn
from hypersonic.engine.session import Session, harness_note
from hypersonic.providers.base import STOP_END, STOP_LENGTH, STOP_PAUSE, STOP_REFUSAL, Reply


def turn(agent, text="do it"):
    return asyncio.run(run_turn(agent, text))


def last_results(agent):
    return [block for block in agent.session.messages[-2]["content"] if block["type"] == "tool_result"]


def test_plain_answer(make_agent):
    agent, events = make_agent([say("hello")])
    assert turn(agent, "hi") == "hello"
    first, reply = agent.session.messages
    assert first["content"][0]["text"].startswith("<env>\ncwd: ")
    assert first["content"][1] == {"type": "text", "text": "hi"}
    assert reply == {"role": "assistant", "content": [{"type": "text", "text": "hello"}]}
    assert [event["type"] for event in events] == ["turn.start", "text", "usage", "turn.end"]
    assert {key: events[2][key] for key in ("limit", "steps", "spent", "prompt")} == {"limit": COMPACT_AT, "steps": 1, "spent": 15, "prompt": 10}
    assert events[-1]["stop"] == "end"


def test_env_block_only_opens_the_session(make_agent):
    agent, _ = make_agent([say("one"), say("two")])
    turn(agent, "first")
    turn(agent, "second")
    assert agent.session.messages[2]["content"] == [{"type": "text", "text": "second"}]


def test_tool_round_trip_and_append_only_history(make_agent, workspace):
    script = [
        use(("shell", {"command": "echo hi > made.txt; cat made.txt"})),
        use(("files", {"command": "view", "path": "made.txt"})),
        say("done"),
    ]
    agent, events = make_agent(script)
    assert turn(agent) == "done"
    assert (workspace / "made.txt").read_text() == "hi\n"
    assert agent.session.messages[2]["content"][0]["content"] == "hi"
    requests = agent.provider.requests
    for earlier, later in zip(requests, requests[1:]):
        assert later[: len(earlier)] == earlier
    usage = [event for event in events if event["type"] == "usage"]
    assert usage[0]["prompt"] == 100 and usage[0]["cache_read"] == 90
    finished = [event for event in events if event["type"] == "tool.result"]
    assert [event["preview"] for event in finished] == ["hi", "hi"] and all(event["seconds"] >= 0 for event in finished)


def test_denied_call_is_not_run(make_agent, workspace):
    agent, _ = make_agent([use(("shell", {"command": "touch made.txt"})), say("ok")], auto=False)
    turn(agent)
    assert not (workspace / "made.txt").exists()
    assert last_results(agent) == [
        {"type": "tool_result", "tool_use_id": last_results(agent)[0]["tool_use_id"], "content": DENIED, "is_error": True}
    ]


def test_read_only_call_never_asks(make_agent, workspace):
    (workspace / "a.txt").write_text("content")

    async def explode(call):
        raise AssertionError("asked for a read-only call")

    agent, _ = make_agent([use(("files", {"command": "view", "path": "a.txt"})), say("ok")], auto=False, ask=explode)
    turn(agent)
    assert last_results(agent)[0]["content"] == "content"


def test_approved_call_runs(make_agent, workspace):
    async def allow(call):
        return True

    agent, events = make_agent([use(("shell", {"command": "touch made.txt"})), say("ok")], auto=False, ask=allow)
    turn(agent)
    assert (workspace / "made.txt").exists()


def test_bad_input_and_unknown_tool_come_back_as_errors(make_agent):
    script = [use(("files", {"command": "view"}), ("nope", {}), ("shell", {"command": 5})), say("ok")]
    agent, _ = make_agent(script)
    turn(agent)
    bad_files, unknown, bad_shell = last_results(agent)
    assert json.loads(bad_files["content"])["problem"] == "missing path"
    assert unknown["content"] == "Unknown tool nope."
    assert json.loads(bad_shell["content"])["problem"] == "command must be string"
    assert all(block["is_error"] for block in (bad_files, unknown, bad_shell))


def test_bad_calls_still_show_up_as_events_and_a_bad_path_does_not_end_the_turn(make_agent, workspace):
    odd = use(("nope", {}), ("files", {"command": "create", "path": "a\x00b", "file_text": "x"}), ("files", {"command": "create", "path": "ok.txt", "file_text": "x"}))
    agent, events = make_agent([odd, say("carried on")])
    assert turn(agent) == "carried on"
    assert [event["name"] for event in events if event["type"] == "tool.call"] == ["nope", "files", "files"]
    assert [event["ok"] for event in events if event["type"] == "tool.result"] == [False, False, True]
    assert (workspace / "ok.txt").exists()


def test_a_call_left_open_by_a_dead_server_is_closed_before_the_next_message(make_agent):
    agent, events = make_agent([say("picking up again")])
    agent.session.append({"role": "user", "content": [{"type": "text", "text": "start"}]})
    agent.session.append({"role": "assistant", "content": [{"type": "tool_use", "id": "c9", "name": "shell", "input": {"command": "ls"}}]})
    resumed = Session.load(agent.session.id)
    agent.session = resumed
    assert turn(agent, "are you there?") == "picking up again"
    closed = resumed.messages[2]["content"][0]
    assert closed["tool_use_id"] == "c9" and closed["content"] == INTERRUPTED and closed["is_error"]
    assert [message["role"] for message in resumed.messages] == ["user", "assistant", "user", "user", "assistant"]


def test_whitespace_or_thinking_alone_counts_as_an_empty_reply(make_agent):
    blank = Reply([{"type": "text", "text": "\n\n"}], "\n\n", [], STOP_END)
    thought = Reply([{"type": "thinking", "thinking": "", "signature": "s"}], "", [], STOP_END)
    agent, events = make_agent([blank, say("first"), use(("shell", {"command": "echo hi"})), thought, say("second")])
    assert turn(agent) == "first"
    assert turn(agent, "again") == "second"
    assert not [event for event in events if event["type"] == "error"]
    assert all(str(message["content"]).strip("[]{}'\\n ") for message in agent.session.messages)
    assert not [message for message in agent.session.messages if message["role"] == "assistant" and not "".join(block.get("text", "x") for block in message["content"]).strip()]


def test_a_model_loaded_with_a_small_window_is_called_out_once(make_agent):
    agent, events = make_agent([say("one"), say("two")])
    agent.provider.window = 4096
    turn(agent)
    turn(agent, "again")
    told = [event for event in events if event["type"] == "notice"]
    assert len(told) == 1 and "4,096" in told[0]["text"] and told[0]["error"]
    assert {event["limit"] for event in events if event["type"] == "usage"} == {4096}


def test_a_refusal_with_no_text_ends_as_a_refusal_not_as_an_empty_reply(make_agent):
    agent, events = make_agent([Reply([], "", [], STOP_REFUSAL)])
    turn(agent)
    assert events[-1] == {"type": "turn.end", "stop": STOP_REFUSAL, "text": ""}
    assert not [event for event in events if event["type"] == "error"]
    assert [message["role"] for message in agent.session.messages] == ["user"]


def test_a_queued_message_takes_its_attachment_with_it(make_agent):
    picture = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}
    agent, _ = make_agent([use(("shell", {"command": "echo hi"})), say("seen")])
    agent.session.queue.append("look at this too")
    agent.session.queue_extra.append(picture)
    turn(agent)
    kinds = [block["type"] for block in agent.session.messages[2]["content"]]
    assert kinds == ["tool_result", "image", "text"] and agent.session.queue_extra == []


def test_tool_error_is_reported_to_the_model(make_agent):
    agent, _ = make_agent([use(("files", {"command": "view", "path": "missing.txt"})), say("ok")])
    turn(agent)
    assert last_results(agent)[0] == {
        "type": "tool_result",
        "tool_use_id": last_results(agent)[0]["tool_use_id"],
        "content": "missing.txt doesn't exist.",
        "is_error": True,
    }


def test_an_empty_reply_gets_one_retry_and_never_enters_the_history(make_agent):
    empty = Reply([], "", [], STOP_END)
    agent, events = make_agent([use(("shell", {"command": "echo hi"})), empty, say("here is the answer")])
    assert turn(agent) == "here is the answer"
    roles = [message["role"] for message in agent.session.messages]
    assert roles == ["user", "assistant", "user", "user", "assistant"]
    assert agent.session.messages[3]["content"] == [{"type": "text", "text": harness_note(EMPTY_NUDGE)}]
    assert all(message["content"] for message in agent.session.messages)
    assert not [event for event in events if event["type"] == "error"]


def test_a_model_that_stays_silent_is_reported_instead_of_ending_blank(make_agent):
    empty = Reply([], "", [], STOP_END)
    agent, events = make_agent([empty, empty])
    assert turn(agent) == ""
    assert [event["message"] for event in events if event["type"] == "error"] == [EMPTY_REPLY]
    assert events[-1] == {"type": "turn.end", "stop": "error", "text": ""}
    assert all(message["content"] for message in agent.session.messages)


def test_truncated_reply_does_not_run_tools(make_agent, workspace):
    agent, _ = make_agent([use(("shell", {"command": "touch made.txt"}), stop=STOP_LENGTH), say("ok")])
    turn(agent)
    assert not (workspace / "made.txt").exists()
    assert last_results(agent)[0]["content"] == TRUNCATED


def test_refusal_stops_the_turn_and_closes_calls(make_agent, workspace):
    agent, events = make_agent([use(("shell", {"command": "touch made.txt"}), stop=STOP_REFUSAL)])
    turn(agent)
    assert not (workspace / "made.txt").exists()
    assert agent.session.messages[-1]["content"][0]["content"] == DECLINED
    assert events[-1]["stop"] == STOP_REFUSAL


def test_paused_turn_is_sent_back(make_agent):
    agent, _ = make_agent([say("part one", stop=STOP_PAUSE), say("part two")])
    assert turn(agent) == "part two"
    assert [message["role"] for message in agent.session.messages] == ["user", "assistant", "assistant"]


def test_queued_message_joins_at_the_tool_boundary(make_agent):
    def first(messages):
        agent.session.queue.append("also check the tests")
        return use(("shell", {"command": "echo hi"}))

    agent, _ = make_agent([first, say("ok")])
    turn(agent)
    assert agent.session.messages[2]["content"][-1] == {"type": "text", "text": "also check the tests"}
    assert agent.session.queue == []


def test_read_only_calls_keep_their_order(make_agent, workspace):
    (workspace / "a.txt").write_text("A")
    (workspace / "b.txt").write_text("B")
    script = [
        use(
            ("files", {"command": "view", "path": "a.txt"}),
            ("files", {"command": "view", "path": "b.txt"}),
            ("shell", {"command": "echo C"}),
        ),
        say("ok"),
    ]
    agent, _ = make_agent(script)
    turn(agent)
    assert [block["content"] for block in last_results(agent)] == ["A", "B", "C"]


def test_abort_closes_open_calls(make_agent):
    agent, events = make_agent([use(("shell", {"command": "sleep 30"}))])

    async def scenario():
        task = asyncio.create_task(run_turn(agent, "wait"))
        while not any(event["type"] == "tool.call" for event in events):
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.2)
        task.cancel()
        await task

    asyncio.run(scenario())
    assert agent.session.messages[-1]["content"][0]["content"] == INTERRUPTED
    assert events[-1] == {"type": "turn.end", "stop": "aborted", "text": ""}


def test_stop_also_drops_what_was_queued(make_agent):
    agent, events = make_agent([use(("shell", {"command": "sleep 30"}))])
    agent.ctx.rag_once = True

    async def scenario():
        task = asyncio.create_task(run_turn(agent, "wait"))
        while not any(event["type"] == "tool.call" for event in events):
            await asyncio.sleep(0.01)
        agent.session.queue.append("also update the README")
        task.cancel()
        await task

    asyncio.run(scenario())
    assert agent.session.queue == [] and agent.ctx.rag_once is False
    assert events[-2] == {"type": "notice", "text": "Stopped. 1 queued message(s) were not sent."}


def test_provider_error_ends_the_turn_cleanly(make_agent):
    agent, events = make_agent([RuntimeError("boom")])
    turn(agent)
    assert events[-2] == {"type": "error", "message": "RuntimeError: boom"}
    assert events[-1]["stop"] == "error"
    assert agent.session.messages[-1]["role"] == "user"
