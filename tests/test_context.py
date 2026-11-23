import asyncio
import json

from conftest import say, use

from hypersonic.context.compact import COMPACT_AT, CONTINUE
from hypersonic.context.squeeze import CHARS_PER_TOKEN
from hypersonic.engine.commands import run_command
from hypersonic.engine.loop import run_turn
from hypersonic.engine.session import Session
from hypersonic.providers.base import STOP_END, Reply, Usage
from hypersonic.rag.ingest import CHAT, CODE, DOCS, index_text
from hypersonic.tools import search
from hypersonic.rag.retrieve import retrieve

BIG = Usage(input=COMPACT_AT, output=10)
HANDBOOK = "# Handbook\n\n## Refunds\n\nRefunds settle within five business days.\n"
QUESTION = "how long do refunds take to settle"
LONG_SESSION_STEPS = 150
LONG_SESSION_CEILING = 35_000


def turn(agent, text="do it"):
    return asyncio.run(run_turn(agent, text))


def index_handbook(agent):
    index_text(agent.ctx.store, str(agent.ctx.workspace), DOCS, "/docs/handbook.md", HANDBOOK, "v1", headings=True)


def test_search_tool_returns_passages(make_agent):
    agent, _ = make_agent([use(("search", {"query": "refunds settle"}), ("search", {"query": "walrus"})), say("ok")])
    index_handbook(agent)
    agent.ctx.rag = True
    turn(agent, "hi")
    hit, miss = agent.session.messages[2]["content"]
    assert hit["content"] == "[docs] /docs/handbook.md:5-5 (Handbook > Refunds)\nRefunds settle within five business days."
    assert miss["content"] == "No matches."


def test_code_search_indexes_the_workspace_by_itself_and_stays_fresh(make_agent, workspace):
    (workspace / "billing.py").write_text("def settle_invoice(total):\n    return round(total, 2)\n")
    find = use(("search", {"query": "settle invoice", "scope": "code"}))
    again = use(("search", {"query": "refund customer", "scope": "code"}))
    agent, _ = make_agent([find, say("found"), again, say("found the new one")])
    agent.ctx.rag = True
    assert retrieve(agent.ctx.store, "settle invoice", agent.ctx.scopes, [CODE]).hits == []
    turn(agent, "where is the invoice code?")
    assert agent.session.messages[2]["content"][0]["content"].startswith(f"[code] {workspace / 'billing.py'}:1-2")
    (workspace / "refunds.py").write_text("def refund_customer(order):\n    return order.total\n")
    turn(agent, "and refunds?")
    assert str(workspace / "refunds.py") in agent.session.messages[-2]["content"][0]["content"]


def test_a_search_without_a_scope_does_not_index_code(make_agent, workspace):
    (workspace / "billing.py").write_text("def settle_invoice(total):\n    return round(total, 2)\n")
    agent, _ = make_agent([use(("search", {"query": "settle invoice"})), say("nothing")])
    agent.ctx.rag = True
    turn(agent, "hi")
    assert agent.session.messages[2]["content"][0]["content"] == "No matches."


def test_a_huge_workspace_is_not_indexed_and_the_agent_is_told_why(make_agent, workspace, monkeypatch):
    monkeypatch.setattr(search, "CODE_FILE_LIMIT", 2)
    for name in ("a.py", "b.py", "c.py"):
        (workspace / name).write_text("x = 1\n")
    agent, _ = make_agent([use(("search", {"query": "x", "scope": "code"})), say("will use rg")])
    agent.ctx.rag = True
    turn(agent, "find x")
    refused = agent.session.messages[2]["content"][0]
    assert refused["is_error"] and "Find code with rg" in refused["content"]
    assert retrieve(agent.ctx.store, "x", agent.ctx.scopes, [CODE]).hits == []


def test_documents_stay_out_until_the_user_asks_with_rag(make_agent):
    tries = [("search", {"query": "refunds settle", "scope": scope}) for scope in ("docs", "code", "memory", "chat")]
    refused = use(*tries, ("search", {"query": "refunds settle"}))
    agent, _ = make_agent([refused, say("ask with /rag"), say("five days"), say("no documents again")])
    index_handbook(agent)
    (agent.ctx.workspace / "handbook.md").write_text(HANDBOOK)
    turn(agent, QUESTION)
    assert len(agent.session.messages[0]["content"]) == 2
    assert len(agent.session.messages[2]["content"]) == 5
    assert all(block["is_error"] and block["content"] == search.RAG_OFF for block in agent.session.messages[2]["content"])
    assert retrieve(agent.ctx.store, "refunds settle", agent.ctx.scopes, [CODE]).hits == []

    outcome = asyncio.run(run_command(agent, f"/rag {QUESTION}"))
    assert outcome.prompt == QUESTION and agent.ctx.rag_on and not agent.ctx.rag
    turn(agent, outcome.prompt)
    assert "Refunds settle within five business days." in agent.session.messages[4]["content"][-1]["text"]
    assert not agent.ctx.rag_on
    agent.ctx.attached.clear()
    turn(agent, QUESTION)
    assert agent.session.messages[-2]["content"] == [{"type": "text", "text": QUESTION}]


def test_a_short_question_still_gets_passages_when_asked_with_rag(make_agent):
    agent, _ = make_agent([say("five days"), say("nothing attached")])
    index_handbook(agent)
    turn(agent, asyncio.run(run_command(agent, "/rag refunds settle?")).prompt)
    assert "Refunds settle within five business days." in agent.session.messages[0]["content"][-1]["text"]
    agent.ctx.attached.clear()
    agent.ctx.rag = True
    turn(agent, "refunds settle?")
    assert agent.session.messages[2]["content"] == [{"type": "text", "text": "refunds settle?"}]


def test_rag_alone_switches_documents_on_and_off_for_the_chat(make_agent):
    agent, _ = make_agent([use(("search", {"query": "refunds settle", "scope": "docs"})), say("five days")])
    index_handbook(agent)
    assert asyncio.run(run_command(agent, "/rag")).notice.startswith("RAG is on")
    turn(agent, QUESTION)
    assert "<context>" in agent.session.messages[0]["content"][-1]["text"]
    assert "Refunds settle" in agent.session.messages[2]["content"][0]["content"]
    assert agent.ctx.rag
    assert asyncio.run(run_command(agent, "/rag")).notice.startswith("RAG is off")
    assert not agent.ctx.rag_on


def test_matching_documents_are_attached_once(make_agent):
    agent, _ = make_agent([say("five days"), say("still five days")])
    index_handbook(agent)
    agent.ctx.rag = True
    turn(agent, QUESTION)
    blocks = agent.session.messages[0]["content"]
    assert [block["text"][:9] for block in blocks] == ["<env>\ncwd", QUESTION[:9], "<context>"]
    assert "Refunds settle within five business days." in blocks[2]["text"]
    turn(agent, QUESTION)
    assert agent.session.messages[2]["content"] == [{"type": "text", "text": QUESTION}]


def test_short_messages_and_switched_off_agents_attach_nothing(make_agent):
    agent, _ = make_agent([say("welcome"), say("ok")])
    index_handbook(agent)
    agent.ctx.rag = True
    turn(agent, "refunds thanks")
    assert len(agent.session.messages[0]["content"]) == 2
    agent.ctx.rag = False
    turn(agent, QUESTION)
    assert agent.session.messages[2]["content"] == [{"type": "text", "text": QUESTION}]


def test_compaction_at_turn_end_replaces_history_with_a_note(make_agent):
    agent, events = make_agent([say("The launch code word is marmalade.", usage=BIG), say("Goal: remember the code word.")])
    turn(agent, "remember this")
    session = agent.session
    assert len(session.messages) == 1
    handoff = session.messages[0]["content"][1]["text"]
    assert handoff.startswith("<handoff>\nGoal: remember the code word.\n</handoff>")
    assert CONTINUE not in handoff
    assert session.messages[0]["content"][0]["text"].startswith("<env>")
    assert session.last == Usage()
    assert [event["type"] for event in events][-2:] == ["turn.end", "compaction"]

    archive = session.dir / "archive" / "001.md"
    assert "## assistant\nThe launch code word is marmalade." in archive.read_text()
    found = retrieve(agent.ctx.store, "launch code word", agent.ctx.scopes, [CHAT])
    assert "marmalade" in found.hits[0].body
    assert found.hits[0].path == str(archive)

    request = agent.provider.requests[1]
    assert request[:2] == [
        {"role": "user", "content": request[0]["content"]},
        {"role": "assistant", "content": [{"type": "text", "text": "The launch code word is marmalade."}]},
    ]
    assert request[2]["content"][0]["text"].startswith("Write a handoff note")
    assert Session.load(session.id).messages == session.messages
    assert Session.load(session.id).compactions == 1


def test_compaction_mid_turn_keeps_the_task_going(make_agent, workspace):
    script = [use(("shell", {"command": "echo step one > log.txt"}), usage=BIG), say("Next steps: write step two."), say("finished")]
    agent, events = make_agent(script)
    assert turn(agent) == "finished"
    messages = agent.session.messages
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[0]["content"][1]["text"].endswith(CONTINUE)
    assert agent.provider.requests[2] == [messages[0]]
    assert (workspace / "log.txt").exists()
    assert [event["type"] for event in events].count("compaction") == 1


def test_other_sessions_cannot_search_this_chat(make_agent):
    agent, _ = make_agent([say("The code word is marmalade.", usage=BIG), say("note")])
    turn(agent)
    stranger, _ = make_agent([])
    assert retrieve(agent.ctx.store, "marmalade", stranger.ctx.scopes).hits == []


def test_empty_note_or_failure_leaves_history_alone(make_agent):
    agent, _ = make_agent([say("answer", usage=BIG), say("")])
    turn(agent)
    assert len(agent.session.messages) == 2

    agent, events = make_agent([say("answer", usage=BIG), RuntimeError("boom")])
    turn(agent)
    assert len(agent.session.messages) == 2
    assert events[-1] == {"type": "error", "message": "Compaction failed. RuntimeError: boom"}


def test_a_handoff_reply_that_calls_a_tool_is_not_taken_as_the_note(make_agent):
    stalling = use(("shell", {"command": "cat notes.md"}))
    stalling.text = "Let me look at the notes file first."
    agent, _ = make_agent([say("answer", usage=BIG), stalling])
    turn(agent)
    assert len(agent.session.messages) == 2 and agent.session.compactions == 0
    assert agent.session.compact_retry_at > COMPACT_AT


def test_a_failed_compaction_is_not_retried_on_every_step(make_agent):
    step = use(("shell", {"command": "echo hi"}), usage=BIG)
    agent, _ = make_agent([step, say(""), step, step, say("done", usage=BIG)])
    turn(agent)
    assert len(agent.provider.requests) == 5 and agent.provider.script == []


def test_long_session_stays_under_budget(make_agent):
    sizes = []
    steps = iter(range(LONG_SESSION_STEPS))

    def model(messages):
        size = len(json.dumps(messages)) // CHARS_PER_TOKEN
        usage = Usage(cache_read=size, output=50)
        if messages[-1]["content"][0].get("text", "").startswith("Write a handoff note"):
            return say("Goal: keep printing numbers. Next steps: print more.", usage=usage)
        sizes.append(size)
        if next(steps, None) is None:
            return say("all done", usage=usage)
        return use(("shell", {"command": "seq 1 4000"}), usage=usage)

    agent, events = make_agent([model] * (LONG_SESSION_STEPS * 2))
    assert turn(agent) == "all done"
    assert len(sizes) == LONG_SESSION_STEPS + 1
    assert max(sizes) < LONG_SESSION_CEILING
    compactions = [event for event in events if event["type"] == "compaction"]
    assert len(compactions) >= 5
    assert len(list((agent.session.dir / "archive").iterdir())) == len(compactions)
