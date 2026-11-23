import json

import pytest

from hypersonic.engine.session import LOG_NAME, Session, list_sessions, resolve_id
from hypersonic.providers.base import Usage


def test_log_replays_the_exact_message_list(workspace):
    session = Session.create(workspace)
    thinking = {"type": "thinking", "thinking": "", "signature": "sig=="}
    session.append({"role": "user", "content": [{"type": "text", "text": "héllo"}]})
    session.append({"role": "assistant", "content": [thinking, {"type": "text", "text": "hi"}]})
    session.record(Usage(input=3, cache_read=7, output=2))

    loaded = Session.load(session.id)
    assert loaded.messages == session.messages
    assert loaded.workspace == workspace
    assert loaded.lines == session.lines


def test_log_lines_form_a_chain(workspace):
    session = Session.create(workspace)
    session.append({"role": "user", "content": []})
    session.record(Usage())
    records = [json.loads(line) for line in (session.dir / LOG_NAME).read_text().splitlines()]
    assert [record["type"] for record in records] == ["meta", "message", "usage"]
    assert [(record["id"], record["parent_id"]) for record in records] == [(1, 0), (2, 1), (3, 2)]


def test_loaded_session_keeps_appending_to_the_same_log(workspace):
    session = Session.create(workspace)
    session.append({"role": "user", "content": []})
    loaded = Session.load(session.id)
    loaded.append({"role": "assistant", "content": []})
    assert len(Session.load(session.id).messages) == 2


def test_an_unusual_line_break_character_does_not_break_the_log(workspace):
    session = Session.create(workspace)
    session.append({"role": "user", "content": [{"type": "text", "text": "before\u2028after\u0085end"}]})
    assert Session.load(session.id).messages == session.messages
    assert [row["id"] for row in list_sessions()] == [session.id]


def test_a_failed_write_keeps_memory_and_log_in_step(workspace):
    session = Session.create(workspace)
    with pytest.raises(UnicodeEncodeError):
        session.append({"role": "user", "content": [{"type": "text", "text": "half an emoji \ud83d"}]})
    assert session.messages == [] and session.lines == 1
    session.append({"role": "user", "content": [{"type": "text", "text": "fine"}]})
    assert len(Session.load(session.id).messages) == 1


def test_sessions_are_found_by_the_start_of_their_id_and_nothing_else(workspace):
    first, second = Session.create(workspace), Session.create(workspace)
    assert resolve_id(first.id) == first.id and resolve_id(first.id[:9]) == first.id
    for wrong in ("", "zzzz", f"../sessions/{first.id}", first.id.upper() + "/"):
        with pytest.raises(FileNotFoundError):
            resolve_id(wrong)
    assert second.id != first.id


def test_missing_session(workspace):
    with pytest.raises(FileNotFoundError):
        Session.load("nope")


def test_resume_restores_step_and_token_counters(workspace):
    session = Session.create(workspace)
    session.append({"role": "user", "content": []})
    session.record(Usage(input=3, cache_read=7, output=2))
    session.record(Usage(input=5, output=4))
    assert (session.steps, session.tokens) == (2, 21)
    loaded = Session.load(session.id)
    assert (loaded.steps, loaded.tokens) == (2, 21)


def test_list_sessions_skips_a_corrupt_log(workspace):
    from hypersonic.engine.session import LOG_NAME, home, list_sessions

    good = Session.create(workspace)
    good.append({"role": "user", "content": [{"type": "text", "text": "find the bug"}]})
    broken = home() / "sessions" / "broken"
    broken.mkdir(parents=True)
    (broken / LOG_NAME).write_text('{"type": "message"}\nnot json at all\n')
    ids = {row["id"] for row in list_sessions()}
    assert good.id in ids  # one unreadable folder must not sink the whole listing
