import asyncio
import base64
import threading
import time

import pytest
import uvicorn
from conftest import FakeProvider, say, use

from hypersonic.cli import Client, ClientError, turn
from hypersonic.engine.session import LOG_NAME, home
from hypersonic.plugins import budgets
from hypersonic.policy.rules import ASK_MODE
from hypersonic.providers.base import STOP_PAUSE, keep_model
from hypersonic.server import CONFIG_NAME, HOST, MAX_IMAGE_BYTES, NAME_CHARS, Live, create_app, free_port

TOKEN = "test-token"


@pytest.fixture
def served():
    scripts = []

    def factory(model, effort):
        script = scripts.pop(0)
        if isinstance(script, Exception):
            raise script
        return FakeProvider(script)

    port = free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(TOKEN, factory), host=HOST, port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.01)
    yield Client(port, TOKEN), scripts
    server.should_exit = True
    thread.join()


def start(client, workspace, **extra):
    return client.call("POST", "/session", {"cwd": str(workspace), **extra})["id"]


def test_token_is_required(served, workspace):
    client, _ = served
    intruder = Client(client.port, "wrong")
    assert intruder.call("GET", "/health")["ok"] is True
    for method, path in (("GET", "/session"), ("POST", "/session")):
        with pytest.raises(ClientError, match="401"):
            intruder.call(method, path, {"cwd": str(workspace)})
    with pytest.raises(ClientError, match="401"):
        list(intruder.events("anything", 0))


def test_full_turn_through_client_and_server(served, workspace):
    client, scripts = served
    scripts.append([use(("shell", {"command": "echo hi > made.txt"})), say("all done")])
    sid = start(client, workspace, mode="yolo")
    end = turn(client, sid, "make the file")
    assert end["text"] == "all done"
    assert end["stop"] == "end"
    assert end["usage"]["prompt"] == 10
    assert (workspace / "made.txt").read_text() == "hi\n"
    listed = client.call("GET", "/session")
    assert listed == [{"id": sid, "cwd": str(workspace), "messages": 4, "busy": False}]


def test_headless_client_denies_unapproved_calls(served, workspace):
    client, scripts = served
    scripts.append([use(("shell", {"command": "touch made.txt"})), say("was denied")])
    sid = start(client, workspace)
    assert turn(client, sid, "try it")["text"] == "was denied"
    assert not (workspace / "made.txt").exists()


def test_event_stream_replays_from_any_point(served, workspace):
    client, scripts = served
    scripts.append([say("hello")])
    sid = start(client, workspace)
    turn(client, sid, "hi")
    kinds = []
    for event in client.events(sid, 0):
        kinds.append(event["type"])
        if event["type"] == "turn.end":
            break
    assert kinds == ["turn.start", "text", "usage", "turn.end"]
    assert next(client.events(sid, 3))["type"] == "turn.end"


def test_hanging_up_frees_a_reader_blocked_on_the_event_stream(served, workspace):
    client, scripts = served
    scripts.append([])
    sid = start(client, workspace)
    reader = threading.Thread(target=lambda: list(client.events(sid, 0)), daemon=True)
    reader.start()
    while not (client.stream and client.stream.sock):
        time.sleep(0.01)
    assert reader.is_alive()
    client.hang_up()
    reader.join(timeout=3)
    assert not reader.is_alive()


def test_a_hang_up_that_comes_before_the_stream_has_connected_still_ends_it(served, workspace):
    client, scripts = served
    scripts.append([])
    sid = start(client, workspace)
    dropped, got = threading.Event(), []
    reader = Client(client.port, TOKEN)
    stream = reader.events(sid, 0, lambda: not dropped.is_set())
    dropped.set()
    reader.hang_up()
    waiting = threading.Thread(target=lambda: got.extend(stream), daemon=True)
    waiting.start()
    waiting.join(timeout=5)
    assert not waiting.is_alive() and got == []


def test_resume_continues_an_open_session_and_reloads_a_closed_one(served, workspace):
    client, scripts = served
    scripts.append([say("first answer"), say("second answer")])
    sid = start(client, workspace)
    turn(client, sid, "one")
    assert start(client, workspace, resume=sid[:8]) == sid and scripts == []
    assert turn(client, sid, "two")["text"] == "second answer"

    closed = Session.create(workspace)
    asked = [{"type": "text", "text": "<env>\ncwd: x\n</env>"}, {"type": "text", "text": "what is the code word?"}]
    closed.append({"role": "user", "content": asked})
    said = [{"type": "text", "text": "marmalade"}, {"type": "tool_use", "id": "c1", "name": "shell", "input": {"command": "ls"}}]
    closed.append({"role": "assistant", "content": said})
    closed.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "a.txt", "is_error": True}]})
    scripts.append([say("it was marmalade")])
    assert start(client, workspace, resume=closed.id[:8]) == closed.id
    history = client.call("GET", f"/session/{closed.id}/history")
    assert history["items"] == [
        {"role": "user", "text": "what is the code word?"},
        {"role": "assistant", "text": "marmalade"},
        {"role": "tool", "id": "c1", "name": "shell", "input": {"command": "ls"}, "ok": False, "preview": "a.txt"},
    ]
    assert history["seq"] == 0 and history["busy"] is False and history["cwd"] == str(workspace)
    assert turn(client, closed.id, "again?")["text"] == "it was marmalade"
    with pytest.raises(ClientError, match="No session zzzz"):
        start(client, workspace, resume="zzzz")


def test_resuming_from_a_trusted_folder_does_not_trust_the_folder_the_session_lives_in(served, workspace, tmp_path):
    client, scripts = served
    other = tmp_path / "other"
    (other / ".hypersonic" / "plugins").mkdir(parents=True)
    marker = tmp_path / "ran.txt"
    (other / ".hypersonic" / "plugins" / "evil.py").write_text(f"open({str(marker)!r}, 'w').write('x')\ndef register(hooks): pass\n")
    saved = Session.create(other.resolve())
    saved.append({"role": "user", "content": [{"type": "text", "text": "hi"}]})
    scripts.append([])
    assert start(client, workspace, resume=saved.id, trusted=True) == saved.id
    assert not marker.exists()
    scripts.append([])
    start(client, other, trusted=True)
    assert marker.exists()


def live_session(workspace, store, script):
    return Live(Session.create(workspace), FakeProvider(script), store, ASK_MODE, False, False, budgets.Budget())


async def test_resume_replays_only_the_reply_still_streaming(workspace, store):
    item = live_session(workspace, store, [say("Already saved.", stop=STOP_PAUSE)])
    provider = item.agent.provider
    first_reply = provider.stream
    prefix_sent, finish = asyncio.Event(), asyncio.Event()

    async def stream(system, tools, messages, on_text):
        if provider.script:
            return await first_reply(system, tools, messages, on_text)
        await on_text("The ")
        await on_text("answer ")
        prefix_sent.set()
        await finish.wait()
        await on_text("is 42.")
        return say("The answer is 42.")

    provider.stream = stream
    item.task = asyncio.create_task(item.drive("What is the answer?"))
    try:
        await asyncio.wait_for(prefix_sent.wait(), timeout=2)
        cursor = item.history_cursor()
        history = item.agent.session.messages
        assert [m["content"][0]["text"] for m in history if m["role"] == "assistant"] == ["Already saved."]
    finally:
        finish.set()
        await asyncio.wait_for(item.task, timeout=2)

    replay = item.events[cursor:]
    assert [e["delta"] for e in replay if e["type"] == "text"] == ["The ", "answer ", "is 42."]
    assert replay[-1]["type"] == "turn.end" and item.history_cursor() == len(item.events)


def test_stopping_a_slash_command_still_ends_the_turn(workspace, store):
    async def scenario():
        item = live_session(workspace, store, [])
        item.task = asyncio.create_task(item.supervise(asyncio.sleep(30)))
        await asyncio.sleep(0)
        item.task.cancel()
        await item.task
        return item.events

    assert asyncio.run(scenario()) == [{"type": "turn.end", "stop": "aborted", "text": "", "seq": 1}]


def test_a_message_sent_during_a_slash_command_is_answered_after_it(workspace, store):
    async def scenario():
        item = live_session(workspace, store, [say("answered")])
        item.agent.session.queue.append("and what about this?")
        await item.command("/help", [])
        return [(event["type"], event.get("stop")) for event in item.events if event["type"] == "turn.end"], item.agent.session

    ends, session = asyncio.run(scenario())
    assert ends == [("turn.end", "command"), ("turn.end", "end")] and session.queue == []
    assert session.messages[0]["content"][-1]["text"] == "and what about this?"


def test_a_message_that_opens_with_a_path_goes_to_the_model_with_its_picture(served, workspace):
    client, scripts = served
    scripts.append([say("compared")])
    sid = start(client, workspace)
    up = client.call("POST", f"/session/{sid}/upload", {"name": "p.png", "media_type": "image/png", "data": PNG_1x1})
    body = {"text": "/tmp/report.txt has the numbers, compare them with this picture", "attachments": [up["id"]]}
    drive_until_end(client, sid, client.call("POST", f"/session/{sid}/message", body)["seq"])
    kinds = [block["type"] for block in Session.load(sid).messages[0]["content"]]
    assert "image" in kinds and Session.load(sid).messages[1]["content"][0]["text"] == "compared"


def test_a_waiting_permission_request_is_sent_again_after_a_reload(served, workspace):
    client, scripts = served
    request = use(("shell", {"command": "touch made.txt"}))
    # Providers can reuse a call id on a later turn.
    scripts.append([request, say("was denied"), request, say("was denied again")])
    sid = start(client, workspace)
    assert turn(client, sid, "first try")["text"] == "was denied"
    seq = client.call("POST", f"/session/{sid}/message", {"text": "try it"})["seq"]
    asked = next(event for event in client.events(sid, seq) if event["type"] == "permission.request")
    client.hang_up()
    past = client.call("GET", f"/session/{sid}/history")
    assert past["busy"] is True and past["seq"] == asked["seq"] - 1
    again = next(iter(client.events(sid, past["seq"])))
    assert again == asked
    client.hang_up()
    client.call("POST", f"/session/{sid}/permission", {"id": again["id"], "allow": False})
    drive_until_end(client, sid, asked["seq"])
    assert client.call("GET", f"/session/{sid}/history")["seq"] > asked["seq"]


def test_a_closed_session_is_freed(served, workspace):
    client, scripts = served
    scripts.append([])
    sid = start(client, workspace)
    assert client.call("DELETE", f"/session/{sid}") == {"ok": True}
    assert client.call("GET", "/session") == []
    with pytest.raises(ClientError, match="404"):
        client.call("POST", f"/session/{sid}/message", {"text": "hi"})


def test_odd_uploads_are_refused_or_saved_as_plain_files(served, workspace):
    client, scripts = served
    scripts.append([])
    sid = start(client, workspace)
    send = lambda **body: client.call("POST", f"/session/{sid}/upload", body)
    assert send(name="photo.heic", media_type="image/heic", data=PNG_1x1)["kind"] == "file"
    assert len(send(name="n" * 300 + ".txt", media_type="text/plain", data=PNG_1x1)["name"]) == NAME_CHARS
    big = base64.b64encode(b"x" * (MAX_IMAGE_BYTES + 1)).decode()
    for refused in ({"media_type": "image/png", "data": big}, {"media_type": "image/png", "data": ""}):
        with pytest.raises(ClientError, match="empty or over"):
            send(name="p.png", **refused)


def test_the_server_answers_bad_input_with_a_clear_refusal(served, workspace):
    client, scripts = served
    scripts.append([])
    sid = start(client, workspace)
    with pytest.raises(ClientError, match="message is empty"):
        client.call("POST", f"/session/{sid}/message", {"text": "  "})
    with pytest.raises(ClientError, match="422"):
        list(client.events(sid, -2))
    with pytest.raises(ClientError, match="not a folder"):
        client.call("POST", "/session", {"cwd": "bad\x00folder"})
    for hidden in ("/docs", "/redoc", "/openapi.json"):
        with pytest.raises(ClientError, match="404"):
            client.call("GET", hidden)
    with pytest.raises(ClientError, match="401"):
        Client(client.port, "caf\u00e9").call("GET", "/session")
    empty = home() / "sessions" / "emptylog"
    empty.mkdir(parents=True)
    (empty / LOG_NAME).write_text("")
    with pytest.raises(ClientError, match="damaged"):
        start(client, workspace, resume="emptylog")
    (home() / CONFIG_NAME).write_text("model = [unclosed")
    with pytest.raises(ClientError, match="can't be read"):
        start(client, workspace)
    (home() / CONFIG_NAME).write_text("model = 5\ncode = 7\n")
    scripts.append([])
    assert start(client, workspace)


def test_a_broken_plugin_is_reported_and_the_session_still_opens(served, workspace):
    client, scripts = served
    (home() / "plugins").mkdir(parents=True)
    (home() / "plugins" / "bad.py").write_text("raise RuntimeError('boom')\n")
    scripts.append([])
    sid = start(client, workspace)
    first = next(iter(client.events(sid, 0)))
    assert first["type"] == "notice" and "bad.py was skipped" in first["text"] and first["error"]


def test_bad_requests(served, workspace):
    client, _ = served
    with pytest.raises(ClientError, match="not a folder"):
        client.call("POST", "/session", {"cwd": str(workspace / "missing")})
    with pytest.raises(ClientError, match="No session"):
        client.call("POST", "/session/nope/message", {"text": "hi"})
    with pytest.raises(ClientError, match="No session nope"):
        client.call("POST", "/session", {"cwd": str(workspace), "resume": "nope"})


def test_index_then_search(served, workspace):
    client, _ = served
    (workspace / "notes").mkdir()
    (workspace / "notes" / "pets.md").write_text("# Pets\n\nThe walrus eats clams every morning.\n")
    counts = client.call("POST", "/index", {"cwd": str(workspace), "folder": "notes"})
    assert counts == {"indexed": 1, "unchanged": 0, "removed": 0}
    found = client.call("GET", f"/search?q=walrus+clams&cwd={workspace}")
    assert found["hits"][0]["path"] == str(workspace / "notes" / "pets.md")
    assert found["hits"][0]["heading"] == "Pets"
    assert found["confidence"] > 0.3
    assert client.call("GET", f"/search?q=walrus&cwd={workspace}&scope=code")["hits"] == []
    with pytest.raises(ClientError, match="doesn't exist"):
        client.call("POST", "/index", {"cwd": str(workspace), "folder": "missing"})
    with pytest.raises(ClientError, match="collection must be"):
        client.call("POST", "/index", {"cwd": str(workspace), "folder": "notes", "collection": "junk"})


def test_indexed_documents_reach_the_model(served, workspace):
    client, scripts = served
    (workspace / "pets.md").write_text("# Pets\n\nThe walrus eats clams every morning.\n")
    client.call("POST", "/index", {"cwd": str(workspace), "folder": "pets.md"})
    question = "what does the walrus eat every morning"
    scripts.append([say("no idea"), say("clams"), say("no idea again")])
    sid = start(client, workspace)
    turn(client, sid, question)
    assert turn(client, sid, f"/rag {question}")["text"] == "clams"
    turn(client, sid, "and what does the walrus drink at night time")
    plain, with_rag, after = [m["content"] for m in Session.load(sid).messages if m["role"] == "user"]
    assert "walrus eats clams" not in str(plain) and "walrus eats clams" not in str(after)
    assert with_rag[0]["text"] == question and "The walrus eats clams every morning." in with_rag[1]["text"]
    assert turn(client, sid, "/rag")["notices"][0].startswith("RAG is on")
    seen = []
    for event in client.events(sid, 0):
        seen.append(event)
        if event["type"] == "notice" and event["text"].startswith("RAG is on"):
            break
    assert {"type": "rag", "on": True} == {key: seen[-2][key] for key in ("type", "on")}


def test_slash_commands_run_without_the_model(served, workspace):
    client, scripts = served
    scripts.append([say("coded")])
    (workspace / "app.py").write_text("def launch_rocket():\n    return 'liftoff'\n")
    scripts[0].insert(0, use(("search", {"query": "launch rocket", "scope": "code"})))
    sid = start(client, workspace)
    end = turn(client, sid, "/check pytest -q")
    assert end["stop"] == "command"
    assert end["notices"] == ["Check command: pytest -q"]
    assert turn(client, sid, "/code")["notices"] == ["Unknown command /code. Try /help."]
    assert turn(client, sid, "/rag launch it")["text"] == "coded"
    assert client.call("GET", f"/search?q=launch+rocket&cwd={workspace}&scope=code")["hits"][0]["path"].endswith("app.py")


def test_trusted_session_indexes_project_rules_and_skills(served, workspace):
    client, scripts = served
    (workspace / "AGENTS.md").write_text("Deploys go through the marmalade pipeline.\n")
    skill = workspace / ".hypersonic" / "skills" / "release"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# Release skill\n\nTag the build, then run the walrus checklist.\n")
    scripts.extend([[say("ok")], [say("ok")]])
    start(client, workspace)
    assert client.call("GET", f"/search?q=walrus+checklist&cwd={workspace}")["hits"] == []
    start(client, workspace, trusted=True)
    assert client.call("GET", f"/search?q=walrus+checklist&cwd={workspace}&scope=skills")["hits"]
    assert client.call("GET", f"/search?q=marmalade+pipeline&cwd={workspace}&scope=docs")["hits"]
    with pytest.raises(ClientError, match="mode must be"):
        start(client, workspace, mode="reckless")


import http.client as _http
from hypersonic.engine.session import Session

PNG_1x1 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="


def drive_until_end(client, sid, seq):
    for ev in client.events(sid, seq):
        if ev["type"] == "turn.end":
            return


def user_blocks(workspace_home_sid):
    home, sid = workspace_home_sid
    session = Session.load(sid)
    first = session.messages[0]["content"]
    return first


def test_ui_page_is_served(served, workspace):
    client, _ = served
    conn = _http.HTTPConnection(HOST, client.port, timeout=10)
    conn.request("GET", "/ui")
    resp = conn.getresponse()
    body = resp.read().decode()
    conn.close()
    assert resp.status == 200
    assert "text/html" in resp.getheader("content-type")
    assert "<!doctype html>" in body and "Hypersonic" in body
    assert "react" in body and "tailwindcss" in body
    assert TOKEN not in body  # the page carries no secret; the token comes from the URL
    allowed = body.split("ALLOWED_TAGS: [")[1].split("]")[0]
    assert "'img'" not in allowed and "'script'" not in allowed and "'iframe'" not in allowed
    assert body.count("dangerouslySetInnerHTML") == 1 and "DOMPurify.sanitize(" in body


def test_image_attachment_becomes_a_vision_block(served, workspace, tmp_path, monkeypatch):
    client, scripts = served
    scripts.append([say("I can see the image.")])
    sid = start(client, workspace)
    up = client.call("POST", f"/session/{sid}/upload", {"name": "pic.png", "media_type": "image/png", "data": PNG_1x1})
    assert up["kind"] == "image"
    seq = client.call("POST", f"/session/{sid}/message", {"text": "what is this image?", "attachments": [up["id"]]})["seq"]
    drive_until_end(client, sid, seq)
    blocks = Session.load(sid).messages[0]["content"]
    images = [b for b in blocks if b.get("type") == "image"]
    assert len(images) == 1
    assert images[0]["source"] == {"type": "base64", "media_type": "image/png", "data": PNG_1x1}
    texts = [b["text"] for b in blocks if b.get("type") == "text"]
    assert any("what is this image?" in t for t in texts)


def test_file_attachment_is_saved_and_referenced(served, workspace):
    client, scripts = served
    scripts.append([say("Read it.")])
    sid = start(client, workspace)
    import base64
    data = base64.b64encode(b"hello from a file").decode()
    up = client.call("POST", f"/session/{sid}/upload", {"name": "notes weird!.txt", "media_type": "text/plain", "data": data})
    assert up["kind"] == "file"
    seq = client.call("POST", f"/session/{sid}/message", {"text": "summarize the file", "attachments": [up["id"]]})["seq"]
    drive_until_end(client, sid, seq)
    saved = list((workspace / ".hypersonic" / "uploads").glob("*notes_weird_.txt"))
    assert len(saved) == 1 and saved[0].read_bytes() == b"hello from a file"
    note = [b["text"] for b in Session.load(sid).messages[0]["content"] if b.get("type") == "text" and "attached" in b["text"].lower()]
    assert note and str(saved[0]) in note[0]


def test_upload_rejects_non_base64_and_oversize(served, workspace):
    client, scripts = served
    scripts.append([])
    sid = start(client, workspace)
    with pytest.raises(ClientError, match="base64"):
        client.call("POST", f"/session/{sid}/upload", {"name": "x", "media_type": "text/plain", "data": "!!!not base64!!!"})


def test_attachments_only_apply_to_their_own_message(served, workspace):
    client, scripts = served
    scripts.extend([[say("one")], [say("two")]])
    sid = start(client, workspace)
    up = client.call("POST", f"/session/{sid}/upload", {"name": "p.png", "media_type": "image/png", "data": PNG_1x1})
    seq = client.call("POST", f"/session/{sid}/message", {"text": "first", "attachments": [up["id"]]})["seq"]
    drive_until_end(client, sid, seq)
    seq2 = client.call("POST", f"/session/{sid}/message", {"text": "second"})["seq"]
    drive_until_end(client, sid, seq2)
    second_turn = Session.load(sid).messages[2]["content"]
    assert all(b.get("type") != "image" for b in second_turn)


def test_models_endpoint_serves_the_catalog_and_reads_it_once(served, monkeypatch):
    client, _ = served
    reads = []
    offered = [{"name": "ollama/qwen3.5:2b", "summary": "on this computer, 3.1 GB"}]
    monkeypatch.setattr("hypersonic.server.list_models", lambda home: reads.append(home) or offered)
    assert client.call("GET", "/models") == offered
    assert client.call("GET", "/models") == offered
    assert len(reads) == 1


def test_switching_the_model_is_announced_and_kept_for_new_sessions(served, workspace, monkeypatch):
    monkeypatch.delenv("HYPERSONIC_MODEL", raising=False)
    client, scripts = served
    scripts.append([])
    sid = start(client, workspace, model="anthropic/claude-haiku-5-5")
    assert turn(client, sid, "/model")["notices"][0].startswith("Model: anthropic/claude-haiku-5-5.")
    end = turn(client, sid, "/model ollama/qwen3.5:2b")
    assert end["notices"] == ["Model switched to ollama/qwen3.5:2b. New chats will use it too."]
    seen = []
    for event in client.events(sid, 0):
        seen.append(event)
        if event["type"] == "turn.end" and len([e for e in seen if e["type"] == "turn.end"]) == 2:
            break
    assert [event["name"] for event in seen if event["type"] == "model"] == ["ollama/qwen3.5:2b"]
    refused = turn(client, sid, "/model nowhere/x")["notices"][0]
    assert refused.startswith("NOT switched to nowhere/x.") and refused.endswith("Still using ollama/qwen3.5:2b.")
    scripts.append([])
    assert client.call("POST", "/session", {"cwd": str(workspace)})["model"] == "ollama/qwen3.5:2b"


def test_a_saved_model_that_cannot_start_falls_back_and_says_so(served, workspace, monkeypatch):
    monkeypatch.delenv("HYPERSONIC_MODEL", raising=False)
    client, scripts = served
    keep_model("openrouter/nvidia/nemotron:free", home())
    scripts.extend([ValueError("openrouter needs an API key."), []])
    created = client.call("POST", "/session", {"cwd": str(workspace)})
    assert not created["model"].startswith("openrouter/") and "couldn't start" in created["notice"]
    first = next(iter(client.events(created["id"], 0)))
    assert first["error"] and "openrouter/nvidia/nemotron:free couldn't start" in first["text"]
    scripts.append(ValueError("openrouter needs an API key."))
    with pytest.raises(ClientError, match="needs an API key"):
        client.call("POST", "/session", {"cwd": str(workspace), "model": "openrouter/nvidia/nemotron:free"})


def test_commands_endpoint_lists_slash_commands(served, workspace):
    client, scripts = served
    scripts.append([])
    sid = start(client, workspace)
    cmds = client.call("GET", f"/session/{sid}/commands")
    names = [c["name"] for c in cmds]
    assert {"model", "check", "compact", "index", "undo", "help"} <= set(names)
    assert all(c.get("summary") for c in cmds)


def test_sessions_endpoint_lists_past_sessions(served, workspace):
    client, scripts = served
    scripts.append([say("ok")])
    sid = start(client, workspace)
    seq = client.call("POST", f"/session/{sid}/message", {"text": "remember the walrus"})["seq"]
    drive_until_end(client, sid, seq)
    listed = client.call("GET", "/sessions")
    me = next((s for s in listed if s["id"] == sid), None)
    assert me is not None
    assert me["preview"] == "remember the walrus" and me["messages"] >= 1 and me["cwd"] == str(workspace)
    scripts.append([])
    untouched = start(client, workspace)
    assert untouched not in [item["id"] for item in client.call("GET", "/sessions")]
