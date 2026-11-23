import threading

import pytest

pytest.importorskip("textual")
from textual.widgets import Input, Static  # noqa: E402

from hypersonic.tui import MENU_LIMIT, RED, Bubble, HypersonicApp, meter, model_label, short, summarize  # noqa: E402


MODELS = [
    {"name": "anthropic/claude-opus-5-5", "summary": "Claude API"},
    {"name": "ollama/qwen3.5:2b", "summary": "on this computer, 3.1 GB"},
    {"name": "ollama/deepseek-v4.1-flash:cloud", "summary": "Ollama cloud"},
]


HISTORY = [
    {"role": "user", "text": "what does the walrus eat?"},
    {"role": "tool", "id": "c1", "name": "shell", "input": {"command": "cat pets.md"}, "ok": True, "preview": "The walrus eats clams."},
    {"role": "assistant", "text": "It eats **clams**."},
    {"role": "notice", "text": "Earlier messages were condensed into a short note."},
]


class FakeClient:
    def __init__(self, events):
        self._events = events
        self.calls = []
        self.open_stream = threading.Event()

    def call(self, method, path, body=None):
        self.calls.append((method, path, body))
        if path == "/session":
            return {"id": "new-sid", "cwd": "/tmp/ws", "model": "anthropic/claude-opus-5-5"}
        if path == "/models":
            return MODELS
        if path.endswith("/history"):
            return {"items": HISTORY, "seq": 7}
        if path == "/sessions":
            return [{"id": "old12345", "cwd": "/tmp/ws", "modified": 1, "messages": 3, "preview": "the old walrus chat"}]
        return {"ok": True}

    def events(self, sid, after, wanted=lambda: True):
        self.streamed_from = after
        open_stream = self.open_stream = threading.Event()
        yield from self._events
        # like the real stream, this one stays open until the client hangs up
        open_stream.wait()

    def hang_up(self):
        self.hung_up = True
        self.open_stream.set()


def transcript_text(app):
    return "\n".join(b.raw for b in app.query(Bubble))


def test_summarize_is_compact():
    assert summarize("shell", {"command": "ls -la\nmore"}) == "ls -la"
    assert summarize("files", {"command": "view", "path": "a.py"}) == "view a.py"
    assert summarize("search", {"query": "refunds"}) == "refunds"


async def test_tui_streams_tools_usage_and_sends():
    events = [
        {"type": "text", "delta": "Hello ", "seq": 1},
        {"type": "text", "delta": "there.", "seq": 2},
        {"type": "tool.call", "id": "t1", "name": "shell", "input": {"command": "ls"}, "seq": 3},
        {"type": "tool.result", "id": "t1", "ok": True, "seq": 4},
        {"type": "usage", "prompt": 1000, "cache_read": 900, "output": 5, "limit": 30000, "steps": 3, "spent": 4200, "seq": 5},
        {"type": "turn.end", "stop": "end", "text": "Hello there.", "seq": 6},
    ]
    client = FakeClient(events)
    app = HypersonicApp(client, "sid1", "/tmp/ws", "anthropic/claude-opus-5-5")
    async with app.run_test() as pilot:
        for _ in range(6):
            await pilot.pause()
        text = transcript_text(app)
        assert "Hello there." in text
        assert "shell  ls" in text
        status = str(app.query_one("#status", Static).render())
        assert "1k/30k" in status and "29k left" in status
        assert "session 4.2k" in status and "3 steps" in status and "cache 90%" in status

        app.query_one("#prompt", Input).value = "run the tests"
        await pilot.press("enter")
        for _ in range(4):
            await pilot.pause()
        sends = [c for c in client.calls if c[1].endswith("/message")]
        assert sends and sends[0][2]["text"] == "run the tests"
        assert "run the tests" in transcript_text(app)
        assert app.query(Bubble).last().role == "user"
        assert app.query_one("#prompt", Input).value == ""


async def test_a_streamed_reply_grows_to_show_every_line():
    wrapped = "word " * 60
    app = HypersonicApp(FakeClient([]), "s", "/tmp/ws", "anthropic/claude-opus-5-5")
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        for delta in ("First line.", "\n\nSecond paragraph."):
            app.handle({"type": "text", "delta": delta})
            await pilot.pause()
        app.handle({"type": "text", "delta": "\n\n" + wrapped})
        reply = app.current
        # checked before anything else can run: the ticker clears this flag a few times a second
        assert reply.raw.endswith(wrapped) and reply.stale
        await pilot.pause(0.3)
        assert not reply.stale and reply.size.height >= 8
        app.handle({"type": "text", "delta": "\n\nLast words."})
        app.handle({"type": "turn.end", "stop": "end", "text": ""})
        await pilot.pause()
        assert not reply.stale and app.current is None


async def test_tui_permission_buttons_post_a_decision():
    events = [{"type": "permission.request", "id": "p9", "name": "shell", "input": {"command": "rm x"}, "seq": 1}]
    client = FakeClient(events)
    app = HypersonicApp(client, "s", "/tmp/ws", "anthropic/claude-opus-5-5")
    async with app.run_test() as pilot:
        for _ in range(4):
            await pilot.pause()
        app.prompts["p9"][0].press()
        await pilot.pause()
        decisions = [c for c in client.calls if c[1].endswith("/permission")]
        assert decisions == [("POST", "/session/s/permission", {"id": "p9", "allow": True})]


async def test_exit_word_quits():
    app = HypersonicApp(FakeClient([]), "s", "/tmp/ws", "anthropic/claude-opus-5-5")
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#prompt", Input).value = "/exit"
        await pilot.press("enter")
        await pilot.pause()
    assert app.is_running is False
    assert app.client.hung_up


MENU_COMMANDS = [
    {"name": "code", "summary": "code mode"},
    {"name": "compact", "summary": "shrink the chat"},
    {"name": "help", "summary": "list commands"},
]


def menu_rows(app):
    return str(app.query_one("#menu", Static).render()).splitlines()


def sent(client):
    return [call[2]["text"] for call in client.calls if call[1].endswith("/message")]


async def test_slash_opens_a_dropdown_with_names_and_summaries():
    app = HypersonicApp(FakeClient([]), "s", "/tmp/ws", "anthropic/claude-opus-5-5", MENU_COMMANDS)
    async with app.run_test() as pilot:
        await pilot.press("/")
        await pilot.pause()
        rows = menu_rows(app)
        assert len(rows) == 3
        assert rows[0].startswith("❯ /code") and "code mode" in rows[0]
        assert "/compact" in rows[1] and "shrink the chat" in rows[1] and "❯" not in rows[1]
        assert app.query_one("#menu", Static).display and not app.query_one("#status", Static).display
        await pilot.press("c", "o")
        await pilot.pause()
        assert len(menu_rows(app)) == 2 and not any("/help" in row for row in menu_rows(app))


async def test_arrows_move_the_choice_and_tab_fills_it_in():
    app = HypersonicApp(FakeClient([]), "s", "/tmp/ws", "anthropic/claude-opus-5-5", MENU_COMMANDS)
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)
        await pilot.press("/", "down")
        await pilot.pause()
        assert menu_rows(app)[1].startswith("❯ /compact")
        await pilot.press("up", "up")
        await pilot.pause()
        assert menu_rows(app)[2].startswith("❯ /help")
        await pilot.press("down", "down", "tab")
        await pilot.pause()
        assert prompt.value == "/compact "
        assert not app.query_one("#menu", Static).display and app.query_one("#status", Static).display
        await pilot.press("n", "o", "w")
        assert prompt.value == "/compact now"


async def test_enter_runs_the_highlighted_command():
    client = FakeClient([])
    app = HypersonicApp(client, "s", "/tmp/ws", "anthropic/claude-opus-5-5", MENU_COMMANDS)
    async with app.run_test() as pilot:
        await pilot.press("/", "h", "enter")
        for _ in range(4):
            await pilot.pause()
        assert sent(client) == ["/help"]
        assert app.query_one("#prompt", Input).value == ""


async def test_escape_closes_the_menu_and_keys_go_back_to_normal():
    client = FakeClient([])
    app = HypersonicApp(client, "s", "/tmp/ws", "anthropic/claude-opus-5-5", MENU_COMMANDS)
    async with app.run_test() as pilot:
        await pilot.press("/", "c", "escape")
        await pilot.pause()
        assert not app.query_one("#menu", Static).display
        await pilot.press("up", "down", "enter")
        for _ in range(4):
            await pilot.pause()
        assert sent(client) == ["/c"]


async def test_a_permission_request_takes_one_key():
    events = [
        {"type": "permission.request", "id": "p1", "name": "shell", "input": {"command": "rm x"}, "seq": 1},
        {"type": "permission.request", "id": "p2", "name": "shell", "input": {"command": "rm y"}, "seq": 2},
    ]
    client = FakeClient(events)
    app = HypersonicApp(client, "s", "/tmp/ws", "anthropic/claude-opus-5-5", MENU_COMMANDS)
    async with app.run_test() as pilot:
        for _ in range(4):
            await pilot.pause()
        assert app.focused is app.prompts["p1"][0]
        assert "y allow" in str(app.query_one("#activity", Static).render())
        await pilot.press("y")
        for _ in range(3):
            await pilot.pause()
        assert app.focused is app.prompts["p2"][0]
        await pilot.press("n")
        for _ in range(3):
            await pilot.pause()
        decisions = [call[2] for call in client.calls if call[1].endswith("/permission")]
        assert decisions == [{"id": "p1", "allow": True}, {"id": "p2", "allow": False}]
        assert app.focused is app.query_one("#prompt", Input)
        await pilot.press("y", "n")
        assert app.query_one("#prompt", Input).value == "yn"


async def test_the_activity_line_follows_the_turn_and_escape_stops_it():
    client = FakeClient([])
    app = HypersonicApp(client, "s", "/tmp/ws", "anthropic/claude-opus-5-5", MENU_COMMANDS)
    async with app.run_test() as pilot:
        await pilot.pause()
        activity = app.query_one("#activity", Static)
        assert not activity.display
        await pilot.press("escape")
        assert not [call for call in client.calls if call[1].endswith("/abort")]

        app.handle({"type": "turn.start"})
        await pilot.pause()
        assert activity.display and "thinking" in str(activity.render()) and "esc to stop" in str(activity.render())
        app.handle({"type": "tool.call", "id": "t1", "name": "shell", "input": {"command": "pytest -q"}})
        await pilot.pause()
        assert "shell  pytest -q" in str(activity.render())
        app.handle({"type": "tool.result", "id": "t1", "ok": True, "seconds": 1.5, "preview": "3 passed"})
        app.handle({"type": "text", "delta": "All " + "good " * 40})
        await pilot.pause()
        assert "writing" in str(activity.render()) and "~51 tokens out" in str(activity.render())

        await pilot.press("escape")
        for _ in range(3):
            await pilot.pause()
        assert ("POST", "/session/s/abort", None) in client.calls
        app.handle({"type": "tool.call", "id": "t2", "name": "shell", "input": {"command": "sleep 60"}})
        app.handle({"type": "permission.request", "id": "p7", "name": "shell", "input": {"command": "sleep 60"}})
        await pilot.pause()
        app.handle({"type": "turn.end", "stop": "aborted", "text": ""})
        await pilot.pause()
        assert not activity.display and app.query(Bubble).last().raw == "stopped"
        assert not app.query(".perm") and app.focused is app.query_one("#prompt", Input)
        assert str(app.tools["t2"].render()).splitlines() == ["  ✗ shell  sleep 60", "      interrupted"]


async def test_tool_lines_show_how_each_call_ended():
    app = HypersonicApp(FakeClient([]), "s", "/tmp/ws", "anthropic/claude-opus-5-5")
    async with app.run_test() as pilot:
        await pilot.pause()
        for tool_id, command in (("t1", "pytest -q"), ("t2", "cat nope.txt")):
            app.handle({"type": "tool.call", "id": tool_id, "name": "shell", "input": {"command": command}})
        running = [str(bubble.render()) for bubble in app.query(Bubble) if bubble.role == "tool"]
        assert all(line.startswith("  … shell") for line in running)
        app.handle({"type": "tool.result", "id": "t1", "ok": True, "seconds": 1.5, "preview": "3 passed\nin 0.2s"})
        app.handle({"type": "tool.result", "id": "t2", "ok": False, "seconds": 0.1, "preview": "cat: nope.txt\nNo such file\n[exit code 1]\nmore"})
        await pilot.pause()
        passed, failed = [str(bubble.render()) for bubble in app.query(Bubble) if bubble.role == "tool"]
        assert passed.splitlines() == ["  ✓ shell  pytest -q  1.5s", "      3 passed"]
        assert failed.splitlines() == ["  ✗ shell  cat nope.txt  0.1s", "      cat: nope.txt", "      No such file", "      [exit code 1]"]


async def test_replies_are_formatted_and_the_screen_opens_with_a_hint():
    app = HypersonicApp(FakeClient([]), "s", "/tmp/my-project", "anthropic/claude-opus-5-5")
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        assert app.query(Bubble).first().role == "hint" and "/rag" in app.query(Bubble).first().raw
        assert "/tmp/my-project" in str(app.query_one("#header", Static).render())
        assert "/ commands" in str(app.query_one("#status", Static).render())
        app.handle({"type": "text", "delta": "The card is **4000 0119**.\n\n- first point\n- second point\n"})
        await pilot.pause()
        await pilot.pause()
        screen = app.export_screenshot()
        assert "4000" in screen and "**" not in screen and "first" in screen


def test_numbers_are_shortened_and_the_meter_fills():
    assert [short(count) for count in (950, 1000, 4206, 30000, 1_250_000)] == ["950", "1k", "4.2k", "30k", "1.2M"]
    assert meter(0, 30000).count("▰") == 0 and meter(15000, 30000).count("▰") == 5
    assert meter(45000, 30000).count("▰") == 10 and RED in meter(29000, 30000)


async def test_model_command_opens_a_list_of_models_to_pick():
    client = FakeClient([])
    app = HypersonicApp(client, "s", "/tmp/ws", "ollama/qwen3.5:2b", [{"name": "model", "summary": "switch"}])
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)
        await pilot.press("/", "m", "o", "enter")
        for _ in range(4):
            await pilot.pause()
        assert prompt.value == "/model " and sent(client) == []
        rows = menu_rows(app)
        assert len(rows) == 3 and "anthropic/claude-opus-5-5" in rows[0]
        assert rows[1].startswith("❯ ollama/qwen3.5:2b") and rows[1].endswith("in use")
        await pilot.press("d", "e", "e", "p")
        await pilot.pause()
        assert menu_rows(app)[0].startswith("❯ ollama/deepseek-v4.1-flash:cloud") and len(menu_rows(app)) == 1
        await pilot.press("enter")
        for _ in range(4):
            await pilot.pause()
        assert sent(client) == ["/model ollama/deepseek-v4.1-flash:cloud"]


async def test_a_model_switch_shows_in_the_status_line_and_new_sessions_keep_it():
    client = FakeClient([{"type": "model", "name": "ollama/qwen3.5:2b", "seq": 1}])
    app = HypersonicApp(client, "s", "/tmp/ws", "anthropic/claude-opus-5-5", [])
    async with app.run_test() as pilot:
        for _ in range(4):
            await pilot.pause()
        assert app.model == "ollama/qwen3.5:2b"
        assert "ollama/qwen3.5:2b" in str(app.query_one("#status", Static).render())
        app.query_one("#prompt", Input).value = "/new"
        await pilot.press("enter")
        for _ in range(5):
            await pilot.pause()
        created = [call[2] for call in client.calls if call[:2] == ("POST", "/session")]
        assert created[0]["model"] == "ollama/qwen3.5:2b"


async def test_a_refused_switch_is_red_and_the_status_names_who_answered():
    events = [
        {"type": "notice", "text": "NOT switched to ollama/x:cloud. Still using anthropic/claude-opus-5-5.", "error": True, "seq": 1},
        {"type": "notice", "text": "Code mode on.", "seq": 2},
        {"type": "usage", "prompt": 10, "cache_read": 0, "output": 1, "model": "claude-opus-5-5", "seq": 3},
    ]
    app = HypersonicApp(FakeClient(events), "s", "/tmp/ws", "anthropic/claude-opus-5-5", [])
    async with app.run_test() as pilot:
        for _ in range(5):
            await pilot.pause()
        said = [(bubble.role, bubble.raw[:12]) for bubble in app.query(Bubble) if bubble.role != "hint"]
        assert said == [("error", "NOT switched"), ("notice", "Code mode on")]
        assert "anthropic/claude-opus-5-5" in str(app.query_one("#status", Static).render())
        app.handle({"type": "rag", "on": True})
        await pilot.pause()
        status = str(app.query_one("#status", Static).render())
        assert "rag on" in status and "ctx 10" in status
        app.handle({"type": "rag", "on": False})
        await pilot.pause()
        assert "rag on" not in str(app.query_one("#status", Static).render())
    assert model_label("ollama/qwen3.5:2b", "qwen3.5:2b") == "ollama/qwen3.5:2b"
    assert model_label("anthropic/claude-opus-5-5", "") == "anthropic/claude-opus-5-5"


async def test_a_long_list_scrolls_with_the_choice():
    many = [{"name": f"cmd{n:02d}", "summary": f"number {n}"} for n in range(12)]
    app = HypersonicApp(FakeClient([]), "s", "/tmp/ws", "anthropic/claude-opus-5-5", many)
    async with app.run_test() as pilot:
        await pilot.press("/")
        await pilot.pause()
        assert len(menu_rows(app)) == MENU_LIMIT + 1 and menu_rows(app)[-1].strip() == "1 of 12"
        await pilot.press("up")
        await pilot.pause()
        rows = menu_rows(app)
        assert rows[MENU_LIMIT - 1].startswith("❯ /cmd11") and rows[-1].strip() == "12 of 12"



async def test_new_starts_fresh_and_resume_brings_the_conversation_back():
    app = HypersonicApp(FakeClient([]), "s0", "/tmp/ws", "anthropic/claude-opus-5-5", [{"name": "resume", "summary": "resume a past session"}])
    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt", Input)
        prompt.value = "/new"
        await pilot.press("enter")
        for _ in range(5):
            await pilot.pause()
        assert app.sid == "new-sid" and [bubble.role for bubble in app.query(Bubble)] == ["hint"]

        await pilot.press("/", "r", "e", "enter")
        for _ in range(5):
            await pilot.pause()
        assert prompt.value == "/resume " and sent(app.client) == []
        assert menu_rows(app) == ["❯ old12345  the old walrus chat · 3 msgs"]
        await pilot.press("w", "a", "l", "r", "u", "s", "enter")
        for _ in range(8):
            await pilot.pause()
        posts = [call[2] for call in app.client.calls if call[:2] == ("POST", "/session")]
        assert posts[-1]["resume"] == "old12345"
        shown = [(bubble.role, bubble.raw) for bubble in app.query(Bubble)]
        assert shown[:4] == [
            ("user", "what does the walrus eat?"),
            ("tool", "shell  cat pets.md"),
            ("assistant", "It eats **clams**."),
            ("notice", "Earlier messages were condensed into a short note."),
        ]
        assert shown[4][0] == "hint" and "Resumed" in shown[4][1] and len(shown) == 5
        assert app.client.streamed_from == 7
        assert str(app.query(Bubble)[1].render()).splitlines() == ["  ✓ shell  cat pets.md", "      The walrus eats clams."]


async def test_up_and_down_bring_back_earlier_messages():
    client = FakeClient([])
    app = HypersonicApp(client, "s", "/tmp/ws", "anthropic/claude-opus-5-5", MENU_COMMANDS)
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)
        await pilot.press("up")
        assert prompt.value == ""
        for text in ("first question", "second question", "first question"):
            prompt.value = text
            await pilot.press("enter")
            await pilot.pause()
        await pilot.press("up")
        assert prompt.value == "first question"
        await pilot.press("up")
        assert prompt.value == "second question"
        await pilot.press("up")
        assert prompt.value == "second question"
        await pilot.press("down", "down")
        assert prompt.value == ""


async def test_a_finished_turn_ends_with_its_time_and_token_totals():
    app = HypersonicApp(FakeClient([]), "s", "/tmp/ws", "anthropic/claude-opus-5-5")
    async with app.run_test() as pilot:
        await pilot.pause()
        app.handle({"type": "turn.start"})
        for prompt_size in (1000, 1400):
            app.handle({"type": "usage", "prompt": prompt_size, "output": 120, "limit": 30000, "steps": 1, "spent": 0})
        app.handle({"type": "text", "delta": "done"})
        app.handle({"type": "turn.end", "stop": "end", "text": "done"})
        await pilot.pause()
        assert app.query(Bubble).last().raw == "✓ 0s · 2 steps · 2.4k tokens in · 240 out"
        app.handle({"type": "turn.end", "stop": "command", "text": ""})
        await pilot.pause()
        assert len([bubble for bubble in app.query(Bubble) if bubble.raw.startswith("✓")]) == 1


NASTY = "grep -rn 'TODO' src/ --include='*.py' | sed -E 's/^[0-9]+://' | sort -u"


def asked(command, pid="p1"):
    return {"type": "permission.request", "id": pid, "name": "shell", "input": {"command": command}}


def shown(app, selector):
    return "\n".join(str(widget.render()) for widget in app.query(selector))


async def test_brackets_and_escape_codes_are_shown_as_text_and_never_acted_on():
    app = HypersonicApp(FakeClient([]), "s", "/tmp/[ws]", "anthropic/claude-opus-5-5")
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        app.handle({"type": "tool.call", "id": "t1", "name": "shell", "input": {"command": NASTY}})
        await pilot.pause(0.3)
        assert "sed -E 's/^[0" in shown(app, "#activity")
        app.handle(asked("rm -rf [X] /important\x1b[2J"))
        app.handle({"type": "tool.result", "id": "t1", "ok": True, "preview": "done\x1b]52;c;aGk=\x1b\\"})
        app.handle({"type": "text", "delta": "all \x1b]0;title\x07good"})
        await pilot.pause(0.3)
        assert "rm -rf [X] /important[2J ?" in shown(app, ".perm Static") and "/tmp/[ws]" in shown(app, "#header")
        assert "\x1b" not in transcript_text(app) + shown(app, "Bubble") + shown(app, ".perm Static")
        assert app.is_running


async def test_the_allow_prompt_shows_the_whole_command_and_keeps_its_buttons_on_screen():
    app = HypersonicApp(FakeClient([]), "s", "/tmp/ws", "anthropic/claude-opus-5-5")
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        app.handle(asked("echo hello\nrm -rf ~/Documents " + "x" * 90, pid="functions.shell:0"))
        await pilot.pause()
        assert "rm -rf ~/Documents" in shown(app, ".perm Static")
        allow, deny = app.prompts["functions.shell:0"]
        assert allow.region.right <= 80 and deny.region.right <= 80


async def test_a_request_that_arrives_while_typing_does_not_take_the_keys():
    client = FakeClient([])
    app = HypersonicApp(client, "s", "/tmp/ws", "anthropic/claude-opus-5-5")
    async with app.run_test() as pilot:
        await pilot.press("c", "a", "n", "space")
        app.handle(asked("rm -rf build"))
        await pilot.pause()
        await pilot.press("y", "o", "u")
        prompt = app.query_one("#prompt", Input)
        assert prompt.value == "can you" and not [call for call in client.calls if call[1].endswith("/permission")]
        allow = app.prompts["p1"][0]
        allow.focus()
        await pilot.press("y")
        for _ in range(3):
            await pilot.pause()
        app.action_answer(True)
        allow.press()
        await pilot.pause()
        assert [call[2] for call in client.calls if call[1].endswith("/permission")] == [{"id": "p1", "allow": True}]
        assert app.focused is prompt and prompt.value == "can you" and app.is_running


async def test_one_event_that_cannot_be_shown_does_not_stop_the_ones_after_it():
    events = [{"type": "text", "seq": 1}, {"type": "text", "delta": "still here", "seq": 2}]
    app = HypersonicApp(FakeClient(events), "s", "/tmp/ws", "anthropic/claude-opus-5-5")
    async with app.run_test() as pilot:
        for _ in range(4):
            await pilot.pause()
        assert "couldn't show a text event" in transcript_text(app) and "still here" in transcript_text(app)
        assert app.after == 2


async def test_a_lost_connection_is_said_and_nothing_is_left_spinning():
    class Dropping(FakeClient):
        def events(self, sid, after, wanted=lambda: True):
            yield {"type": "turn.start", "seq": 1}
            yield {"type": "tool.call", "id": "t1", "name": "shell", "input": {"command": "sleep 9"}, "seq": 2}
            yield asked("rm x")
            raise ConnectionResetError("reset")

    app = HypersonicApp(Dropping([]), "s", "/tmp/ws", "anthropic/claude-opus-5-5")
    async with app.run_test() as pilot:
        for _ in range(5):
            await pilot.pause()
        assert "Lost the connection" in transcript_text(app) and "/resume" in transcript_text(app)
        assert app.busy is False and app.prompts == {} and not app.query(".perm")
        assert "✗ shell" in shown(app, "Bubble.tool")


async def test_resuming_into_a_running_turn_shows_it_as_running():
    class Busy(FakeClient):
        def call(self, method, path, body=None):
            if path.endswith("/history"):
                running = {"role": "tool", "id": "c2", "name": "shell", "input": {"command": "pytest"}, "ok": None, "preview": ""}
                return {"items": [*HISTORY, running], "seq": 9, "busy": True, "rag": True}
            if path == "/session":
                self.calls.append((method, path, body))
                return {"id": "old12345", "cwd": "/tmp/other", "model": "anthropic/claude-opus-5-5"}
            return super().call(method, path, body)

    options = {"cwd": "/tmp/ws", "mode": "auto", "network": True, "trusted": True, "max_steps": 40}
    app = HypersonicApp(Busy([]), "s", "/tmp/ws", "anthropic/claude-opus-5-5", [], options, resumed=True)
    async with app.run_test() as pilot:
        for _ in range(5):
            await pilot.pause()
        assert app.busy and "rag on" in shown(app, "#status") and app.client.streamed_from == 9
        assert "… shell  pytest" in shown(app, "Bubble.tool")
        app.handle({"type": "tool.result", "id": "c2", "ok": True, "seconds": 2.0, "preview": "3 passed"})
        await pilot.pause()
        assert "✓ shell  pytest" in shown(app, "Bubble.tool")
        app.query_one("#prompt", Input).value = "/new"
        await pilot.press("enter")
        for _ in range(5):
            await pilot.pause()
        posted = [call[2] for call in app.client.calls if call[:2] == ("POST", "/session")][-1]
        assert posted == {**options, "model": "anthropic/claude-opus-5-5", "resume": None}
        assert "/tmp/other" in shown(app, "#header")


async def test_recall_keeps_a_draft_and_steps_past_a_slash_command():
    commands = [{"name": "model", "summary": "switch"}, {"name": "mode", "summary": "permissions"}, {"name": "rag", "summary": "documents"}]
    client = FakeClient([])
    app = HypersonicApp(client, "s", "/tmp/ws", "anthropic/claude-opus-5-5", commands)
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)
        for text in ("question", "/rag"):
            prompt.value = text
            await pilot.press("escape", "enter")
            await pilot.pause()
        for key in "my draft":
            await pilot.press("space" if key == " " else key)
        await pilot.press("down")
        assert prompt.value == "my draft"
        await pilot.press("up", "up")
        assert prompt.value == "question"
        await pilot.press("down", "down")
        assert prompt.value == "my draft"
        prompt.value = ""
        await pilot.press("/", "m", "o", "d", "e", "enter")
        await pilot.pause()
        assert sent(client)[-1] == "/mode"
