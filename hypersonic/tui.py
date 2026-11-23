from __future__ import annotations

import time
from typing import Any

from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, Input, Static

from hypersonic.context.squeeze import CHARS_PER_TOKEN
from hypersonic.providers.base import STOP_END

MENU_LIMIT = 8
POINTER = "❯"
COMMAND_LEAD = "/"
MODEL_LEAD = "/model "
RESUME_LEAD = "/resume "
PICKERS = {MODEL_LEAD: "/models", RESUME_LEAD: "/sessions"}
LOADING = "loading…"
EXIT_WORDS = ("/exit", "/quit")
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
TICK_SECONDS = 0.1
BAR_CELLS = 10
BAR_WARN, BAR_FULL = 0.6, 0.85
THOUSAND, MILLION = 1_000, 1_000_000
FAILED_LINES, PASSED_LINES = 3, 1
DOING_CHARS = 60
THINKING, WRITING, ASKING = "thinking", "writing", "waiting for your answer · y allow · n deny"
STOPPED = "stopped"
INTERRUPTED = "interrupted"
ESCAPE = "\x1b"
LOST = "Lost the connection to the server. Type /resume to pick the chat up again."
RESUMED = "Resumed. The conversation so far is above."
WELCOME = (
    "Type a message and press Enter.\n"
    "/  commands   ·   /rag  use your documents   ·   /model  switch model   ·   esc  stop a reply"
)
IDLE_HINT = "/ commands · esc stop · ctrl+c quit"
GAP = " · "
ACCENT, GREEN, RED, AMBER, DIM, PLAIN = "#7aa2f7", "#9ece6a", "#f7768e", "#e0af68", "#565f89", "#c0caf5"
MARKS = {None: ("…", DIM), True: ("✓", GREEN), False: ("✗", RED)}


def model_label(picked: str, answered: str) -> str:
    """The picked model, plus the name the provider itself reported when the two differ."""
    return picked if not answered or picked.endswith(answered) else f"{picked} ({answered})"


def plain(text: str) -> str:
    """Without escape codes, which the terminal would act on (clear the screen, write the clipboard)."""
    return str(text).replace(ESCAPE, "")


def literal(text: str) -> str:
    """Text for a line that uses markup, shown exactly as written. Every [ would otherwise start a tag."""
    return plain(text).replace("[", "\\[")


def summarize(name: str, args: dict[str, Any]) -> str:
    if name == "shell":
        return str(args.get("command", "")).split("\n", 1)[0]
    if name == "files":
        return f"{args.get('command', '')} {args.get('path', '')}".strip()
    if name in ("search", "web"):
        return str(args.get("query") or args.get("url", ""))
    return ""


def short(count: int) -> str:
    """4206 becomes 4.2k, so numbers stay readable in one status line."""
    for size, unit in ((MILLION, "M"), (THOUSAND, "k")):
        if count >= size:
            return f"{count / size:.1f}".removesuffix(".0") + unit
    return str(count)


def meter(used: int, limit: int) -> str:
    share = min(1.0, used / limit)
    colour = RED if share >= BAR_FULL else AMBER if share >= BAR_WARN else GREEN
    filled = round(BAR_CELLS * share)
    return f"[{colour}]{'▰' * filled}[/][{DIM}]{'▱' * (BAR_CELLS - filled)}[/]"


class Bubble(Static):
    """One entry in the conversation: user, assistant (streamed), tool call, notice, hint or error."""

    def __init__(self, role: str, text: str = "") -> None:
        super().__init__()
        self.role = role
        self.raw = plain(text)
        self.passed: bool | None = None
        self.seconds = 0.0
        self.preview = ""
        self.stale = False
        self.add_class(role)

    def render(self):
        if self.role == "assistant":
            grid = Table.grid(padding=(0, 1))
            grid.add_column(width=1)
            grid.add_column(ratio=1)
            grid.add_row(Text("●", style=GREEN), Markdown(self.raw))
            return grid
        if self.role == "user":
            return Text.assemble((f"{POINTER} ", f"bold {ACCENT}"), (self.raw, "bold"))
        if self.role == "tool":
            return self.tool_line()
        return Text(self.raw, style=RED if self.role == "error" else DIM)

    def tool_line(self) -> Text:
        mark, colour = MARKS[self.passed]
        name, _, detail = self.raw.partition("  ")
        line = Text.assemble((f"  {mark} ", colour), (name, f"bold {PLAIN}"), (f"  {detail}", DIM))
        if self.passed is None:
            return line
        if self.seconds:
            line.append(f"  {self.seconds:g}s", DIM)
        keep = PASSED_LINES if self.passed else FAILED_LINES
        for row in self.preview.strip().splitlines()[:keep]:
            line.append(f"\n      {row}", DIM if self.passed else RED)
        return line

    def append(self, delta: str) -> None:
        self.raw += plain(delta)
        self.stale = True

    def redraw(self) -> None:
        """Formatting a long reply is slow, so the app calls this a few times a second, not once per word."""
        if self.stale:
            self.stale = False
            self.refresh(layout=True)

    def finish(self, passed: bool, seconds: float, preview: str) -> None:
        self.passed, self.seconds, self.preview = passed, seconds, plain(preview)
        self.refresh(layout=True)


class HypersonicApp(App):
    ENABLE_COMMAND_PALETTE = False
    CSS = """
    Screen { background: #0b0e14; }
    #header { height: 1; padding: 0 2; background: #11151c; }
    #transcript { padding: 0 2 1 2; }
    Bubble { margin-top: 1; }
    Bubble.tool { margin-top: 0; text-wrap: nowrap; text-overflow: ellipsis; }
    .perm { height: auto; margin-top: 1; }
    .perm Static { width: 1fr; padding: 1 1 0 0; }
    Button { margin: 0 1; min-width: 10; }
    #activity { height: 1; padding: 0 2; color: #7aa2f7; display: none; }
    #prompt { border: round #2a313c; background: #161b22; }
    #menu { padding: 0 2; height: auto; display: none; text-wrap: nowrap; text-overflow: ellipsis; }
    #status { color: #565f89; padding: 0 2; height: 1; text-wrap: nowrap; text-overflow: ellipsis; }
    """
    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+q", "quit", "Quit"),
        Binding("tab", "complete", "Complete command", priority=True),
        Binding("down", "menu_move(1)", "Next command", priority=True),
        Binding("up", "menu_move(-1)", "Previous command", priority=True),
        Binding("escape", "escape", "Close the menu or stop the reply", priority=True),
        Binding("y", "answer(True)", "Allow"),
        Binding("n", "answer(False)", "Deny"),
    ]

    def __init__(
        self, client: Any, sid: str, cwd: str, model: str, commands: list | None = None,
        options: dict[str, Any] | None = None, resumed: bool = False,
    ) -> None:
        super().__init__()
        self.client = client
        self.sid = sid
        self.cwd = cwd
        self.model = model
        self.commands = commands or []
        # what the app was started with (mode, network, trust, budget), so /new and /resume keep it
        self.options = options or {"cwd": cwd, "mode": "ask"}
        self.resumed = resumed
        self.draft = ""
        self.current: Bubble | None = None
        self.stream_gen = 0
        self.found: list = []
        self.menu_index = 0
        self.lead = COMMAND_LEAD
        self.rag = False
        self.answered = ""
        self.usage: dict[str, Any] = {}
        self.picks: dict[str, list] = {}
        self.picking = ""
        self.loading = False
        self.sent: list[str] = []
        self.recall = 0
        self.turn = {"steps": 0, "in": 0, "out": 0}
        self.after = 0
        self.busy = False
        self.doing = THINKING
        self.began = 0.0
        self.typed = 0
        self.tools: dict[str, Bubble] = {}
        # each waiting request, in the order asked: the call's id and its Allow and Deny buttons
        self.prompts: dict[str, tuple[Button, Button]] = {}

    def header_text(self) -> str:
        return f"[b {ACCENT}]Hypersonic[/]  [{DIM}]{literal(self.cwd)}[/]"

    def compose(self) -> ComposeResult:
        yield Static(self.header_text(), id="header")
        yield VerticalScroll(id="transcript")
        yield Static("", id="activity")
        yield Input(placeholder="Message Hypersonic…", id="prompt", select_on_focus=False)
        yield Static("", id="menu")
        yield Static(self.status_text(), id="status")

    def status_text(self) -> str:
        parts = [f"[{PLAIN}]{literal(model_label(self.model, self.answered))}[/]"]
        if self.rag:
            parts.append(f"[{GREEN}]rag on[/]")
        used, limit = self.usage.get("prompt", 0), self.usage.get("limit", 0)
        if not self.usage:
            return GAP.join([*parts, IDLE_HINT])
        if limit:
            parts += [f"ctx {meter(used, limit)} {short(used)}/{short(limit)}", f"{short(max(0, limit - used))} left"]
        else:
            parts.append(f"ctx {short(used)}")
        cached = round(100 * self.usage.get("cache_read", 0) / used) if used else 0
        parts += [f"session {short(self.usage.get('spent', 0))}", f"{self.usage.get('steps', 0)} steps", f"cache {cached}%"]
        return GAP.join(parts)

    def show_status(self) -> None:
        self.query_one("#status", Static).update(self.status_text())

    def activity_text(self) -> str:
        spent = time.monotonic() - self.began
        parts = [f"{SPINNER[int(spent / TICK_SECONDS) % len(SPINNER)]} {literal(self.doing)}", f"{int(spent)}s"]
        if self.typed:
            parts.append(f"~{short(self.typed // CHARS_PER_TOKEN)} tokens out")
        return GAP.join([*parts, f"[{DIM}]esc to stop[/]"])

    def tick(self) -> None:
        self.show_activity()
        if self.current:
            self.current.redraw()

    def show_activity(self) -> None:
        # a tick can still arrive while the screen is closing, when this line is already gone
        for line in self.query("#activity"):
            line.update(self.activity_text())

    def set_busy(self, doing: str | None) -> None:
        """Show what the agent is doing right now, or hide the line when the turn is over."""
        if doing and not self.busy:
            self.began, self.typed = time.monotonic(), 0
            self.turn = {"steps": 0, "in": 0, "out": 0}
        self.busy, self.doing = doing is not None, doing or THINKING
        self.ticker.resume() if self.busy else self.ticker.pause()
        self.query_one("#activity", Static).display = self.busy
        if self.busy:
            self.show_activity()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == "menu_move":
            return bool(self.found) or (bool(self.sent) and self.focused is self.query_one("#prompt", Input))
        if action == "complete":
            return bool(self.found)
        if action == "escape":
            return bool(self.found) or self.busy
        if action == "answer":
            return bool(self.prompts) and isinstance(self.focused, Button)
        return True

    @on(Input.Changed, "#prompt")
    def refresh_menu(self, event: Input.Changed) -> None:
        if event.value not in self.sent:
            self.recall = 0
        self.offer(event.value)

    def offer(self, value: str) -> None:
        was_picking, self.picking = self.picking, next((lead for lead in PICKERS if value.startswith(lead)), "")
        if self.picking and self.picking != was_picking:
            self.loading = True
            self.load_picks(self.picking)
        # the row already chosen stays chosen when the list is filtered or reloaded; a fresh list opens on the one in use
        same_list = bool(self.picking) and self.picking == was_picking and bool(self.found)
        want = self.found[self.menu_index]["name"] if same_list else self.model
        if self.picking:
            typed = value[len(self.picking) :].lower()
            rows = self.picks.get(self.picking, [])
            self.lead, self.found = self.picking, [row for row in rows if typed in f"{row['name']} {row['summary']}".lower()]
        elif value.startswith(COMMAND_LEAD) and " " not in value:
            want = value[len(COMMAND_LEAD) :].lower()
            self.lead, self.found = COMMAND_LEAD, [c for c in self.commands if c["name"].lower().startswith(want)]
        else:
            self.found = []
        names = [row["name"] for row in self.found]
        self.menu_index = names.index(want) if want in names else 0
        self.draw_menu()

    @work(thread=True)
    def load_picks(self, lead: str) -> None:
        try:
            rows = self.client.call("GET", PICKERS[lead])
        except Exception:
            rows = []
        if lead == RESUME_LEAD:
            rows = [{"name": row["id"], "summary": f"{' '.join((row.get('preview') or '(no messages)').split())} · {row['messages']} msgs"} for row in rows]
        self.call_from_thread(self.show_picks, lead, rows)

    def show_picks(self, lead: str, rows: list) -> None:
        self.picks[lead], self.loading = rows, False
        self.offer(self.query_one("#prompt", Input).value)

    def draw_menu(self) -> None:
        menu = self.query_one("#menu", Static)
        waiting = bool(self.picking) and self.loading and not self.found
        menu.display = bool(self.found) or waiting
        self.query_one("#status", Static).display = not menu.display
        first = max(0, self.menu_index - MENU_LIMIT + 1)
        visible = self.found[first : first + MENU_LIMIT]
        shown = COMMAND_LEAD if self.lead == COMMAND_LEAD else ""
        width = max((len(c["name"]) for c in visible), default=0) + len(shown)
        rows = [f"  [{DIM}]{LOADING}[/]"] if waiting else []
        for position, command in enumerate(visible, first):
            name, summary = literal(f"{shown}{command['name']}".ljust(width)), literal(command.get("summary", ""))
            if self.picking == MODEL_LEAD and command["name"] == self.model:
                summary += " · in use"
            if position == self.menu_index:
                rows.append(f"[b {ACCENT}]{POINTER} {name}  {summary}[/]")
            else:
                rows.append(f"  [{PLAIN}]{name}[/]  [{DIM}]{summary}[/]")
        if len(self.found) > MENU_LIMIT:
            rows.append(f"  [{DIM}]{self.menu_index + 1} of {len(self.found)}[/]")
        menu.update("\n".join(rows))

    def action_menu_move(self, step: int) -> None:
        # while earlier messages are being stepped through, a recalled /command must not hand the arrows to its menu
        if self.found and not self.recall:
            self.menu_index = (self.menu_index + step) % len(self.found)
            self.draw_menu()
            return
        target = max(0, min(len(self.sent), self.recall - step))
        if target == self.recall:
            return
        if not self.recall:
            self.draft = self.query_one("#prompt", Input).value
        self.recall = target
        self.fill(self.sent[-target] if target else self.draft)

    def action_escape(self) -> None:
        if self.found:
            self.found = []
            self.draw_menu()
        else:
            self.stop()

    def action_complete(self) -> None:
        self.fill(self.lead + self.found[self.menu_index]["name"] + (" " if self.lead == COMMAND_LEAD else ""))

    def action_answer(self, allow: bool) -> None:
        pair = next((pair for pair in self.prompts.values() if self.focused in pair), None)
        if pair:
            pair[0 if allow else 1].press()

    def fill(self, text: str) -> None:
        prompt = self.query_one("#prompt", Input)
        prompt.value = text
        prompt.cursor_position = len(text)

    def on_mount(self) -> None:
        self.ticker = self.set_interval(TICK_SECONDS, self.tick, pause=True)
        self.query_one("#transcript", VerticalScroll).anchor()
        self.query_one("#prompt", Input).focus()
        if self.resumed:
            self.load_history(self.stream_gen)
        else:
            self.say("hint", WELCOME)
            self.stream_events()

    def on_unmount(self) -> None:
        self.ticker.stop()
        self.stream_gen += 1
        self.client.hang_up()

    def say(self, role: str, text: str = "") -> Bubble:
        bubble = Bubble(role, text)
        self.query_one("#transcript", VerticalScroll).mount(bubble)
        return bubble

    @on(Input.Submitted, "#prompt")
    def submit(self, event: Input.Submitted) -> None:
        text = self.lead + self.found[self.menu_index]["name"] if self.found else event.value.strip()
        if f"{text} " in PICKERS:
            self.fill(f"{text} ")
            return
        self.query_one("#prompt", Input).value = ""
        if not text:
            return
        self.sent, self.recall = [*[old for old in self.sent if old != text], text], 0
        if text in EXIT_WORDS:
            self.exit()
            return
        if text.startswith("/"):
            cmd, _, arg = text[1:].partition(" ")
            if cmd in ("new", "clear"):
                self.new_session(None)
                return
            if cmd == "resume":
                self.new_session(arg.strip())
                return
        self.say("user", text)
        self.query_one("#transcript", VerticalScroll).anchor()
        self.close_reply()
        self.send(text)

    @work(thread=True)
    def send(self, text: str) -> None:
        try:
            self.client.call("POST", f"/session/{self.sid}/message", {"text": text, "attachments": []})
        except Exception as error:
            self.call_from_thread(self.say, "error", f"send failed: {error}")

    @work(thread=True)
    def decide(self, pid: str, allow: bool) -> None:
        try:
            self.client.call("POST", f"/session/{self.sid}/permission", {"id": pid, "allow": allow})
        except Exception:
            pass

    @work(thread=True)
    def stop(self) -> None:
        try:
            self.client.call("POST", f"/session/{self.sid}/abort")
        except Exception as error:
            self.call_from_thread(self.say, "error", f"couldn't stop: {error}")

    @work(thread=True)
    def stream_events(self) -> None:
        mine = self.stream_gen
        sid = self.sid
        problem = LOST
        try:
            for event in self.client.events(sid, self.after, lambda: self.stream_gen == mine):
                if self.stream_gen != mine:
                    return
                self.call_from_thread(self.show, event, mine)
        except Exception as error:
            problem = f"{LOST} ({error})"
        if self.stream_gen == mine:
            self.call_from_thread(self.lost, problem)

    def show(self, event: dict[str, Any], gen: int) -> None:
        """One event that can't be drawn must not end the stream for every event after it."""
        try:
            self.handle(event, gen)
        except Exception as error:
            self.say("error", f"couldn't show a {event.get('type')} event: {error}")

    def lost(self, problem: str) -> None:
        self.settle()
        self.set_busy(None)
        self.say("error", problem)

    @work(thread=True)
    def new_session(self, resume_id: str | None) -> None:
        try:
            created = self.client.call("POST", "/session", {**self.options, "model": self.model, "resume": resume_id})
        except Exception as error:
            self.call_from_thread(self.say, "error", f"couldn't start session: {error}")
            return
        self.call_from_thread(self._switch, created, resume_id)

    def _switch(self, created: dict[str, Any], resume_id: str | None) -> None:
        self.stream_gen += 1
        self.client.hang_up()
        self.sid = created["id"]
        self.model, self.cwd = created.get("model", self.model), created.get("cwd", self.cwd)
        self.current = None
        self.rag, self.answered, self.usage = False, "", {}
        self.tools, self.prompts, self.after = {}, {}, 0
        self.set_busy(None)
        self.query_one("#transcript", VerticalScroll).remove_children()
        self.query_one("#header", Static).update(self.header_text())
        self.show_status()
        if resume_id:
            self.load_history(self.stream_gen)
        else:
            self.say("hint", WELCOME)
            self.stream_events()

    @work(thread=True)
    def load_history(self, mine: int) -> None:
        try:
            past = self.client.call("GET", f"/session/{self.sid}/history")
        except Exception as error:
            past = {"items": [{"role": "error", "text": f"couldn't load the earlier conversation: {error}"}]}
        self.call_from_thread(self.show_history, past, mine)

    def show_history(self, past: dict[str, Any], mine: int) -> None:
        if mine != self.stream_gen:
            return
        for entry in past["items"]:
            if entry["role"] != "tool":
                self.say(entry["role"], entry["text"])
                continue
            line = self.say("tool", f"{entry['name']}  {summarize(entry['name'], entry['input'])}")
            if entry["ok"] is None:
                # still running: its result will arrive as an event
                self.tools[entry["id"]] = line
            else:
                line.finish(entry["ok"], 0.0, entry["preview"])
        self.say("hint", RESUMED)
        self.rag = past.get("rag", False)
        self.show_status()
        if past.get("busy"):
            self.set_busy(THINKING)
        # what is above already covers everything up to here, so only newer events are streamed
        self.after = past.get("seq", 0)
        self.stream_events()

    def handle(self, ev: dict[str, Any], gen: int | None = None) -> None:
        # a stale event queued just before a session switch must not land in the new transcript
        if gen is not None and gen != self.stream_gen:
            return
        self.after = ev.get("seq", self.after)
        kind = ev.get("type")
        if kind == "turn.start":
            self.set_busy(THINKING)
        elif kind == "text":
            fresh = self.current is None
            if fresh:
                self.current = self.say("assistant")
            self.current.append(ev["delta"])
            if fresh:
                self.current.redraw()
            self.typed += len(ev["delta"])
            if not self.busy or self.doing != WRITING:
                self.set_busy(WRITING)
        elif kind == "tool.call":
            label = f"{ev['name']}  {summarize(ev['name'], ev['input'])}"
            self.close_reply()
            self.tools[ev["id"]] = self.say("tool", label)
            self.set_busy(label[:DOING_CHARS])
        elif kind == "tool.result":
            if ev["id"] in self.tools:
                self.tools[ev["id"]].finish(ev["ok"], ev.get("seconds", 0.0), ev.get("preview", ""))
            self.set_busy(ASKING if self.prompts else THINKING)
        elif kind == "permission.request":
            self.ask_permission(ev)
        elif kind == "usage":
            self.usage, self.answered = ev, ev.get("model", "")
            self.turn = {"steps": self.turn["steps"] + 1, "in": self.turn["in"] + ev.get("prompt", 0), "out": self.turn["out"] + ev.get("output", 0)}
            self.show_status()
        elif kind == "model":
            self.model, self.answered = ev["name"], ""
            self.show_status()
        elif kind == "rag":
            self.rag = ev["on"]
            self.show_status()
        elif kind == "notice":
            self.say("error" if ev.get("error") else "notice", ev["text"])
        elif kind == "error":
            self.say("error", ev["message"])
        elif kind == "compaction":
            self.say("notice", "the chat was condensed to keep the context small")
        elif kind == "turn.end":
            self.close_reply()
            self.settle()
            if ev.get("stop") == "aborted":
                self.say("notice", STOPPED)
            elif self.turn["steps"] and self.busy:
                took = int(time.monotonic() - self.began)
                ended = "✓" if ev.get("stop") == STOP_END else f"✗ ended early ({ev.get('stop')}) ·"
                self.say("hint", f"{ended} {took}s · {self.turn['steps']} steps · {short(self.turn['in'])} tokens in · {short(self.turn['out'])} out")
            self.turn = {"steps": 0, "in": 0, "out": 0}
            self.set_busy(None)

    def close_reply(self) -> None:
        if self.current:
            self.current.redraw()
        self.current = None

    def settle(self) -> None:
        """Nothing can still be running or waiting once the turn is over."""
        for bubble in self.tools.values():
            if bubble.passed is None:
                bubble.finish(False, 0.0, INTERRUPTED)
        if self.prompts:
            self.query(".perm").remove()
            self.prompts = {}
            self.query_one("#prompt", Input).focus()

    def ask_permission(self, ev: dict[str, Any]) -> None:
        row = Horizontal(classes="perm")
        self.query_one("#transcript", VerticalScroll).mount(row)
        # the whole command, every line of it: this is what the user is agreeing to
        asked = ev["input"].get("command", "") if ev["name"] == "shell" else summarize(ev["name"], ev["input"])
        allow, deny = Button("Allow (y)", variant="success"), Button("Deny (n)", variant="error")
        row.mount(Static(f"[{AMBER}]allow[/] {literal(ev['name'])}: {literal(asked)} ?"), allow, deny)
        self.prompts[ev["id"]] = (allow, deny)
        self.set_busy(ASKING)
        # a half-typed message keeps the keyboard, or its next letters would answer the request
        if len(self.prompts) == 1 and not self.query_one("#prompt", Input).value:
            allow.focus()

    @on(Button.Pressed)
    def on_decision(self, event: Button.Pressed) -> None:
        pid = next((pid for pid, pair in self.prompts.items() if event.button in pair), None)
        if pid is None:
            return
        self.decide(pid, self.prompts.pop(pid)[0] is event.button)
        event.button.parent.remove()
        if self.prompts:
            next(iter(self.prompts.values()))[0].focus()
        else:
            self.query_one("#prompt", Input).focus()
            self.set_busy(THINKING)


def run_tui(client: Any, opened: dict[str, Any], options: dict[str, Any], resumed: bool) -> None:
    try:
        commands = client.call("GET", f"/session/{opened['id']}/commands")
    except Exception:
        commands = []
    HypersonicApp(client, opened["id"], opened["cwd"], opened.get("model", "?"), commands, options, resumed).run()
