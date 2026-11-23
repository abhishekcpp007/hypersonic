"""Rough tests for Hypersonic: the real server and sandbox, driven through the real client by a stand-in model.

    uv run python evals/stress/run.py                 # every round
    uv run python evals/stress/run.py abort sandbox   # only the rounds named

Nothing here needs a model or the network. Everything it writes goes under .stress/ in the current folder.
"""
from __future__ import annotations

import http.client
import json
import os
import queue
import random
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path.cwd() / ".stress"
shutil.rmtree(ROOT, ignore_errors=True)
HOME, FAKE_HOME, OUTSIDE = ROOT / "home", ROOT / "user-home", ROOT / "outside"
for folder in (HOME / "providers", FAKE_HOME / ".ssh", OUTSIDE):
    folder.mkdir(parents=True)
(FAKE_HOME / ".ssh" / "id_test").write_text("FAKE-PRIVATE-KEY")
(FAKE_HOME / ".zshrc").write_text("# untouched\n")
(OUTSIDE / "target.txt").write_text("untouched")
shutil.copy(Path(__file__).with_name("chaos.py"), HOME / "providers" / "chaos.py")
os.environ.update(HYPERSONIC_HOME=str(HOME), HOME=str(FAKE_HOME))
os.environ.pop("HYPERSONIC_MODEL", None)

from hypersonic import cli  # noqa: E402
from hypersonic.engine.loop import INTERRUPTED  # noqa: E402
from hypersonic.engine.session import Session  # noqa: E402

MODEL = "chaos/model"
TURN_SECONDS = 90
SERVER_ERROR = 500
MEGABYTE = 1024 * 1024
LONG_NAP, SHORT_NAP = "sleep 613", "sleep 271"
results: list[tuple[str, bool]] = []
known: list[cli.Client] = []
random.seed(7)


def check(name: str, ok: bool, detail: object = "") -> bool:
    results.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"   <- {str(detail)[:300]}" if not ok else ""), flush=True)
    return bool(ok)


def server() -> cli.Client:
    """A fresh client for the one server under test (each has its own event stream to hang up)."""
    if not known:
        known.append(cli.connect())
    return cli.Client(known[0].port, known[0].headers["Authorization"].removeprefix("Bearer "))


def server_pid() -> int:
    return json.loads((HOME / "server.json").read_text())["pid"]


def memory_mb(pid: int) -> float:
    return int(subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout or 0) / 1024


def open_files(pid: int) -> int:
    proc = Path(f"/proc/{pid}/fd")
    if proc.is_dir():
        return len(list(proc.iterdir()))
    return len(subprocess.run(["lsof", "-p", str(pid)], capture_output=True, text=True).stdout.splitlines())


def running(pattern: str) -> list[str]:
    return subprocess.run(["pgrep", "-fx", pattern], capture_output=True, text=True).stdout.split()


def workspace(name: str) -> Path:
    folder = ROOT / f"ws-{name}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def start(folder: Path, **extra: object) -> str:
    return server().call("POST", "/session", {"cwd": str(folder), "model": MODEL, "mode": "auto", **extra})["id"]


def post(sid: str, text: str) -> int:
    return server().call("POST", f"/session/{sid}/message", {"text": text})["seq"]


def follow(sid: str, after: int, allow: bool | None = None, on_event=None, wait: float = TURN_SECONDS) -> dict:
    """Read one turn's events to its end. Returns the events, the final text and how it stopped."""
    reader = server()
    inbox: queue.Queue = queue.Queue()

    def pump() -> None:
        try:
            for event in reader.events(sid, after):
                inbox.put(event)
        except Exception as error:
            inbox.put({"type": "stream.error", "message": repr(error)})

    threading.Thread(target=pump, daemon=True).start()
    seen, deadline = [], time.monotonic() + wait
    try:
        while True:
            try:
                event = inbox.get(timeout=max(0.01, deadline - time.monotonic()))
            except queue.Empty:
                return {"events": seen, "stop": "TIMED OUT", "text": ""}
            seen.append(event)
            if on_event:
                on_event(event)
            if event["type"] == "permission.request" and allow is not None:
                server().call("POST", f"/session/{sid}/permission", {"id": event["id"], "allow": allow})
            if event["type"] in ("turn.end", "stream.error"):
                return {"events": seen, "stop": event.get("stop", "STREAM ERROR"), "text": event.get("text", "")}
    finally:
        reader.hang_up()


def turn(sid: str, text: str, **more: object) -> dict:
    return follow(sid, post(sid, text), **more)


def ask(sid: str, order: str) -> str:
    """One order, with every permission request approved, so only the sandbox stands in the way."""
    return turn(sid, order, allow=True)["text"]


def status_of(method: str, path: str, body: object, token: str | None = None) -> int:
    client = server()
    headers = dict(client.headers) if token is None else {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    connection = http.client.HTTPConnection(cli.HOST, client.port, timeout=30)
    try:
        payload = body if isinstance(body, (bytes, type(None))) else json.dumps(body).encode()
        connection.request(method, path, payload, headers)
        response = connection.getresponse()
        response.read()
        return response.status
    finally:
        connection.close()


# ---------------------------------------------------------------- rounds


def round_concurrency() -> None:
    sessions, steps = 24, 15
    slowest, stop_pinging = [0.0], threading.Event()

    def ping() -> None:
        while not stop_pinging.is_set():
            began = time.monotonic()
            server().call("GET", "/health")
            slowest[0] = max(slowest[0], time.monotonic() - began)
            time.sleep(0.05)

    outcome: dict[int, dict] = {}

    def work(number: int) -> None:
        sid = start(workspace(f"conc-{number}"))
        outcome[number] = {**turn(sid, f"steps {steps}"), "sid": sid}

    threading.Thread(target=ping, daemon=True).start()
    threads = [threading.Thread(target=work, args=(number,)) for number in range(sessions)]
    began = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    stop_pinging.set()
    wanted = [f"step-{number}" for number in range(1, steps + 1)]
    bad = [n for n in range(sessions) if outcome.get(n, {}).get("text") != f"done {steps}"
           or (workspace(f"conc-{n}") / "steps.txt").read_text().split() != wanted]
    check(f"concurrency: {sessions} chats ran {steps} tool steps each at once, none mixed up ({time.monotonic() - began:.1f}s)", not bad, bad)
    check(f"concurrency: the server kept answering (slowest health check {slowest[0]:.2f}s)", slowest[0] < 3, slowest[0])
    replayed = [len(Session.load(outcome[n]["sid"]).messages) for n in range(sessions)]
    check("concurrency: every chat's log replays to the full conversation", set(replayed) == {2 * steps + 2}, replayed)
    shared = workspace("shared")
    sids = [start(shared) for _ in range(8)]
    done = {}
    threads = [threading.Thread(target=lambda s=s: done.update({s: ask(s, "steps 10")})) for s in sids]
    [thread.start() for thread in threads]
    [thread.join() for thread in threads]
    lines = (shared / "steps.txt").read_text().split()
    check("concurrency: 8 chats in ONE folder all finish and no write is lost", set(done.values()) == {"done 10"} and len(lines) == 80, (done, len(lines)))


def round_abort() -> None:
    sid = start(workspace("abort"))
    bad = []
    for number in range(25):
        seq = post(sid, "sleep")
        began = [0.0]

        def stop_soon(event: dict) -> None:
            if event["type"] == "tool.call":
                time.sleep(random.random() * 0.4)
                began[0] = time.monotonic()
                server().call("POST", f"/session/{sid}/abort")

        got = follow(sid, seq, on_event=stop_soon, wait=15)
        ends = [event for event in got["events"] if event["type"] == "turn.end"]
        if got["stop"] != "aborted" or len(ends) != 1 or time.monotonic() - began[0] > 5:
            bad.append((number, got["stop"], len(ends)))
    check("abort: 25 stops at random moments each end the turn once, within 5s", not bad, bad)
    check("abort: the chat still answers afterwards", ask(sid, "echo alive") == "alive")
    time.sleep(1)
    check("abort: no stopped command is left running", not running(LONG_NAP), running(LONG_NAP))
    bad = []
    for number in range(10):
        seq = post(sid, "slowtext")
        threading.Timer(0.2 + random.random() * 0.5, lambda: server().call("POST", f"/session/{sid}/abort")).start()
        got = follow(sid, seq, wait=15)
        if got["stop"] != "aborted":
            bad.append((number, got["stop"]))
    check("abort: 10 stops in the middle of streamed text", not bad, bad)
    server().call("POST", f"/session/{sid}/abort")
    server().call("POST", f"/session/{sid}/abort")
    roles = [message["role"] for message in Session.load(sid).messages]
    check("abort: stopping an idle chat is harmless and the history still alternates",
          ask(sid, "echo still fine") == "still fine" and all(a != b or a == "user" for a, b in zip(roles, roles[1:])), roles[-8:])


def round_garbage() -> None:
    sid = start(workspace("garbage"))
    got = turn(sid, "garbage")
    ended = [event for event in got["events"] if event["type"] == "tool.result"]
    check("garbage: 45 broken tool calls come back as errors and the turn finishes", got["text"] == "survived the garbage" and len(ended) == 45, (got["stop"], len(ended)))
    check("garbage: nothing was reported as a harness error", not [event for event in got["events"] if event["type"] == "error"])
    check("garbage: a tool call cut off by the output limit is never run", ask(sid, "partial") == "partial done" and not (workspace("garbage") / "should-not-exist.txt").exists())
    check("garbage: a model that fails ends the turn cleanly", turn(sid, "raise")["stop"] == "error" and ask(sid, "echo ok") == "ok")
    check("garbage: a model that only sends blanks is stopped", turn(sid, "blank")["stop"] == "error" and ask(sid, "echo ok") == "ok")
    check("garbage: 40 tool calls in one reply", "steps.txt" not in ask(sid, "parallel 40") and ask(sid, "echo ok") == "ok")


def round_flood() -> None:
    sid = start(workspace("flood"))
    pid = server_pid()
    before = memory_mb(pid)
    began = time.monotonic()
    big = ask(sid, "shell yes hypersonic | head -c 300000000")
    check(f"flood: 300 MB of output is cut to a short result ({len(big):,} chars, {time.monotonic() - began:.1f}s)", 0 < len(big) < 20_000 and "omitted" in big, big[:200])
    saved = sorted((HOME / "sessions" / sid / "out").glob("*.txt"), key=lambda path: path.stat().st_size)
    check("flood: the saved copy stops at the size cap", saved and saved[-1].stat().st_size < 60 * MEGABYTE, [path.stat().st_size for path in saved])
    check("flood: binary output", "exit code" not in ask(sid, "shell head -c 3000000 /dev/urandom")[-60:])
    check("flood: output that is not valid text", bool(ask(sid, "shell python3 -c \"import sys; sys.stdout.buffer.write(bytes([255, 254]) * 100000)\"")))
    check("flood: one 20 MB line", "omitted" in ask(sid, "shell head -c 20000000 /dev/zero | tr '\\0' a"))
    colours = ask(sid, "shell printf '\\033[2J\\033]52;c;aGk=\\007 plain \\033[31mred\\033[0m'")
    check("flood: terminal escape codes are removed from results", "\x1b" not in colours and "red" in colours, repr(colours))
    began = time.monotonic()
    spawned = ask(sid, f"shell for i in $(seq 1 150); do {SHORT_NAP} & done; echo spawned")
    check(f"flood: 150 background jobs don't hold the command ({time.monotonic() - began:.1f}s)", spawned.startswith("spawned") and time.monotonic() - began < 15, spawned[:120])
    grown = memory_mb(pid) - before
    check(f"flood: the server's memory stayed flat ({grown:+.0f} MB)", grown < 300, grown)


def round_queue() -> None:
    sid = start(workspace("queue"))
    seq = post(sid, "hold 3")
    sent = []

    def pile_on(event: dict) -> None:
        if event["type"] == "tool.call" and not sent:
            for number in range(40):
                sent.append(server().call("POST", f"/session/{sid}/message", {"text": f"echo q-{number}"})["queued"])

    got = follow(sid, seq, on_event=pile_on)
    time.sleep(0.5)
    typed = [block["text"] for message in Session.load(sid).messages if message["role"] == "user"
             for block in message["content"] if block.get("type") == "text" and block["text"].startswith("echo q-")]
    check("queue: 40 messages sent during a tool call are all queued", sent == [True] * 40, sent)
    check("queue: each reaches the model exactly once, in order", typed == [f"echo q-{n}" for n in range(40)] and got["text"] == "q-39", (len(typed), got["text"]))


def round_crash() -> None:
    sid = start(workspace("crash"))
    ask(sid, "steps 3")
    seq = post(sid, "sleep")
    follow(sid, seq, on_event=lambda event: event["type"] == "tool.call" and os.kill(server_pid(), signal.SIGKILL), wait=10)
    known.clear()
    time.sleep(0.5)
    again = server().call("POST", "/session", {"cwd": str(workspace("crash")), "model": MODEL, "mode": "auto", "resume": sid})["id"]
    check("crash: after kill -9 a new server starts and the chat resumes", again == sid and ask(sid, "echo back") == "back")
    closed = [block for message in Session.load(sid).messages for block in message["content"] if block.get("type") == "tool_result" and block.get("content") == INTERRUPTED]
    check("crash: the tool call that was cut off is closed in the history", len(closed) == 1, closed)
    time.sleep(3)
    check("crash: the command the dead server was running does not live on", not running(LONG_NAP), running(LONG_NAP))
    for pid in running(LONG_NAP):
        os.kill(int(pid), signal.SIGKILL)


def drain(reader: cli.Client, sid: str, after: int, dropped: threading.Event) -> None:
    try:
        list(reader.events(sid, after, lambda: not dropped.is_set()))
    except (OSError, http.client.HTTPException, cli.ClientError):
        pass  # the line was dropped on purpose


def round_reconnect() -> None:
    sid = start(workspace("reconnect"))
    ask(sid, "steps 5")
    pid = server_pid()
    before = open_files(pid)
    stuck = 0
    for _ in range(200):
        reader = server()
        # some of these start past the newest event, so the reader is blocked with nothing to read when the line drops
        dropped = threading.Event()
        waiting = threading.Thread(target=drain, args=(reader, sid, random.choice([0, 3, 10, 10**9]), dropped), daemon=True)
        waiting.start()
        time.sleep(random.random() * 0.02)
        dropped.set()
        reader.hang_up()
        waiting.join(timeout=3)
        stuck += waiting.is_alive()
    time.sleep(1.5)
    leaked = open_files(pid) - before
    check("reconnect: hanging up frees every one of 200 readers, blocked or not", stuck == 0, stuck)
    check(f"reconnect: the server keeps no open files for dropped streams ({leaked:+d})", leaked < 20, leaked)
    check("reconnect: the chat still answers", ask(sid, "echo fine") == "fine")


def round_fuzz() -> None:
    sid = start(workspace("fuzz"))
    junk = [None, 0, -1, 2**70, 1.5, True, "", "x", "\x00", "💥" * 1000, "../" * 40, [], [1, [2]], {}, {"a": {"b": None}}, "A" * 300_000]
    paths = ["/session", f"/session/{sid}/message", f"/session/{sid}/upload", f"/session/{sid}/permission", f"/session/{sid}/abort",
             "/session/nope/message", "/session/../../etc/passwd", f"/session/{sid}/events?after=abc", "/search?q=&cwd=", "/nothing"]
    worst, tried = [], 0
    for _ in range(400):
        path, method = random.choice(paths), random.choice(["POST", "POST", "GET", "DELETE", "PUT"])
        if method == "DELETE" and path.endswith(sid):
            continue
        shape = random.choice(["junk", "object", "bytes"])
        body = random.choice(junk) if shape == "junk" else b"{not json" if shape == "bytes" else {
            key: random.choice(junk) for key in random.sample(["cwd", "mode", "text", "attachments", "id", "allow", "name", "media_type", "data", "resume", "model", "folder"], 4)}
        tried += 1
        code = status_of(method, path, body)
        if code >= SERVER_ERROR:
            worst.append((code, method, path, str(body)[:80]))
    check(f"fuzz: {tried} malformed requests never produce a server error", not worst, worst[:4])
    check("fuzz: a wrong token is refused everywhere", {status_of("POST", path, {}, token="wrong") for path in paths[:6]} == {401})
    check("fuzz: a 3 MB message goes through", ask(sid, "echo " + "z" * (3 * MEGABYTE))[:3] == "zzz")
    check("fuzz: the server is still healthy", server().call("GET", "/health")["ok"] is True and ask(sid, "echo ok") == "ok")


def round_compact() -> None:
    sid = start(workspace("compact"))
    got = turn(sid, "big 12")
    squeezed = [event for event in got["events"] if event["type"] == "compaction"]
    lines = (workspace("compact") / "big.txt").read_text().split()
    check(f"compact: the chat is condensed {len(squeezed)} times mid-task and the task still finishes in order", got["text"] == "big done" and len(squeezed) >= 5 and lines == [f"big-{n}" for n in range(1, 13)], (got["stop"], len(squeezed), lines))
    kept = Session.load(sid)
    check("compact: the live history stays short and the log still replays", len(kept.messages) < 12 and kept.compactions == len(squeezed), (len(kept.messages), kept.compactions))
    check("compact: the chat answers afterwards", ask(sid, "echo after") == "after")


def round_limits() -> None:
    capped = start(workspace("limits"), max_steps=6)
    got = turn(capped, "forever")
    written = (workspace("limits") / "forever.txt").read_text().split()
    check("limits: a step budget stops a model that never stops", got["stop"] == "error" and len(written) <= 6, (got["stop"], len(written)))
    free = start(workspace("limits-free"))
    began = time.monotonic()
    got = turn(free, "forever", wait=300)
    nudges = sum("same tool call" in json.dumps(message) for message in Session.load(free).messages)
    check(f"limits: with no budget the step cap ends it ({time.monotonic() - began:.0f}s, {nudges} nudges)", got["stop"] == "max_steps" and nudges >= 10, (got["stop"], nudges))
    check("limits: the chat answers after hitting the cap", ask(free, "echo ok") == "ok")


def round_sandbox() -> None:
    sid = start(workspace("sandbox"))
    here = workspace("sandbox")
    (here / ".git" / "hooks").mkdir(parents=True)
    token = json.loads((HOME / "server.json").read_text())["token"]
    port = server().port
    attacks = {
        "write outside the workspace": f"shell echo x > {OUTSIDE}/escape.txt",
        "append to a shell start-up file": "shell echo 'evil' >> ~/.zshrc",
        "plant a git hook": "shell echo x > .git/hooks/pre-commit",
        "plant a git hook through other letter case": "shell echo x > .GIT/hooks/pre-commit",
        "write through a link that points outside": f"shell ln -s {OUTSIDE}/target.txt link.txt; echo changed > link.txt",
        "move the protected folder away": "shell mv .git .git-moved && mkdir -p .git/hooks && echo x > .git/hooks/pre-commit",
        "hard link to an outside file": f"shell ln {OUTSIDE}/target.txt hard.txt && echo changed > hard.txt",
        "overwrite the server's own settings": f"shell echo '{{}}' > {HOME}/server.json",
        "write into the provider plug-in folder": f"shell echo 'x = 1' > {HOME}/providers/evil.py",
        "delete the outside folder": f"shell rm -rf {OUTSIDE}",
        "files tool: path that climbs out": 'files {"command": "create", "path": "../outside/escape2.txt", "file_text": "x"}',
        "files tool: absolute path outside": f'files {{"command": "create", "path": "{OUTSIDE}/escape3.txt", "file_text": "x"}}',
        "files tool: protected folder": 'files {"command": "create", "path": ".git/hooks/pre-push", "file_text": "x"}',
        "files tool: protected folder, other case": 'files {"command": "create", "path": ".GIT/hooks/pre-push", "file_text": "x"}',
    }
    for order in attacks.values():
        ask(sid, order)
    untouched = {
        "write outside the workspace": not (OUTSIDE / "escape.txt").exists(),
        "append to a shell start-up file": (FAKE_HOME / ".zshrc").read_text() == "# untouched\n",
        "plant a git hook": not list((here / ".git" / "hooks").glob("pre-*")),
        "write through a link or hard link": (OUTSIDE / "target.txt").read_text() == "untouched",
        "overwrite the server's own settings": json.loads((HOME / "server.json").read_text()).get("token") == token,
        "write into the provider plug-in folder": not (HOME / "providers" / "evil.py").exists(),
        "delete the outside folder": OUTSIDE.is_dir(),
        "move the protected folder away": not (here / ".git-moved").exists(),
        "files tool escapes": not list(OUTSIDE.glob("escape*")),
    }
    for name, held in untouched.items():
        check(f"sandbox: {name} is blocked", held)
    reads = {
        "an ssh key": ("shell cat ~/.ssh/id_test; ls ~/.ssh", "FAKE-PRIVATE-KEY"),
        "the server token": (f"shell cat {HOME}/server.json; cp {HOME}/server.json copy.json; cat copy.json", token),
        "the ssh key through the files tool": ('files {"command": "view", "path": "~/.ssh/id_test"}', "FAKE-PRIVATE-KEY"),
        "the ssh key through a link and the files tool": ('shell ln -sf ~/.ssh/id_test peek.txt', "FAKE-PRIVATE-KEY"),
    }
    for name, (order, secret) in reads.items():
        check(f"sandbox: {name} can't be read", secret not in ask(sid, order))
    check("sandbox: a planted link is not followed by the files tool", "FAKE-PRIVATE-KEY" not in ask(sid, 'files {"command": "view", "path": "peek.txt"}'))
    check("sandbox: the network is off (the server's own port can't be reached)", '"ok"' not in ask(sid, f"shell curl -s -m 3 http://127.0.0.1:{port}/health || python3 -c \"import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:{port}/health', timeout=3).read())\""))
    check("sandbox: the web tool refuses local addresses and files", all(sign not in ask(sid, f"web {url}") for url in (f"http://127.0.0.1:{port}/health", "file:///etc/passwd", "http://[::1]/") for sign in ('"ok"', "root:")))
    pid = server_pid()
    ask(sid, f"shell kill -9 {pid}; echo tried")
    time.sleep(0.5)
    alive = subprocess.run(["ps", "-p", str(pid)], capture_output=True).stdout.count(b"\n") > 1
    check("sandbox: a command can't kill the server", alive)
    if not alive:
        known.clear()
        sid = start(here)
    check("sandbox: honest work in the workspace still runs", ask(sid, "shell echo hi > ok.txt && cat ok.txt && python3 -c 'print(6*7)'").split() == ["hi", "42"])
    asking = start(workspace("ask"), mode="ask")
    got = turn(asking, "shell cat <<EOF\n$(touch pwned.txt)\nEOF", allow=False)
    asked = [event for event in got["events"] if event["type"] == "permission.request"]
    check("ask mode: a command hidden in a here-document asks first, and No stops it", len(asked) == 1 and not (workspace("ask") / "pwned.txt").exists(), len(asked))
    got = turn(asking, "shell ls -la", allow=False)
    check("ask mode: a plain read-only command does not ask", not [event for event in got["events"] if event["type"] == "permission.request"])
    planning = start(workspace("plan"), mode="plan")
    for order in ("shell echo x > made.txt", "shell sort -o made.txt /etc/hosts", "shell find . -fprint made.txt", 'files {"command": "create", "path": "made.txt", "file_text": "x"}'):
        turn(planning, order, allow=True)
    check("plan mode: four different ways to write a file all do nothing", not (workspace("plan") / "made.txt").exists())


def round_soak() -> None:
    sid = start(workspace("soak"))
    pid = server_pid()
    before, took, wrong = memory_mb(pid), [], 0
    for number in range(300):
        began = time.monotonic()
        wrong += ask(sid, f"echo n{number}") != f"n{number}"
        took.append(time.monotonic() - began)
    grown = memory_mb(pid) - before
    check(f"soak: 300 turns in one chat, all correct (median {statistics.median(took) * 1000:.0f} ms, slowest {max(took) * 1000:.0f} ms)", wrong == 0, wrong)
    check(f"soak: memory after 300 turns ({grown:+.0f} MB)", grown < 150, grown)
    check("soak: the 600-message log still replays", len(Session.load(sid).messages) == 600, len(Session.load(sid).messages))


ROUNDS = {name.removeprefix("round_"): work for name, work in list(globals().items()) if name.startswith("round_")}


def main(wanted: list[str]) -> int:
    began = time.monotonic()
    try:
        for name in wanted or ROUNDS:
            print(f"--- {name}", flush=True)
            try:
                ROUNDS[name]()
            except Exception as error:
                check(f"{name}: the round itself ran", False, repr(error))
    finally:
        cli.terminate_saved()
        for pid in running(LONG_NAP) + running(SHORT_NAP):
            try:
                os.kill(int(pid), signal.SIGKILL)
            except OSError:
                pass
    failed = [name for name, ok in results if not ok]
    print(f"\n{len(results) - len(failed)} of {len(results)} checks passed in {time.monotonic() - began:.0f}s")
    for name in failed:
        print("  FAILED:", name)
    if not failed:
        shutil.rmtree(ROOT, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
