from __future__ import annotations

import argparse
import fcntl
import getpass
import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import quote, urlencode

from hypersonic import __version__, code_fingerprint
from hypersonic.context.prefix import INSTRUCTION_FILES
from hypersonic.engine.commands import PROJECT_FOLDER
from hypersonic.engine.loop import STOP_COMMAND
from hypersonic.engine.session import home
from hypersonic.policy.rules import ASK_MODE, AUTO, MODES
from hypersonic.providers.base import MODEL_ENV, OPENAI_STYLE, save_key
from hypersonic.server import HOST, INFO_NAME, serve

CALL_TIMEOUT = 30
SESSION_TIMEOUT = 90
LOCK_NAME = "server.lock"
SERVER_WORDS = ("hypersonic", "serve")
PROJECT_PARTS = ("commands", "skills", "plugins")
INDEX_TIMEOUT = 900
PREVIEW_LINES = 3
TRUSTED_NAME = "trusted.json"
GOOD_STOPS = ("end", STOP_COMMAND)
START_TIMEOUT = 15
POLL_SECONDS = 0.2
SUMMARY_CHARS = 100
DATA_PREFIX = "data: "
DIM, RESET = "\033[2m", "\033[0m"


class ClientError(Exception):
    pass


class Client:
    def __init__(self, port: int, token: str):
        self.port = port
        self.headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        self.stream: http.client.HTTPConnection | None = None

    def call(
        self, method: str, path: str, body: dict[str, Any] | None = None, timeout: int = CALL_TIMEOUT
    ) -> Any:
        connection = http.client.HTTPConnection(HOST, self.port, timeout=timeout)
        try:
            connection.request(method, path, None if body is None else json.dumps(body), self.headers)
            response = connection.getresponse()
            data = response.read().decode()
        finally:
            connection.close()
        if response.status >= http.client.BAD_REQUEST:
            raise ClientError(f"{response.status}: {data}")
        return json.loads(data) if data else None

    def hang_up(self) -> None:
        """End the open event stream, so a thread waiting on it stops instead of blocking forever."""
        stream, self.stream = self.stream, None
        if stream and stream.sock:
            try:
                stream.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def alive(self) -> bool:
        try:
            return self.call("GET", "/health").get("fingerprint") == code_fingerprint()
        except (OSError, http.client.HTTPException, ClientError, KeyError, ValueError, AttributeError):
            return False

    def events(self, sid: str, after: int, wanted: Callable[[], bool] = lambda: True) -> Iterator[dict[str, Any]]:
        """The session's events from `after` on, until the line drops.

        A reader in another thread passes `wanted`, and makes it false before it calls hang_up(). Then a hang-up
        that arrives while this is still connecting ends the stream too, where it would otherwise be missed.
        """
        connection = http.client.HTTPConnection(HOST, self.port)
        try:
            connection.connect()
            self.stream = connection
            if not wanted():
                return
            connection.request("GET", f"/session/{sid}/events?after={after}", headers=self.headers)
            response = connection.getresponse()
            if response.status != http.client.OK:
                raise ClientError(f"{response.status}: {response.read().decode()}")
            for raw in response:
                line = raw.decode().rstrip("\n")
                if line.startswith(DATA_PREFIX):
                    yield json.loads(line.removeprefix(DATA_PREFIX))
        finally:
            connection.close()


def saved_client() -> Client | None:
    try:
        info = json.loads((home() / INFO_NAME).read_text())
        return Client(info["port"], info["token"])
    except (OSError, ValueError, KeyError):
        return None


def command_of(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        pass
    try:
        return subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True).stdout
    except OSError:
        return ""


def saved_pid() -> int | None:
    """The saved server's process, only while it still is one. After a crash or reboot the number can belong to anything."""
    try:
        pid = int(json.loads((home() / INFO_NAME).read_text())["pid"])
    except (OSError, ValueError, LookupError, TypeError):
        return None
    return pid if all(word in command_of(pid) for word in SERVER_WORDS) else None


def terminate_saved() -> bool:
    pid = saved_pid()
    try:
        if pid:
            os.kill(pid, signal.SIGTERM)
    except OSError:
        pid = None
    (home() / INFO_NAME).unlink(missing_ok=True)
    return bool(pid)


def connect() -> Client:
    client = saved_client()
    if client and client.alive():
        return client
    home().mkdir(parents=True, exist_ok=True)
    with (home() / LOCK_NAME).open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        client = saved_client()
        if client and client.alive():
            # another client started the server while this one waited for the lock
            return client
        terminate_saved()
        with (home() / "server.log").open("ab") as log:
            subprocess.Popen(
                # -P keeps the current folder off the import path, so a project can't replace the server's code
                [sys.executable, "-P", "-m", "hypersonic.cli", "serve"],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        deadline = time.monotonic() + START_TIMEOUT
        while time.monotonic() < deadline:
            client = saved_client()
            if client and client.alive():
                return client
            time.sleep(POLL_SECONDS)
    raise ClientError(f"Server didn't start. See {home() / 'server.log'}.")


def note(text: str) -> None:
    print(f"{DIM}{text}{RESET}", file=sys.stderr, flush=True)


def confirm(question: str = "  allow?") -> bool:
    try:
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def trusted_folders() -> list[str]:
    try:
        return json.loads((home() / TRUSTED_NAME).read_text())
    except (OSError, ValueError):
        return []


def is_trusted(folder: str, args: argparse.Namespace, interactive: bool) -> bool:
    known = trusted_folders()
    if folder in known:
        return True
    parts = (*INSTRUCTION_FILES, *(f"{PROJECT_FOLDER}/{part}" for part in PROJECT_PARTS))
    carries_config = any((Path(folder) / name).exists() for name in parts)
    if not getattr(args, "trust", False):
        if not (interactive and carries_config):
            return False
        print(f"{folder} has its own agent instructions or commands.")
        if not confirm("Trust this folder and load them?"):
            return False
    home().mkdir(parents=True, exist_ok=True)
    (home() / TRUSTED_NAME).write_text(json.dumps(sorted({*known, folder}), indent=1))
    return True


def turn(client: Client, sid: str, text: str) -> dict[str, Any]:
    """Send one message from a script and wait for the turn to end. Nobody is there to approve a tool call, so each is denied."""
    after = client.call("POST", f"/session/{sid}/message", {"text": text})["seq"]
    usage: dict[str, Any] = {}
    notices: list[str] = []
    failed = False
    stream = client.events(sid, after)
    while True:
        try:
            event = next(stream)
        except StopIteration:
            raise ClientError("Lost the connection to the server.")
        except KeyboardInterrupt:
            # the interrupt ends the reader, so open it again to see the turn stop
            client.call("POST", f"/session/{sid}/abort")
            stream = client.events(sid, after)
            continue
        after = event["seq"]
        kind = event["type"]
        if kind == "permission.request":
            client.call("POST", f"/session/{sid}/permission", {"id": event["id"], "allow": False})
        elif kind == "usage":
            usage = event
        elif kind == "notice":
            notices.append(event["text"])
            failed = failed or bool(event.get("error"))
        elif kind == "error":
            note(f"error: {event['message']}")
        elif kind == "turn.end":
            return {**event, "usage": usage, "notices": notices, "failed": failed}


def folder_of(args: argparse.Namespace) -> str:
    return str(Path(getattr(args, "cwd", None) or os.getcwd()).expanduser().resolve())


def session_options(args: argparse.Namespace, interactive: bool) -> dict[str, Any]:
    folder = folder_of(args)
    return {
        "cwd": folder,
        "trusted": is_trusted(folder, args, interactive),
        "mode": getattr(args, "mode", None) or (AUTO if getattr(args, "yes", False) else ASK_MODE),
        "network": getattr(args, "network", False),
        "max_steps": getattr(args, "max_steps", None),
        "max_seconds": getattr(args, "max_seconds", None),
        "max_tokens": getattr(args, "max_tokens", None),
        "model": getattr(args, "model", None) or os.environ.get(MODEL_ENV),
    }


def start_session(client: Client, options: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    # starting a model can take a while the first time, longer than an ordinary call
    return client.call("POST", "/session", {**options, "resume": getattr(args, "resume", None)}, timeout=SESSION_TIMEOUT)


def run(args: argparse.Namespace) -> int:
    client = connect()
    opened = start_session(client, session_options(args, interactive=False), args)
    if opened.get("notice"):
        note(opened["notice"])
    try:
        end = turn(client, opened["id"], args.task)
    finally:
        try:
            client.call("DELETE", f"/session/{opened['id']}")
        except (ClientError, OSError):
            pass
    for line in (*end["notices"], end["text"]):
        if line:
            print(line)
    return 0 if end["stop"] in GOOD_STOPS and not (end["stop"] == STOP_COMMAND and end["failed"]) else 1


def index(args: argparse.Namespace) -> int:
    body = {"cwd": folder_of(args), "folder": args.folder}
    counts = connect().call("POST", "/index", body, timeout=INDEX_TIMEOUT)
    print(", ".join(f"{count} {name}" for name, count in counts.items()))
    return 0


def search(args: argparse.Namespace) -> int:
    query = urlencode({"q": args.query, "cwd": folder_of(args)})
    found = connect().call("GET", f"/search?{query}")
    print(f"confidence {found['confidence']}")
    for hit in found["hits"]:
        preview = " ".join(hit["body"].splitlines()[:PREVIEW_LINES])[:SUMMARY_CHARS]
        print(f"{hit['score']:6.2f}  [{hit['collection']}] {hit['path']}:{hit['first_line']}-{hit['last_line']}  {preview}")
    return 0


def desktop(args: argparse.Namespace) -> int:
    from hypersonic.app import MISSING, run_app

    connect()
    info = json.loads((home() / INFO_NAME).read_text())
    cwd = folder_of(args)
    url = f"http://{HOST}:{info['port']}/ui?token={quote(info['token'])}&cwd={quote(cwd)}"
    note(f"Opening the Hypersonic desktop app for {cwd}")
    note("The server keeps running in the background; `hypersonic stop` ends it.")
    try:
        run_app(url)
    except ImportError:
        print(MISSING, file=sys.stderr)
        return 1
    return 0


def tui(args: argparse.Namespace) -> int:
    from hypersonic.tui import run_tui
    client = connect()
    options = session_options(args, interactive=True)
    run_tui(client, start_session(client, options, args), options, resumed=bool(getattr(args, "resume", None)))
    return 0


def web(args: argparse.Namespace) -> int:
    connect()
    info = json.loads((home() / INFO_NAME).read_text())
    cwd = folder_of(args)
    url = f"http://{HOST}:{info['port']}/ui?token={quote(info['token'])}&cwd={quote(cwd)}"
    note(f"Opening the Hypersonic web UI for {cwd}")
    note("Workspace is that folder. The server keeps running in the background; `hypersonic stop` ends it.")
    webbrowser.open(url)
    print(url)
    return 0


def stop(_: argparse.Namespace) -> int:
    print("Server stopped." if terminate_saved() else "No server running.")
    return 0


def key(args: argparse.Namespace) -> int:
    secret = getpass.getpass(f"Paste your {args.provider} API key (it stays hidden): ").strip()
    if not secret:
        print("Nothing saved.")
        return 1
    save_key(args.provider, secret, home())
    print(f"Saved. Open the chat, type /model and a space, then pick a {args.provider} model.")
    return 0


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--mode", choices=MODES, default=argparse.SUPPRESS, help="permission mode (default ask)")
    common.add_argument("--yes", action="store_true", default=argparse.SUPPRESS, help="same as --mode auto")
    common.add_argument("--network", action="store_true", default=argparse.SUPPRESS, help="let shell commands use the network")
    common.add_argument("--trust", action="store_true", default=argparse.SUPPRESS, help="load this folder's instructions and commands")
    for limit in ("steps", "seconds", "tokens"):
        common.add_argument(f"--max-{limit}", type=int, default=argparse.SUPPRESS, help=f"session budget in {limit}")
    common.add_argument("--cwd", default=argparse.SUPPRESS, help="workspace folder")
    common.add_argument("--model", default=argparse.SUPPRESS, help="provider/model")
    common.add_argument("--resume", default=argparse.SUPPRESS, help="session id to continue")
    parser = argparse.ArgumentParser(prog="hypersonic", parents=[common])
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("chat", parents=[common], help="terminal chat UI (default; same as tui)")
    runner = commands.add_parser("run", parents=[common], help="one task, prints the final answer")
    runner.add_argument("task")
    server = commands.add_parser("serve", help="run the server in the foreground")
    server.add_argument("--port", type=int)
    commands.add_parser("stop", help="stop the background server")
    commands.add_parser("web", parents=[common], help="open the browser chat UI")
    commands.add_parser("tui", parents=[common], help="terminal chat UI (same as the default)")
    commands.add_parser("app", parents=[common], help="open the desktop app (native window)")
    indexer = commands.add_parser("index", parents=[common], help="add a folder or file to the search index")
    indexer.add_argument("folder")
    searcher = commands.add_parser("search", parents=[common], help="query the search index")
    searcher.add_argument("query")
    keys = commands.add_parser("key", help="save an API key for a model provider")
    keys.add_argument("provider", choices=[vendor for vendor, (_, _, needs) in OPENAI_STYLE.items() if needs])
    args = parser.parse_args(argv)
    try:
        if args.command == "serve":
            serve(args.port)
            return 0
        handler = {"run": run, "stop": stop, "index": index, "search": search, "web": web, "tui": tui, "chat": tui, "app": desktop, "key": key}.get(args.command, tui)
        return handler(args)
    except (ClientError, OSError) as error:
        print(f"hypersonic: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
