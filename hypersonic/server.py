from __future__ import annotations

import asyncio
import hmac
import json
import os
import secrets
import base64
import binascii
import re
import signal
import socket
import sys
import time
import tomllib
from dataclasses import asdict
from http import HTTPStatus
from pathlib import Path
from typing import Any, AsyncIterator, Callable

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from hypersonic import __version__, code_fingerprint
from hypersonic.context.prefix import instruction_file, system_prompt
from hypersonic.engine.commands import PROJECT_FOLDER, CommandError, command_list, is_command, run_command
from hypersonic.engine.events import SESSION_START, load_plugins, register_files
from hypersonic.engine.loop import STOP_ABORTED, STOP_COMMAND, Agent, run_turn, text_block
from hypersonic.engine.session import Session, home, list_sessions as past_sessions, resolve_id, transcript
from hypersonic.plugins import budgets, done_gate, loop_guard
from hypersonic.policy.rules import ASK_MODE, MODES, YOLO, Policy, load_rules
from hypersonic.providers.base import (
    PROVIDER_FOLDER, VENDORS, Provider, ToolCall, default_model, make_provider, picked_model, write_private,
)
from hypersonic.providers.catalog import list_models
from hypersonic.rag.ingest import COLLECTIONS, DOCS, MEMORY, SKILLS, index_folder
from hypersonic.rag.retrieve import retrieve
from hypersonic.rag.store import Store
from hypersonic.tools import TOOLS
from hypersonic.tools.base import GLOBAL_SCOPE, ToolContext

HOST = "127.0.0.1"
DEFAULT_EFFORT = "medium"
TOKEN_BYTES = 32
KEEPALIVE_SECONDS = 15
MODEL_LIST_SECONDS = 60
INFO_NAME = "server.json"
INDEX_NAME = "index.db"
STARTED_FINGERPRINT = code_fingerprint()
CONFIG_NAME = "config.toml"
SHUTDOWN_SECONDS = 2
MEGABYTE = 1024 * 1024
MAX_UPLOAD_BYTES = 15 * MEGABYTE
MAX_IMAGE_BYTES = 5 * MEGABYTE
IMAGE_TYPES = ("image/jpeg", "image/png", "image/gif", "image/webp")
NAME_CHARS = 80
UPLOAD_FOLDER = ".hypersonic/uploads"
UI_PATH = Path(__file__).resolve().parent / "ui.html"
SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
FILE_NOTE = "The user attached these files. Read them with your tools if they are relevant:\n{listing}"
FELL_BACK = "Your saved model {picked} couldn't start. {reason} Using {model} for now."
BAD_CONFIG = "{path} can't be read. {problem}"
HOME_WORKSPACE = (
    "Start Hypersonic inside a project folder, not {folder}. "
    "With your home folder as the workspace, the agent could read and change your keys and shell settings."
)

ProviderFactory = Callable[[str, str], Provider]


class NewSession(BaseModel):
    cwd: str
    mode: str = ASK_MODE
    network: bool = False
    trusted: bool = False
    max_steps: int | None = None
    max_seconds: int | None = None
    max_tokens: int | None = None
    model: str | None = None
    effort: str | None = None
    resume: str | None = None


class Prompt(BaseModel):
    text: str
    attachments: list[str] = []


class UploadBody(BaseModel):
    name: str
    media_type: str
    data: str


class IndexRequest(BaseModel):
    cwd: str
    folder: str
    collection: str = DOCS


class Decision(BaseModel):
    id: str
    allow: bool


class Live:
    def __init__(
        self,
        session: Session,
        provider: Provider,
        store: Store,
        mode: str,
        network: bool,
        trusted: bool,
        budget: budgets.Budget,
    ):
        self.events: list[dict[str, Any]] = []
        self.changed = asyncio.Condition()
        self.task: asyncio.Task | None = None
        self.pending: dict[str, asyncio.Future[bool]] = {}
        self.uploads: dict[str, dict[str, Any]] = {}
        ctx = ToolContext(session.workspace, session.dir, store, sandbox=mode != YOLO, network=network)
        policy = Policy(mode, load_rules(home() / CONFIG_NAME))
        self.agent = Agent(provider, session, ctx, TOOLS, system_prompt(), self.emit, self.ask, policy)
        self.agent.trusted = trusted
        self.agent.check = check_command(home() / CONFIG_NAME)
        self.index_standing(store, session.workspace, trusted)
        hooks = self.agent.hooks
        loop_guard.register(hooks)
        done_gate.register(hooks)
        budgets.register(hooks, budget)
        folders = [home() / "plugins"]
        if trusted:
            folders.append(session.workspace / PROJECT_FOLDER / "plugins")
        load_plugins(hooks, folders)

    @staticmethod
    def index_standing(store: Store, workspace: Path, trusted: bool) -> None:
        sources = [(GLOBAL_SCOPE, SKILLS, home() / "skills"), (GLOBAL_SCOPE, MEMORY, home() / "memory")]
        if trusted:
            sources.append((str(workspace), SKILLS, workspace / PROJECT_FOLDER / "skills"))
            rules = instruction_file(workspace)
            if rules:
                sources.append((str(workspace), DOCS, rules))
        for scope, collection, path in sources:
            if path.exists():
                index_folder(store, scope, collection, path)

    async def command(self, line: str, attached: list[str]) -> None:
        model, rag = self.agent.model, self.agent.ctx.rag
        try:
            outcome = await run_command(self.agent, line)
        except CommandError as error:
            outcome = None
            await self.emit({"type": "notice", "text": str(error), "error": True})
        if self.agent.model != model:
            await self.emit({"type": "model", "name": self.agent.model})
        if self.agent.ctx.rag != rag:
            await self.emit({"type": "rag", "on": self.agent.ctx.rag})
        if outcome and outcome.notice:
            await self.emit({"type": "notice", "text": outcome.notice})
        if outcome and outcome.prompt:
            await self.drive(outcome.prompt, self.take_attachments(attached))
            return
        await self.emit({"type": "turn.end", "stop": STOP_COMMAND, "text": ""})
        session = self.agent.session
        if session.queue:
            await self.drive("\n".join(session.drain()), session.drain_extra())

    @property
    def busy(self) -> bool:
        return self.task is not None and not self.task.done()

    def history_cursor(self) -> int:
        """Where a client should start reading events after displaying the saved transcript."""
        cursor = len(self.events)
        if not self.busy:
            return cursor
        waiting = {pid for pid, answer in self.pending.items() if not answer.done()}
        for event in reversed(self.events):
            kind = event["type"]
            # Usage marks a completed provider reply. Earlier text is already in the transcript.
            if kind in ("usage", "turn.start", "turn.end"):
                break
            if kind == "text":
                cursor = event["seq"] - 1
            elif kind == "permission.request" and event["id"] in waiting:
                cursor = event["seq"] - 1
                waiting.remove(event["id"])
        return cursor

    async def emit(self, event: dict[str, Any]) -> None:
        async with self.changed:
            event["seq"] = len(self.events) + 1
            self.events.append(event)
            self.changed.notify_all()

    async def ask(self, call: ToolCall) -> bool:
        answer: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self.pending[call.id] = answer
        await self.emit({"type": "permission.request", "id": call.id, "name": call.name, "input": call.input})
        try:
            return await answer
        finally:
            self.pending.pop(call.id, None)

    async def supervise(self, coro: Any) -> None:
        try:
            await coro
        except asyncio.CancelledError:
            # a stopped command or compaction has no turn of its own to end, and clients wait for this event
            if not self.events or self.events[-1]["type"] != "turn.end":
                await self.emit({"type": "turn.end", "stop": STOP_ABORTED, "text": ""})
        except Exception as error:
            await self.emit({"type": "error", "message": f"{type(error).__name__}: {error}"})
            await self.emit({"type": "turn.end", "stop": "error", "text": ""})

    async def drive(self, text: str, extra: list[dict[str, Any]] | None = None) -> None:
        session = self.agent.session
        while True:
            await run_turn(self.agent, text, extra or ())
            if not session.queue:
                return
            text = "\n".join(session.drain())
            extra = session.drain_extra()

    def take_attachments(self, ids: list[str]) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        files: list[str] = []
        for aid in ids:
            item = self.uploads.pop(aid, None)
            if item is None:
                continue
            if item["kind"] == "image":
                blocks.append(item["block"])
            else:
                files.append(f"- {item['path']} ({item['name']})")
        if files:
            blocks.append(text_block(FILE_NOTE.format(listing="\n".join(files))))
        return blocks

    async def follow(self, after: int) -> AsyncIterator[str]:
        position = min(after, len(self.events))
        while True:
            while position < len(self.events):
                event = self.events[position]
                position += 1
                yield f"id: {event['seq']}\ndata: {json.dumps(event)}\n\n"
            try:
                async with self.changed:
                    await asyncio.wait_for(
                        self.changed.wait_for(lambda: position < len(self.events)), KEEPALIVE_SECONDS
                    )
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"


def create_app(token: str, provider_factory: ProviderFactory | None = None) -> FastAPI:
    provider_factory = provider_factory or real_provider
    app = FastAPI(title="Hypersonic", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)
    live: dict[str, Live] = {}
    store = Store(home() / INDEX_NAME)
    cold_start = asyncio.Lock()
    offered: dict[str, Any] = {"rows": [], "until": 0.0}
    _, unloaded = register_files(VENDORS, [home() / PROVIDER_FOLDER], "Provider")

    def auth(authorization: str | None = Header(default=None)) -> None:
        if not hmac.compare_digest((authorization or "").encode(), f"Bearer {token}".encode()):
            raise HTTPException(HTTPStatus.UNAUTHORIZED, "Bad or missing token.")

    def find(sid: str) -> Live:
        if sid not in live:
            raise HTTPException(HTTPStatus.NOT_FOUND, f"No session {sid}.")
        return live[sid]

    guarded = [Depends(auth)]

    async def open_provider(model: str, effort: str) -> Provider:
        # a provider may have to start a helper program first: off the event loop, and one at a time
        # so two sessions opening together don't each start one
        async with cold_start:
            return await asyncio.to_thread(provider_factory, model, effort)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "version": __version__, "fingerprint": STARTED_FINGERPRINT}

    @app.get("/ui", response_class=HTMLResponse)
    async def ui() -> str:
        return UI_PATH.read_text()

    def opened(item: Live, notice: str = "") -> dict[str, Any]:
        session = item.agent.session
        return {"id": session.id, "cwd": str(session.workspace), "model": item.agent.model, "notice": notice}

    @app.post("/session", dependencies=guarded)
    async def new_session(body: NewSession) -> dict[str, Any]:
        try:
            workspace = Path(body.cwd).expanduser().resolve()
            is_folder = workspace.is_dir()
        except (OSError, ValueError):
            is_folder = False
        if not is_folder:
            raise HTTPException(HTTPStatus.BAD_REQUEST, f"{body.cwd!r} is not a folder.")
        if body.mode not in MODES:
            raise HTTPException(HTTPStatus.BAD_REQUEST, f"mode must be one of {list(MODES)}.")
        if Path.home().resolve().is_relative_to(workspace):
            raise HTTPException(HTTPStatus.BAD_REQUEST, HOME_WORKSPACE.format(folder=workspace))
        effort = body.effort or DEFAULT_EFFORT
        trouble = ""
        try:
            sid = resolve_id(body.resume) if body.resume else None
            if sid in live:
                # the session is already open here, so hand it back instead of opening it twice
                return opened(live[sid])
            configured = configured_model(home() / CONFIG_NAME)
            picked = "" if body.model or configured else picked_model(home())
            session = Session.load(sid) if sid else None
            model = body.model or default_model(configured or picked)
            try:
                provider = await open_provider(model, effort)
            except ValueError as error:
                if model != picked:
                    raise
                model = default_model(configured)
                provider = await open_provider(model, effort)
                trouble = FELL_BACK.format(picked=picked, reason=error, model=model)
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(HTTPStatus.BAD_REQUEST, str(error))
        if sid in live:
            # a second resume of the same session got here first while the model was starting
            return opened(live[sid])
        session = session or Session.create(workspace)
        budget = budgets.Budget(body.max_steps, body.max_seconds, body.max_tokens)
        # trust was decided for the folder the client is in, which a resumed session may not live in
        trusted = body.trusted and session.workspace == workspace
        try:
            item = Live(session, provider, store, body.mode, body.network, trusted, budget)
        except ValueError as error:
            raise HTTPException(HTTPStatus.BAD_REQUEST, str(error))
        item.agent.model, item.agent.effort = model, effort
        live[session.id] = item
        await item.agent.hooks.fire(SESSION_START, item.agent)
        for problem in [*unloaded, *item.agent.hooks.broken, *([trouble] if trouble else [])]:
            await item.emit({"type": "notice", "text": problem, "error": True})
        return opened(item, trouble)

    @app.delete("/session/{sid}", dependencies=guarded)
    async def close_session(sid: str) -> dict[str, Any]:
        item = find(sid)
        if item.busy and item.task:
            item.task.cancel()
        del live[sid]
        return {"ok": True}

    @app.get("/models", dependencies=guarded)
    async def models() -> list[dict[str, str]]:
        if time.monotonic() >= offered["until"]:
            offered["rows"] = await asyncio.to_thread(list_models, home())
            offered["until"] = time.monotonic() + MODEL_LIST_SECONDS
        return offered["rows"]

    @app.get("/sessions", dependencies=guarded)
    async def recent_sessions() -> list[dict[str, Any]]:
        return await asyncio.to_thread(past_sessions)

    @app.get("/session", dependencies=guarded)
    async def list_sessions() -> list[dict[str, Any]]:
        return [
            {
                "id": sid,
                "cwd": str(item.agent.session.workspace),
                "messages": len(item.agent.session.messages),
                "busy": item.busy,
            }
            for sid, item in live.items()
        ]

    @app.post("/session/{sid}/message", dependencies=guarded)
    async def message(sid: str, body: Prompt) -> dict[str, Any]:
        item = find(sid)
        seq = len(item.events)
        if not body.text.strip():
            raise HTTPException(HTTPStatus.BAD_REQUEST, "The message is empty.")
        command = is_command(body.text)
        if item.busy and command:
            raise HTTPException(HTTPStatus.CONFLICT, "Busy. Send the command after this turn ends.")
        if item.busy:
            item.agent.session.queue.append(body.text)
            item.agent.session.queue_extra.extend(item.take_attachments(body.attachments))
            return {"queued": True, "seq": seq}
        if command:
            work = item.command(body.text, body.attachments)
        else:
            work = item.drive(body.text, item.take_attachments(body.attachments))
        item.task = asyncio.create_task(item.supervise(work))
        return {"queued": False, "seq": seq}

    @app.post("/session/{sid}/upload", dependencies=guarded)
    async def upload(sid: str, body: UploadBody) -> dict[str, Any]:
        item = find(sid)
        try:
            raw = base64.b64decode(body.data, validate=True)
        except (binascii.Error, ValueError):
            raise HTTPException(HTTPStatus.BAD_REQUEST, "attachment data must be base64.")
        picture = body.media_type in IMAGE_TYPES
        limit = MAX_IMAGE_BYTES if picture else MAX_UPLOAD_BYTES
        if not raw or len(raw) > limit:
            raise HTTPException(HTTPStatus.BAD_REQUEST, f"attachment is empty or over {limit // MEGABYTE} MB.")
        aid = secrets.token_hex(8)
        name = (SAFE_NAME.sub("_", Path(body.name).name).strip("._") or "file")[-NAME_CHARS:]
        if picture:
            block = {"type": "image", "source": {"type": "base64", "media_type": body.media_type, "data": body.data}}
            item.uploads[aid] = {"kind": "image", "name": name, "block": block}
            return {"id": aid, "kind": "image", "name": name}
        folder = item.agent.ctx.workspace / UPLOAD_FOLDER
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{aid[:6]}-{name}"
        path.write_bytes(raw)
        item.uploads[aid] = {"kind": "file", "name": name, "path": str(path)}
        return {"id": aid, "kind": "file", "name": name}

    @app.get("/session/{sid}/history", dependencies=guarded)
    async def history(sid: str) -> dict[str, Any]:
        item = find(sid)
        session = item.agent.session
        return {
            "items": transcript(session.messages),
            "seq": item.history_cursor(),
            "busy": item.busy,
            "cwd": str(session.workspace),
            "model": item.agent.model,
            "rag": item.agent.ctx.rag,
        }

    @app.get("/session/{sid}/commands", dependencies=guarded)
    async def commands(sid: str) -> list[dict[str, str]]:
        return command_list(find(sid).agent)

    @app.get("/session/{sid}/events", dependencies=guarded)
    async def events(sid: str, after: int = Query(0, ge=0)) -> StreamingResponse:
        return StreamingResponse(find(sid).follow(after), media_type="text/event-stream")

    @app.post("/session/{sid}/permission", dependencies=guarded)
    async def permission(sid: str, body: Decision) -> dict[str, Any]:
        answer = find(sid).pending.get(body.id)
        if answer is None or answer.done():
            raise HTTPException(HTTPStatus.NOT_FOUND, "No such pending request.")
        answer.set_result(body.allow)
        return {"ok": True}

    @app.post("/session/{sid}/abort", dependencies=guarded)
    async def abort(sid: str) -> dict[str, Any]:
        item = find(sid)
        if item.busy and item.task:
            item.task.cancel()
        return {"ok": True}

    @app.post("/index", dependencies=guarded)
    async def index(body: IndexRequest) -> dict[str, int]:
        workspace = Path(body.cwd).expanduser().resolve()
        folder = Path(body.folder).expanduser()
        folder = (folder if folder.is_absolute() else workspace / folder).resolve()
        if not folder.exists():
            raise HTTPException(HTTPStatus.BAD_REQUEST, f"{body.folder} doesn't exist.")
        if body.collection not in COLLECTIONS:
            raise HTTPException(HTTPStatus.BAD_REQUEST, f"collection must be one of {list(COLLECTIONS)}.")
        return await asyncio.to_thread(index_folder, store, str(workspace), body.collection, folder)

    @app.get("/search", dependencies=guarded)
    async def search(q: str, cwd: str, scope: str | None = None) -> dict[str, Any]:
        workspace = str(Path(cwd).expanduser().resolve())
        found = retrieve(store, q, [GLOBAL_SCOPE, workspace], [scope] if scope else None)
        return {"confidence": found.confidence, "hits": [asdict(hit) for hit in found.hits]}

    return app


def read_config(config: Path) -> dict[str, Any]:
    if not config.is_file():
        return {}
    try:
        return tomllib.loads(config.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(BAD_CONFIG.format(path=config, problem=error))


def configured_model(config: Path) -> str:
    model = read_config(config).get("model", "")
    return model if isinstance(model, str) else ""


def real_provider(model: str, effort: str) -> Provider:
    return make_provider(model, effort, home())


def check_command(config: Path) -> str:
    section = read_config(config).get("code", {})
    check = section.get("check", "") if isinstance(section, dict) else ""
    return check if isinstance(check, str) else ""


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind((HOST, 0))
        return probe.getsockname()[1]


def serve(port: int | None = None) -> None:
    port = port or free_port()
    token = secrets.token_urlsafe(TOKEN_BYTES)
    info = home() / INFO_NAME
    mine = os.getpid()
    write_private(info, json.dumps({"port": port, "token": token, "pid": mine}))
    # uvicorn raises the signal again after its own shutdown, which would skip the cleanup below
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        uvicorn.run(create_app(token), host=HOST, port=port, log_level="warning", timeout_graceful_shutdown=SHUTDOWN_SECONDS)
    finally:
        # only remove the info file if it still points at this process (a newer server may own it)
        try:
            if json.loads(info.read_text()).get("pid") == mine:
                info.unlink(missing_ok=True)
        except (OSError, ValueError):
            pass
