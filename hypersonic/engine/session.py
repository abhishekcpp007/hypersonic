from __future__ import annotations

import json
import os
import secrets
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from hypersonic.providers.base import Message, Usage

ID_BYTES = 6
LOG_NAME = "log.jsonl"
MODEL_SLOT = "model"
PERMISSION_SLOT = "permission"
PREVIEW_CHARS = 300
USAGE_FIELDS = ("input", "cache_read", "cache_write", "output")
HANDOFF_TAG = "<handoff>"
CONDENSED = "Earlier messages were condensed into a short note."


def home() -> Path:
    return Path(os.environ.get("HYPERSONIC_HOME") or Path.home() / ".hypersonic")


ENV_TAG = "<env>"
HARNESS_TAG = "<harness>"
DAMAGED = "The log of session {sid} is damaged, so it can't be continued."
HARNESS_BLOCKS = (ENV_TAG, "<instructions", "<context>", HARNESS_TAG)


def harness_note(text: str) -> str:
    """Text the harness adds to the conversation by itself, marked so nobody takes it for the user's words."""
    return f"{HARNESS_TAG}\n{text}\n</harness>"


def list_sessions(limit: int = 20) -> list[dict[str, Any]]:
    base = home() / "sessions"
    if not base.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for folder in base.iterdir():
        log = folder / LOG_NAME
        if not log.is_file():
            continue
        try:
            records = [json.loads(line) for line in log.read_text().split("\n") if line]
            if not records:
                continue
            messages = [r for r in records if r.get("type") == "message"]
            if not messages:
                continue
            preview = ""
            for record in messages:
                message = record.get("message", {})
                if message.get("role") != "user":
                    continue
                content = message.get("content", "")
                blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
                texts = [b.get("text", "") for b in blocks if b.get("type") == "text" and not b.get("text", "").startswith(HARNESS_BLOCKS)]
                if texts:
                    preview = texts[0][:80]
                    break
            out.append({
                "id": folder.name,
                "cwd": records[0].get("workspace", ""),
                "modified": log.stat().st_mtime,
                "messages": len(messages),
                "preview": preview,
            })
        except (OSError, ValueError, KeyError, TypeError):
            continue
    out.sort(key=lambda item: item["modified"], reverse=True)
    return out[:limit]


def resolve_id(sid: str) -> str:
    """The full session id for an id, or for the start of one as session lists show it."""
    base = home() / "sessions"
    # compared with the real folder names, so odd spellings and ../ tricks can never open a second copy or another file
    names = [folder.name for folder in base.iterdir() if (folder / LOG_NAME).is_file()] if base.is_dir() else []
    if sid in names:
        return sid
    matches = [name for name in names if sid and name.startswith(sid)]
    if len(matches) > 1:
        raise FileNotFoundError(f"{len(matches)} sessions start with {sid}. Type more of the id.")
    if not matches:
        raise FileNotFoundError(f"No session {sid}.")
    return matches[0]


def transcript(messages: list[Message]) -> list[dict[str, Any]]:
    """The conversation as a client shows it: what the user typed, what the agent said, each tool call and how it ended."""
    shown: list[dict[str, Any]] = []
    calls: dict[str, dict[str, Any]] = {}
    for message in messages:
        content = message["content"]
        blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
        for block in blocks if message["role"] != "system" else []:
            kind, text = block.get("type"), block.get("text", "")
            if kind == "tool_use":
                calls[block["id"]] = {"role": "tool", "id": block["id"], "name": block["name"], "input": block["input"], "ok": None, "preview": ""}
                shown.append(calls[block["id"]])
            elif kind == "tool_result" and block["tool_use_id"] in calls:
                calls[block["tool_use_id"]].update(ok="is_error" not in block, preview=str(block["content"])[:PREVIEW_CHARS])
            elif kind == "text" and text.startswith(HANDOFF_TAG):
                shown.append({"role": "notice", "text": CONDENSED})
            elif kind == "text" and text.strip() and not text.startswith(HARNESS_BLOCKS):
                shown.append({"role": message["role"], "text": text})
    return shown


class Session:
    def __init__(self, sid: str, workspace: Path):
        self.id = sid
        self.workspace = workspace
        self.dir = home() / "sessions" / sid
        self.messages: list[Message] = []
        self.queue: list[str] = []
        self.queue_extra: list[dict[str, Any]] = []
        self.last = Usage()
        self.lines = 0
        self.compactions = 0
        self.pending_system: dict[str, str] = {}
        self.model_note = ""
        self.compact_retry_at = 0
        self.steps = 0
        self.tokens = 0
        self.dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(cls, workspace: Path) -> Session:
        session = cls(secrets.token_hex(ID_BYTES), workspace)
        session.write("meta", workspace=str(workspace))
        return session

    @classmethod
    def load(cls, sid: str) -> Session:
        sid = resolve_id(sid)
        path = home() / "sessions" / sid / LOG_NAME
        try:
            records = [json.loads(line) for line in path.read_text().split("\n") if line]
            session = cls(sid, Path(records[0]["workspace"]))
        except (ValueError, LookupError, TypeError):
            raise ValueError(DAMAGED.format(sid=sid))
        session.lines = len(records)
        for record in records:
            if record["type"] == "message":
                session.messages.append(record["message"])
            elif record["type"] == "compaction":
                session.messages = []
                session.compactions += 1
                session.tokens += sum(record.get(name, 0) for name in USAGE_FIELDS)
            elif record["type"] == "usage":
                session.steps += 1
                session.tokens += sum(record.get(name, 0) for name in USAGE_FIELDS)
        return session

    def write(self, kind: str, **data: Any) -> None:
        record = {"id": self.lines + 1, "parent_id": self.lines, "ts": time.time(), "type": kind, **data}
        with (self.dir / LOG_NAME).open("a") as out:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.lines += 1

    def append(self, message: Message) -> None:
        # the log first: if the write fails, memory and disk must still agree
        self.write("message", message=message)
        self.messages.append(message)

    def record(self, usage: Usage) -> None:
        self.last = usage
        self.steps += 1
        self.tokens += usage.prompt + usage.output
        self.write("usage", **asdict(usage))

    def compact(self, opening: Message, usage: Usage) -> None:
        self.write("compaction", dropped=len(self.messages), **asdict(usage))
        self.tokens += usage.prompt + usage.output
        self.messages = []
        self.compactions += 1
        self.last = Usage()
        self.append(opening)

    def queue_system(self, slot: str, block: str) -> None:
        """Queue a system note for the next turn. A newer note in the same slot replaces the older one."""
        self.pending_system[slot] = block

    def take_system(self) -> list[str]:
        pending, self.pending_system = self.pending_system, {}
        return list(pending.values())

    def drain(self) -> list[str]:
        queued, self.queue = self.queue, []
        return queued

    def drain_extra(self) -> list[dict[str, Any]]:
        extra, self.queue_extra = self.queue_extra, []
        return extra
