from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from hypersonic.providers.base import ToolSpec
from hypersonic.rag.store import Store

GLOBAL_SCOPE = "global"
RESULT_TOKENS = 2000
NAME_BYTES = 4
JSON_TYPES = {"string": str, "integer": int, "boolean": bool, "array": list, "object": dict}


class ToolError(Exception):
    pass


@dataclass
class ToolContext:
    workspace: Path
    session_dir: Path
    store: Store | None = None
    sandbox: bool = True
    network: bool = False
    local_web: bool = False
    cwd: Path | None = None
    seen: dict[str, tuple[int, int]] = field(default_factory=dict)
    attached: set[int] = field(default_factory=set)
    rag: bool = False
    rag_once: bool = False

    def __post_init__(self) -> None:
        self.workspace = self.workspace.resolve()
        self.session_dir = self.session_dir.resolve()
        self.cwd = self.cwd or self.workspace

    @property
    def rag_on(self) -> bool:
        """True when the user asked for their documents: for this chat (/rag) or this message (/rag <question>)."""
        return self.rag or self.rag_once

    @property
    def chat_scope(self) -> str:
        return f"session:{self.session_dir.name}"

    @property
    def scopes(self) -> list[str]:
        return [GLOBAL_SCOPE, str(self.workspace), self.chat_scope]

    def new_file(self, folder: str, suffix: str) -> Path:
        directory = self.session_dir / folder
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{secrets.token_hex(NAME_BYTES)}{suffix}"


@dataclass(frozen=True)
class Tool:
    spec: ToolSpec
    run: Callable[[ToolContext, dict[str, Any]], Awaitable[str]]
    read_only: Callable[[dict[str, Any]], bool]


def invalid(schema: dict[str, Any], args: Any) -> str | None:
    if not isinstance(args, dict):
        return "input must be an object"
    for name in schema.get("required", []):
        if name not in args:
            return f"missing {name}"
    for name, value in args.items():
        rule = schema["properties"].get(name)
        if rule is None:
            continue
        expected = JSON_TYPES[rule["type"]]
        if not isinstance(value, expected) or (expected is int and isinstance(value, bool)):
            return f"{name} must be {rule['type']}"
        if "enum" in rule and value not in rule["enum"]:
            return f"{name} must be one of {rule['enum']}"
    return None
