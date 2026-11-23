from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

SESSION_START = "session_start"
TURN_START = "turn_start"
PRE_TOOL = "pre_tool"
POST_TOOL = "post_tool"
STOP = "stop"
EVENTS = (SESSION_START, TURN_START, PRE_TOOL, POST_TOOL, STOP)

Handler = Callable[..., Awaitable[Any]]


class Hooks:
    def __init__(self) -> None:
        self.handlers: dict[str, list[Handler]] = {name: [] for name in EVENTS}
        self.broken: list[str] = []

    def on(self, event: str, handler: Handler) -> None:
        self.handlers[event].append(handler)

    async def fire(self, event: str, *args: Any) -> None:
        for handler in self.handlers[event]:
            await handler(*args)

    async def first(self, event: str, *args: Any) -> str | None:
        for handler in self.handlers[event]:
            value = await handler(*args)
            if value:
                return value
        return None


def register_files(target: Any, folders: Iterable[Path], kind: str = "Plugin") -> tuple[list[str], list[str]]:
    """Import each .py file in the folders and hand `target` to its register(). Returns the names loaded and what failed."""
    loaded: list[str] = []
    broken: list[str] = []
    for folder in folders:
        for path in sorted(folder.glob("*.py")) if folder.is_dir() else []:
            spec = importlib.util.spec_from_file_location(f"hypersonic_{kind.lower()}_{path.stem}", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            # dataclasses and pickling look a class's module up by name, so it has to be listed while it loads
            sys.modules[spec.name] = module
            try:
                spec.loader.exec_module(module)
                module.register(target)
            except Exception as error:
                # one bad file must not stop every new session
                sys.modules.pop(spec.name, None)
                broken.append(f"{kind} {path.name} was skipped. {type(error).__name__}: {error}")
                continue
            loaded.append(path.stem)
    return loaded, broken


def load_plugins(hooks: Hooks, folders: Iterable[Path]) -> list[str]:
    loaded, broken = register_files(hooks, folders)
    hooks.broken += broken
    return loaded
