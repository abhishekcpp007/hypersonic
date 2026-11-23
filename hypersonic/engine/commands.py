from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from hypersonic.context.compact import compact
from hypersonic.engine.loop import Agent
from hypersonic.engine.session import MODEL_SLOT, PERMISSION_SLOT, harness_note, home
from hypersonic.rag import ingest
from hypersonic.tools.base import GLOBAL_SCOPE
from hypersonic.tools.files import undo_last

ARGUMENTS = "$ARGUMENTS"
COMMAND_FOLDER = "commands"
PROJECT_FOLDER = ".hypersonic"
MEMORY_FOLDER = "memory"
SLUG_WORDS = 6
NO_INDEX = "No search index in this session."
MODE_NOTE = "The permission mode is now {mode}. A tool call that was refused under another mode may be allowed now, so try it again when asked."
RAG_ON = "RAG is on for this chat. Every question now searches your indexed documents. Type /rag again to turn it off."
RAG_OFF = "RAG is off. Your documents are used only when a message starts with /rag."
MODEL_NOTE = (
    "From here on, replies come from the model {model}. Earlier replies in this conversation may have "
    "come from a different model. If asked which model you are, say {model}."
)


class CommandError(Exception):
    pass


@dataclass
class Outcome:
    notice: str = ""
    prompt: str = ""


def counted(counts: dict[str, int]) -> str:
    return ", ".join(f"{count} {name}" for name, count in counts.items())


async def compact_now(agent: Agent, rest: str) -> Outcome:
    if len(agent.session.messages) < 2:
        return Outcome("Nothing to compact yet.")
    done = await compact(agent, resume=False)
    return Outcome("Compacted. The earlier messages are saved in the session folder." if done else "Compaction produced no note.")


async def index(agent: Agent, rest: str) -> Outcome:
    if agent.ctx.store is None:
        raise CommandError(NO_INDEX)
    if not rest:
        raise CommandError("Usage: /index <folder or file>")
    folder = Path(rest).expanduser()
    folder = (folder if folder.is_absolute() else agent.ctx.workspace / folder).resolve()
    if not folder.exists():
        raise CommandError(f"{rest} doesn't exist.")
    scope = str(agent.ctx.workspace)
    counts = await asyncio.to_thread(ingest.index_folder, agent.ctx.store, scope, ingest.DOCS, folder)
    return Outcome(f"Indexed {folder}: {counted(counts)}.")


async def rag(agent: Agent, rest: str) -> Outcome:
    if agent.ctx.store is None:
        raise CommandError(NO_INDEX)
    if rest:
        agent.ctx.rag_once = True
        return Outcome(prompt=rest)
    agent.ctx.rag = not agent.ctx.rag
    return Outcome(RAG_ON if agent.ctx.rag else RAG_OFF)


async def undo(agent: Agent, rest: str) -> Outcome:
    return Outcome(undo_last(agent.ctx))


async def remember(agent: Agent, rest: str) -> Outcome:
    if agent.ctx.store is None:
        raise CommandError(NO_INDEX)
    if not rest:
        raise CommandError("Usage: /remember <fact>")
    folder = home() / MEMORY_FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    slug = "-".join(re.findall(r"[a-z0-9]+", rest.lower())[:SLUG_WORDS]) or "note"
    (folder / f"{int(time.time())}-{slug}.md").write_text(rest + "\n")
    ingest.index_folder(agent.ctx.store, GLOBAL_SCOPE, ingest.MEMORY, folder)
    return Outcome("Saved to memory.")


async def check(agent: Agent, rest: str) -> Outcome:
    if rest:
        agent.check = rest
    return Outcome(f"Check command: {agent.check}" if agent.check else "No check command set. Usage: /check <command>")


async def model(agent: Agent, rest: str) -> Outcome:
    if not rest:
        current = agent.model or getattr(agent.provider, "model", "?")
        return Outcome(f"Model: {current}. Type /model and a space to pick another from the list.")
    from hypersonic.providers.base import keep_model, make_provider

    kept = f"Still using {agent.model or getattr(agent.provider, 'model', 'the same model')}."
    try:
        agent.provider = await asyncio.to_thread(make_provider, rest, agent.effort or DEFAULT_EFFORT, home())
    except Exception as error:
        raise CommandError(f"NOT switched to {rest}. {error} {kept}")
    agent.model = rest
    agent.session.model_note = harness_note(MODEL_NOTE.format(model=rest))
    agent.session.queue_system(MODEL_SLOT, agent.session.model_note)
    keep_model(rest, home())
    return Outcome(f"Model switched to {rest}. New chats will use it too.")


async def set_mode(agent: Agent, rest: str) -> Outcome:
    from hypersonic.policy.rules import MODES, YOLO

    if rest not in MODES:
        return Outcome(f"Permission mode: {agent.policy.mode}. Usage: /mode <{' | '.join(MODES)}>")
    agent.policy.mode = rest
    agent.ctx.sandbox = rest != YOLO
    agent.session.queue_system(PERMISSION_SLOT, harness_note(MODE_NOTE.format(mode=rest)))
    return Outcome(f"Permission mode: {rest}.")


async def show_help(agent: Agent, rest: str) -> Outcome:
    lines = [f"/{c['name']}  {c['summary']}" for c in command_list(agent)]
    return Outcome("\n".join(lines))


Handler = Callable[[Agent, str], Awaitable[Outcome]]
DEFAULT_EFFORT = "medium"
BUILT_IN: dict[str, tuple[Handler, str]] = {
    "model": (model, "show or switch the model"),
    "mode": (set_mode, "change the permission mode"),
    "rag": (rag, "answer from your indexed documents: /rag <question>, or alone to keep it on"),
    "compact": (compact_now, "shrink the conversation to a short note now"),
    "index": (index, "add a folder or file to the document index"),
    "undo": (undo, "restore the last file the agent changed"),
    "remember": (remember, "save a fact to long-term memory"),
    "check": (check, "set the command that must pass before the agent finishes a change"),
    "help": (show_help, "list commands"),
}
CLIENT_COMMANDS = [
    {"name": "new", "summary": "start a fresh conversation"},
    {"name": "resume", "summary": "resume a past session"},
    {"name": "clear", "summary": "clear this conversation"},
    {"name": "exit", "summary": "quit"},
]


def command_list(agent: Agent) -> list[dict[str, str]]:
    builtin = [{"name": name, "summary": summary} for name, (_, summary) in BUILT_IN.items()]
    custom = [{"name": name, "summary": "custom command"} for name in sorted(custom_commands(agent))]
    return builtin + custom + CLIENT_COMMANDS


def custom_commands(agent: Agent) -> dict[str, Path]:
    folders = [home() / COMMAND_FOLDER]
    if agent.trusted:
        folders.append(agent.ctx.workspace / PROJECT_FOLDER / COMMAND_FOLDER)
    found: dict[str, Path] = {}
    for folder in folders:
        if folder.is_dir():
            found.update({path.stem: path for path in sorted(folder.glob("*.md"))})
    return found


def is_command(text: str) -> bool:
    """A slash command is one word after the slash. A message that opens with a path, like /tmp/notes.txt, is not one."""
    return text.startswith("/") and "/" not in text.split(maxsplit=1)[0][1:]


async def run_command(agent: Agent, line: str) -> Outcome:
    name, _, rest = line.removeprefix("/").partition(" ")
    rest = rest.strip()
    if name in BUILT_IN:
        return await BUILT_IN[name][0](agent, rest)
    custom = custom_commands(agent).get(name)
    if custom is None:
        raise CommandError(f"Unknown command /{name}. Try /help.")
    template = custom.read_text().strip()
    return Outcome(prompt=template.replace(ARGUMENTS, rest) if ARGUMENTS in template else f"{template}\n{rest}".strip())
