from __future__ import annotations

import platform
from datetime import date
from pathlib import Path
from typing import Any

from hypersonic.context.squeeze import CHARS_PER_TOKEN
from hypersonic.providers.base import text_block

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"
HEAD_PREFIX = "ref: refs/heads/"
GITDIR_PREFIX = "gitdir: "
INSTRUCTION_FILES = ("AGENTS.md", "CLAUDE.md")
INSTRUCTION_TOKENS = 1000


def system_prompt() -> str:
    return (PROMPTS / "base.md").read_text().strip()


def git_branch(workspace: Path) -> str:
    folder = workspace / ".git"
    if folder.is_file():
        # a worktree or submodule keeps a one-line pointer here in place of the folder
        folder = workspace / folder.read_text().partition(GITDIR_PREFIX)[2].strip()
    head = folder / "HEAD"
    if not head.is_file():
        return "no repo"
    return head.read_text().strip().removeprefix(HEAD_PREFIX)


def env_block(workspace: Path) -> str:
    facts = {
        "cwd": workspace,
        "os": f"{platform.system()} {platform.release()}",
        "date": date.today().isoformat(),
        "git": git_branch(workspace),
    }
    body = "\n".join(f"{name}: {value}" for name, value in facts.items())
    return f"<env>\n{body}\n</env>"


def instruction_file(workspace: Path) -> Path | None:
    return next((workspace / name for name in INSTRUCTION_FILES if (workspace / name).is_file()), None)


def instructions(workspace: Path) -> str:
    path = instruction_file(workspace)
    if path is None:
        return ""
    text = path.read_text(errors="replace").strip()
    limit = INSTRUCTION_TOKENS * CHARS_PER_TOKEN
    if len(text) > limit:
        text = text[:limit].rsplit("\n", 1)[0]
        text += f"\n[... cut. Read the rest of {path.name} at {path}.]"
    return f'<instructions source="{path.name}">\n{text}\n</instructions>'


def opening(workspace: Path, trusted: bool) -> list[dict[str, Any]]:
    blocks = [text_block(env_block(workspace))]
    found = instructions(workspace) if trusted else ""
    if found:
        blocks.append(text_block(found))
    return blocks
