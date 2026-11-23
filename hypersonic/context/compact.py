from __future__ import annotations

import json
from typing import TYPE_CHECKING

from hypersonic.context.prefix import PROMPTS, opening
from hypersonic.engine.session import MODEL_SLOT
from hypersonic.providers.base import STOP_END, Message, text_block
from hypersonic.rag.ingest import CHAT, index_folder

if TYPE_CHECKING:
    from hypersonic.engine.loop import Agent
    from hypersonic.engine.session import Session

COMPACT_AT = 30_000
WINDOW_WANTED = 40_000
RETRY_AFTER_TOKENS = 5_000
INPUT_CHARS = 2000
INDEXED_FOLDERS = ("archive", "out")
CONTINUE = "Continue the task from the next steps."
NOTE = (
    "<handoff>\n{note}\n</handoff>\n"
    "Earlier messages were removed to save context. The full text is in {archive}. "
    "Read it with rg or files when you need a detail."
)


def due(session: Session) -> bool:
    return session.last.prompt + session.last.output >= COMPACT_AT


def render(messages: list[Message]) -> str:
    parts: list[str] = []
    for message in messages:
        content = message["content"]
        for block in content if isinstance(content, list) else [text_block(content)]:
            kind = block.get("type")
            if kind == "text":
                parts.append(f"## {message['role']}\n{block['text']}")
            elif kind == "tool_use":
                call = json.dumps(block["input"], ensure_ascii=False)[:INPUT_CHARS]
                parts.append(f"## tool call {block['name']}\n{call}")
            elif kind == "tool_result":
                parts.append(f"## tool result\n{block['content']}")
    return "\n\n".join(parts) + "\n"


async def discard(delta: str) -> None:
    return None


async def compact(agent: Agent, resume: bool) -> bool:
    session, ctx = agent.session, agent.ctx
    ask = {"role": "user", "content": [text_block((PROMPTS / "compact.md").read_text().strip())]}
    reply = await agent.provider.stream(agent.system, agent.specs, session.messages + [ask], discard)
    note = reply.text.strip()
    if not note or reply.tool_calls or reply.stop != STOP_END:
        return False
    archive = ctx.session_dir / "archive" / f"{session.compactions + 1:03d}.md"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_text(render(session.messages))
    if ctx.store:
        for name in INDEXED_FOLDERS:
            if (ctx.session_dir / name).is_dir():
                index_folder(ctx.store, ctx.chat_scope, CHAT, ctx.session_dir / name)
    before = session.last.prompt
    handoff = NOTE.format(note=note, archive=archive) + (f"\n{CONTINUE}" if resume else "")
    standing = [text_block(session.model_note)] if session.model_note and resume else []
    first = {"role": "user", "content": opening(ctx.workspace, agent.trusted) + [text_block(handoff)] + standing}
    session.compact(first, reply.usage)
    ctx.attached.clear()
    if session.model_note and not resume:
        session.queue_system(MODEL_SLOT, session.model_note)
    await agent.emit({"type": "compaction", "before": before})
    return True
