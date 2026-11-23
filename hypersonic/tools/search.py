from __future__ import annotations

import asyncio
from itertools import islice
from typing import Any

from hypersonic.providers.base import ToolSpec
from hypersonic.rag.ingest import CODE, COLLECTIONS, index_folder, walk
from hypersonic.rag.retrieve import pack, retrieve
from hypersonic.tools.base import Tool, ToolContext, ToolError

SEARCH_TOKENS = 1500
NO_MATCHES = "No matches."
RAG_OFF = "The index is searched only when the user asks with /rag. Use rg and files instead, or tell them to start the question with /rag."
CODE_FILE_LIMIT = 2000
TOO_BIG = "This workspace has more than {limit} files, so its code isn't indexed. Find code with rg through shell instead."

SPEC = ToolSpec(
    "search",
    "Search the local index for passages, with sources. It works only when the user asked "
    "with /rag. Covers their documents, earlier conversation that left context, skills and "
    "memory. Code is searched only with scope code.",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "scope": {"type": "string", "enum": list(COLLECTIONS)},
        },
        "required": ["query"],
    },
)


def refresh_code(ctx: ToolContext) -> None:
    """Bring the workspace's code index up to date. Only changed files are read again."""
    if sum(1 for _ in islice(walk(ctx.workspace), CODE_FILE_LIMIT + 1)) > CODE_FILE_LIMIT:
        raise ToolError(TOO_BIG.format(limit=CODE_FILE_LIMIT))
    index_folder(ctx.store, str(ctx.workspace), CODE, ctx.workspace)


async def run(ctx: ToolContext, args: dict[str, Any]) -> str:
    if ctx.store is None:
        raise ToolError("No index is available in this session.")
    if not ctx.rag_on:
        raise ToolError(RAG_OFF)
    scope = args.get("scope")
    if scope == CODE:
        await asyncio.to_thread(refresh_code, ctx)
    # code is only brought up to date on the line above, so a general search leaves it out instead of showing old code
    wanted = [scope] if scope else [name for name in COLLECTIONS if name != CODE]
    found = retrieve(ctx.store, args["query"], ctx.scopes, wanted)
    return pack(found.hits, SEARCH_TOKENS, ctx.attached, fresh_only=False) or NO_MATCHES


TOOL = Tool(SPEC, run, lambda args: True)
