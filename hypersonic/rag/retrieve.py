from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from hypersonic.context.squeeze import CHARS_PER_TOKEN
from hypersonic.rag.ingest import DOCS, MEMORY, SKILLS
from hypersonic.rag.store import Hit, Store
from hypersonic.rag.tokens import tokenize

TOP_K = 8
OVERLAP_WEIGHT = 0.5
SCORE_CAP = 12.0
GATE = 0.3
MIN_SCORED_CHUNKS = 20
MIN_QUERY_TOKENS = 3
ATTACH_HITS = 3
ATTACH_TOKENS = 600
RAG_COLLECTIONS = (DOCS, SKILLS, MEMORY)


@dataclass
class Found:
    hits: list[Hit]
    confidence: float


def confidence(query_tokens: Sequence[str], hits: Sequence[Hit], scored: bool) -> float:
    wanted = set(query_tokens)
    if not hits or not wanted:
        return 0.0
    overlap = max(len(wanted & set(tokenize(f"{hit.heading} {hit.body}"))) / len(wanted) for hit in hits)
    if not scored:
        return round(overlap, 4)
    strength = min(hits[0].score / SCORE_CAP, 1.0)
    return round(OVERLAP_WEIGHT * overlap + (1 - OVERLAP_WEIGHT) * strength, 4)


def retrieve(
    store: Store, query: str, scopes: Sequence[str], collections: Sequence[str] | None = None, limit: int = TOP_K
) -> Found:
    tokens = tokenize(query)
    hits = store.search(tokens, scopes, collections, limit)
    return Found(hits, confidence(tokens, hits, store.has_at_least(scopes, MIN_SCORED_CHUNKS)))


def render(hit: Hit) -> str:
    where = f"{hit.path}:{hit.first_line}-{hit.last_line}"
    title = f" ({hit.heading})" if hit.heading else ""
    return f"[{hit.collection}] {where}{title}\n{hit.body}"


def mark(hit: Hit) -> int:
    """Identifies a passage by what it says. Row numbers are reused when a file is indexed again."""
    return hash((hit.path, hit.body))


def pack(hits: Sequence[Hit], budget_tokens: int, seen: set[int], fresh_only: bool = True) -> str:
    limit = budget_tokens * CHARS_PER_TOKEN
    parts: list[str] = []
    packed: set[int] = set()
    used = 0
    for hit in hits:
        if mark(hit) in packed or (fresh_only and mark(hit) in seen):
            continue
        text = render(hit)
        if parts and used + len(text) > limit:
            break
        parts.append(text[:limit])
        used += len(text)
        packed.add(mark(hit))
    seen |= packed
    return "\n\n".join(parts)


def attach(store: Store, query: str, scopes: Sequence[str], seen: set[int], asked: bool = False) -> str:
    """Passages for a message. A short message is skipped unless the user asked for retrieval by name."""
    tokens = tokenize(query)
    if not tokens or (not asked and len(set(tokens)) < MIN_QUERY_TOKENS):
        return ""
    hits = store.search(tokens, scopes, RAG_COLLECTIONS, ATTACH_HITS)
    if confidence(tokens, hits, store.has_at_least(scopes, MIN_SCORED_CHUNKS)) < GATE:
        return ""
    body = pack(hits, ATTACH_TOKENS, seen)
    return f"<context>\n{body}\n</context>" if body else ""
