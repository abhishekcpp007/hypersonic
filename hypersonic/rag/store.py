from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from hypersonic.providers.base import OWNER_ONLY
from hypersonic.rag.chunk import Chunk
from hypersonic.rag.tokens import tokenize

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    scope TEXT NOT NULL, collection TEXT NOT NULL, path TEXT NOT NULL, stamp TEXT NOT NULL,
    PRIMARY KEY (scope, collection, path)
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
    tokens, title, body UNINDEXED, heading UNINDEXED, scope UNINDEXED, collection UNINDEXED,
    path UNINDEXED, first_line UNINDEXED, last_line UNINDEXED,
    tokenize = "unicode61 tokenchars '-_.'"
);
"""
TEXT_WEIGHT = 1.0
TITLE_WEIGHT = 1.0
COLUMNS = "body, heading, collection, path, first_line, last_line"
SOURCE = "scope = ? AND collection = ? AND path = ?"


@dataclass
class Hit:
    rowid: int
    score: float
    body: str
    heading: str
    collection: str
    path: str
    first_line: int
    last_line: int


def marks(items: Sequence[object]) -> str:
    return ", ".join("?" for _ in items)


class Store:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db, db:
            db.execute("PRAGMA journal_mode = WAL")
            db.executescript(SCHEMA)
        path.chmod(OWNER_ONLY)

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def stamp(self, scope: str, collection: str, path: str) -> str | None:
        with closing(self.connect()) as db:
            row = db.execute(f"SELECT stamp FROM sources WHERE {SOURCE}", (scope, collection, path)).fetchone()
        return row[0] if row else None

    def paths(self, scope: str, collection: str) -> list[str]:
        query = "SELECT path FROM sources WHERE scope = ? AND collection = ?"
        with closing(self.connect()) as db:
            return [row[0] for row in db.execute(query, (scope, collection))]

    def replace(self, scope: str, collection: str, path: str, stamp: str, chunks: Iterable[Chunk]) -> None:
        key = (scope, collection, path)
        name = Path(path).stem
        rows = [
            (
                " ".join(tokenize(item.text)),
                " ".join(tokenize(f"{name} {item.heading}")),
                item.text,
                item.heading,
                *key,
                item.first_line,
                item.last_line,
            )
            for item in chunks
        ]
        with closing(self.connect()) as db, db:
            db.execute(f"DELETE FROM chunks WHERE {SOURCE}", key)
            db.execute("INSERT OR REPLACE INTO sources VALUES (?, ?, ?, ?)", (*key, stamp))
            db.executemany(
                "INSERT INTO chunks (tokens, title, body, heading, scope, collection, path, first_line, last_line) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )

    def remove(self, scope: str, collection: str, path: str) -> None:
        key = (scope, collection, path)
        with closing(self.connect()) as db, db:
            db.execute(f"DELETE FROM chunks WHERE {SOURCE}", key)
            db.execute(f"DELETE FROM sources WHERE {SOURCE}", key)

    def has_at_least(self, scopes: Sequence[str], count: int) -> bool:
        query = f"SELECT count(*) FROM (SELECT 1 FROM chunks WHERE scope IN ({marks(scopes)}) LIMIT ?)"
        with closing(self.connect()) as db:
            return db.execute(query, (*scopes, count)).fetchone()[0] >= count

    def search(
        self, tokens: Sequence[str], scopes: Sequence[str], collections: Sequence[str] | None, limit: int
    ) -> list[Hit]:
        if not tokens or not scopes:
            return []
        match = " OR ".join(f'"{token}"' for token in dict.fromkeys(tokens))
        where = f"chunks MATCH ? AND scope IN ({marks(scopes)})"
        values: list[object] = [match, *scopes]
        if collections:
            where += f" AND collection IN ({marks(collections)})"
            values += collections
        rank = f"bm25(chunks, {TEXT_WEIGHT}, {TITLE_WEIGHT})"
        query = f"SELECT rowid, -{rank} AS score, {COLUMNS} FROM chunks WHERE {where} ORDER BY {rank} LIMIT ?"
        with closing(self.connect()) as db:
            return [Hit(*row) for row in db.execute(query, (*values, limit))]
