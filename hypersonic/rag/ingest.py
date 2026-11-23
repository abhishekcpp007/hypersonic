from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterator

from hypersonic.rag.chunk import chunk
from hypersonic.rag.store import Store

DOCS = "docs"
CHAT = "chat"
SKILLS = "skills"
MEMORY = "memory"
CODE = "code"
COLLECTIONS = (DOCS, CHAT, SKILLS, MEMORY, CODE)
MARKDOWN = {".md", ".markdown", ".mdx", ".rst", ".txt"}
SKIP_DIRS = {
    ".git", ".hg", ".venv", "venv", "node_modules", "__pycache__", ".hypersonic", "dist", "build",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "target",
}
MAX_FILE_BYTES = 1_000_000
SNIFF_BYTES = 8192
GIT_LIST = ["git", "-c", "core.fsmonitor=false", "ls-files", "-co", "--exclude-standard", "-z"]


def index_text(store: Store, scope: str, collection: str, path: str, text: str, stamp: str, headings: bool) -> int:
    chunks = chunk(text, headings=headings)
    store.replace(scope, collection, path, stamp, chunks)
    return len(chunks)


def readable(path: Path) -> bool:
    try:
        if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
            return False
        with path.open("rb") as source:
            return b"\0" not in source.read(SNIFF_BYTES)
    except OSError:
        return False


def walk(root: Path) -> Iterator[Path]:
    if root.is_file():
        yield root
        return
    try:
        listed = subprocess.run(GIT_LIST, cwd=root, capture_output=True)
    except OSError:
        listed = None
    if listed and listed.returncode == 0:
        for name in listed.stdout.decode(errors="replace").split("\0"):
            if not name or any(part in SKIP_DIRS for part in Path(name).parts[:-1]):
                continue
            path = root / name
            if path.is_dir() and not path.is_symlink():
                yield from walk(path)
            else:
                yield path
        return
    for folder, dirs, names in os.walk(root):
        dirs[:] = sorted(name for name in dirs if name not in SKIP_DIRS and not name.startswith("."))
        for name in sorted(names):
            yield Path(folder) / name


def index_folder(store: Store, scope: str, collection: str, root: Path) -> dict[str, int]:
    root = root.resolve()
    counts = {"indexed": 0, "unchanged": 0, "removed": 0}
    present: set[str] = set()
    for path in walk(root):
        # a link may point anywhere: only files that really live under the folder are read
        if not readable(path) or not path.resolve().is_relative_to(root):
            continue
        key = str(path)
        try:
            status = path.stat()
        except OSError:
            continue  # file vanished between the readable() check and here
        present.add(key)
        stamp = f"{status.st_mtime_ns}:{status.st_size}"
        if store.stamp(scope, collection, key) == stamp:
            counts["unchanged"] += 1
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            present.discard(key)
            continue
        index_text(store, scope, collection, key, text, stamp, headings=path.suffix.lower() in MARKDOWN)
        counts["indexed"] += 1
    prefix = str(root)
    for key in store.paths(scope, collection):
        if key not in present and (key == prefix or key.startswith(prefix + os.sep)):
            store.remove(scope, collection, key)
            counts["removed"] += 1
    return counts
