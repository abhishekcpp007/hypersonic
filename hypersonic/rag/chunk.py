from __future__ import annotations

import re
from dataclasses import dataclass, field

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
FENCE = re.compile(r"^\s*(```|~~~)")
WORDS = 180
OVERLAP = 40
OVERSIZE = 3


@dataclass
class Chunk:
    ordinal: int
    heading: str
    text: str
    first_line: int
    last_line: int


@dataclass
class Block:
    lines: list[str] = field(default_factory=list)
    first_line: int = 0
    atomic: bool = False

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()

    @property
    def words(self) -> int:
        return len(self.text.split())

    @property
    def last_line(self) -> int:
        return self.first_line + len(self.lines) - 1


def split_blocks(lines: list[str], headings: bool) -> list[tuple[str, Block]]:
    out: list[tuple[str, Block]] = []
    stack: list[tuple[int, str]] = []
    current = Block(first_line=1)
    fenced = False

    def flush() -> None:
        nonlocal current
        if current.text:
            out.append((" > ".join(title for _, title in stack), current))
        current = Block()

    for number, raw in enumerate(lines, start=1):
        line = raw.rstrip("\n")
        if FENCE.match(line):
            if not fenced:
                flush()
                current = Block(first_line=number, atomic=True)
            current.lines.append(line)
            if fenced:
                flush()
            fenced = not fenced
            continue
        if fenced:
            current.lines.append(line)
            continue
        match = HEADING.match(line) if headings else None
        if match:
            flush()
            level = len(match.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, match.group(2).strip()))
            continue
        if not line.strip():
            flush()
            continue
        if not current.lines:
            current.first_line = number
        current.lines.append(line)
    flush()
    return out


def slice_oversized(block: Block, words: int) -> list[Block]:
    if block.words <= OVERSIZE * words:
        return [block]
    pieces: list[Block] = []
    piece = Block(first_line=block.first_line, atomic=block.atomic)
    for offset, line in enumerate(block.lines):
        if piece.lines and piece.words + len(line.split()) > words:
            pieces.append(piece)
            piece = Block(first_line=block.first_line + offset, atomic=block.atomic)
        piece.lines.append(line)
    pieces.append(piece)
    return pieces


def overlap_tail(buffer: list[Block], overlap: int) -> list[Block]:
    if overlap <= 0 or not buffer or buffer[-1].atomic:
        return []
    last = buffer[-1]
    words = last.text.split()
    if len(words) <= overlap:
        return [last]
    return [Block(lines=[" ".join(words[-overlap:])], first_line=last.first_line)]


def chunk(text: str, headings: bool = True, words: int = WORDS, overlap: int = OVERLAP) -> list[Chunk]:
    chunks: list[Chunk] = []
    buffer: list[Block] = []
    heading = ""

    def emit() -> None:
        nonlocal buffer
        body = "\n\n".join(block.text for block in buffer).strip()
        if body:
            chunks.append(Chunk(len(chunks), heading, body, buffer[0].first_line, buffer[-1].last_line))
        buffer = []

    for path, whole in split_blocks(text.splitlines(), headings):
        for block in slice_oversized(whole, words):
            if path != heading:
                emit()
                heading = path
            if buffer and sum(item.words for item in buffer) + block.words > words:
                tail = overlap_tail(buffer, overlap)
                emit()
                buffer = tail
            buffer.append(block)
    emit()
    return chunks
