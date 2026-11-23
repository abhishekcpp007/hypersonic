from __future__ import annotations

import re
from itertools import groupby

CHARS_PER_TOKEN = 4
REPEAT_MIN = 3
ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])")


def clean(text: str) -> str:
    lines = [ANSI.sub("", raw).rstrip("\r").rsplit("\r", 1)[-1].rstrip() for raw in text.split("\n")]
    out: list[str] = []
    for line, group in groupby(lines):
        count = sum(1 for _ in group)
        if not line:
            out.append("")
        elif count >= REPEAT_MIN:
            out.append(f"{line}  [repeated {count}x]")
        else:
            out.extend([line] * count)
    return "\n".join(out).strip("\n")


def cap(text: str, max_tokens: int) -> tuple[str, int]:
    limit = max_tokens * CHARS_PER_TOKEN
    if len(text) <= limit:
        return text, 0
    half = limit // 2
    head = text[:half].rsplit("\n", 1)[0]
    tail = text[-half:].split("\n", 1)[-1]
    cut = len(text) - len(head) - len(tail)
    return f"{head}\n[... {cut:,} chars cut ...]\n{tail}", cut
