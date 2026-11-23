from __future__ import annotations

import re
from functools import lru_cache

WORD = re.compile(r"[^\W\d_][\w\-]*|\d[\d_\-\.]*")
SPLIT = re.compile(r"[-_\.]")
CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
EDGES = "-_."
SIBILANT = ("s", "x", "z", "ch", "sh")
CACHE_SIZE = 1 << 17
STOP = frozenset(
    """a an and are as at be but by can do does for from has have how i if in is it its of on or
    our that the then there they this to was we what when where which who why will with you
    your""".split()
)


def stem(word: str) -> str:
    if len(word) > 4 and word.endswith("es") and word[:-2].endswith(SIBILANT):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


@lru_cache(maxsize=CACHE_SIZE)
def expand(raw: str) -> tuple[str, ...]:
    raw = raw.strip(EDGES)
    low = raw.lower()
    out = [low]
    parts = [part for part in SPLIT.split(low) if part]
    if len(parts) > 1:
        out.extend(parts)
    if CAMEL.search(raw):
        out.extend(part.lower() for part in CAMEL.split(raw) if part)
    return tuple(stem(token) for token in out if token and token not in STOP)


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for raw in WORD.findall(text):
        out.extend(expand(raw))
    return out
