from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import stat as modes
from pathlib import Path
from typing import Any

from hypersonic.context.squeeze import CHARS_PER_TOKEN, cap
from hypersonic.providers.base import ToolSpec
from hypersonic.tools.base import RESULT_TOKENS, Tool, ToolContext, ToolError

VIEW_LINES = 400
VIEW_CHARS = RESULT_TOKENS * CHARS_PER_TOKEN
DIR_ENTRIES = 200
LAST_LINE = -1
MAX_FILE_BYTES = 10_000_000
PATH_BYTES = 1024
NEW_FILE_MODE = 0o666
WRITE = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
LINE = re.compile(r".*\n|.+")
OUTSIDE = "{name} is outside the workspace {workspace}."
DONE = {"create": "Wrote", "str_replace": "Edited", "insert": "Inserted into"}

SPEC = ToolSpec(
    "files",
    "View, create or edit workspace files. view: file text (view_range [start, end], -1 = "
    "last line) or a directory listing. create: write file_text. str_replace: old_str must "
    "match exactly once. insert: add insert_text after line insert_line (0 = top).",
    {
        "type": "object",
        "properties": {
            "command": {"type": "string", "enum": ["view", "create", "str_replace", "insert"]},
            "path": {"type": "string"},
            "view_range": {"type": "array", "items": {"type": "integer"}},
            "file_text": {"type": "string"},
            "old_str": {"type": "string"},
            "new_str": {"type": "string"},
            "insert_line": {"type": "integer"},
            "insert_text": {"type": "string"},
        },
        "required": ["command", "path"],
    },
)


def resolve(ctx: ToolContext, raw: str, write: bool) -> Path:
    path = Path(raw).expanduser()
    path = (path if path.is_absolute() else ctx.cwd / path).resolve()
    roots = [ctx.workspace] if write else [ctx.workspace, ctx.session_dir]
    if not any(path.is_relative_to(root) for root in roots):
        raise ToolError(OUTSIDE.format(name=raw, workspace=ctx.workspace))
    return path


def real_path(fd: int) -> Path:
    """Where an open file or folder really is, whatever name it was opened by."""
    if hasattr(fcntl, "F_GETPATH"):
        return Path(os.fsdecode(fcntl.fcntl(fd, fcntl.F_GETPATH, bytes(PATH_BYTES)).split(b"\0", 1)[0]))
    return Path(os.readlink(f"/proc/self/fd/{fd}"))


def real_folder(folder: Path) -> Path:
    fd = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
    try:
        return real_path(fd)
    finally:
        os.close(fd)


def open_inside(ctx: ToolContext, path: Path, flags: int, write: bool) -> int:
    """Open a file through its folder, after checking where that folder really is.

    resolve() saw the path as it was a moment ago. A command still running in the shell can swap a link
    in since then, so the check is made on what was actually opened.
    """
    roots = [ctx.workspace] if write else [ctx.workspace, ctx.session_dir]
    try:
        if flags & os.O_CREAT:
            path.parent.mkdir(parents=True, exist_ok=True)
        parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    except OSError as error:
        raise ToolError(f"Can't open {path.name}. {error.strerror}.")
    try:
        if not any(real_path(parent).is_relative_to(real_folder(root)) for root in roots):
            raise ToolError(OUTSIDE.format(name=path.name, workspace=ctx.workspace))
        return os.open(path.name, flags | os.O_NOFOLLOW, NEW_FILE_MODE, dir_fd=parent)
    except OSError as error:
        raise ToolError(f"Can't open {path.name}. {error.strerror}.")
    finally:
        os.close(parent)


def stamp(path: Path) -> tuple[int, int]:
    status = path.stat()
    return status.st_mtime_ns, status.st_size


def need(args: dict[str, Any], name: str) -> Any:
    if name not in args:
        raise ToolError(f"{args['command']} needs {name}.")
    return args[name]


def read(ctx: ToolContext, path: Path, editing: bool = False) -> str:
    if not path.is_file():
        raise ToolError(f"{path.name} doesn't exist.")
    with os.fdopen(open_inside(ctx, path, os.O_RDONLY, write=False), "rb") as source:
        size = os.fstat(source.fileno()).st_size
        if size > MAX_FILE_BYTES:
            raise ToolError(f"{path.name} is {size:,} bytes, too big to load. Read parts of it with rg, head or sed.")
        data = source.read()
    if b"\0" in data:
        raise ToolError(f"{path.name} is a binary file.")
    try:
        return data.decode(errors="strict" if editing else "replace")
    except UnicodeDecodeError:
        raise ToolError(f"{path.name} isn't UTF-8 text, so an edit here would damage it. Change it with shell instead.")


def copy_into(source: Any, out: Any) -> None:
    shutil.copyfileobj(source, out)
    os.fchmod(out.fileno(), modes.S_IMODE(os.fstat(source.fileno()).st_mode))


def check_fresh(ctx: ToolContext, path: Path) -> None:
    known = ctx.seen.get(str(path))
    if known and path.exists() and stamp(path) != known:
        raise ToolError(f"{path.name} changed on disk since you last saw it. View it again first.")


def backup(ctx: ToolContext, path: Path) -> Path | None:
    if not path.exists():
        return None
    copy = ctx.new_file("undo", f"-{path.name}")
    with os.fdopen(open_inside(ctx, path, os.O_RDONLY, write=True), "rb") as source, copy.open("wb") as out:
        copy_into(source, out)
    return copy


def remember(ctx: ToolContext, path: Path, copy: Path | None) -> None:
    index = ctx.session_dir / "undo" / "index.jsonl"
    index.parent.mkdir(parents=True, exist_ok=True)
    with index.open("a") as out:
        out.write(json.dumps({"path": str(path), "backup": copy and str(copy)}) + "\n")


def view(ctx: ToolContext, path: Path, view_range: list[int] | None) -> str:
    if path.is_dir():
        names = sorted(entry.name + ("/" if entry.is_dir() else "") for entry in path.iterdir())
        more = len(names) - DIR_ENTRIES
        listing = "\n".join(names[:DIR_ENTRIES]) or "[empty directory]"
        return listing + (f"\n[... {more} more entries]" if more > 0 else "")
    text = read(ctx, path)
    ctx.seen[str(path)] = stamp(path)
    lines = [line.rstrip("\r\n") for line in LINE.findall(text)]
    total = len(lines)
    if not total:
        return "[empty file]"
    first, last = 1, total
    if view_range:
        if len(view_range) != 2:
            raise ToolError("view_range must be [start, end].")
        first = max(1, view_range[0])
        last = total if view_range[1] == LAST_LINE else min(total, view_range[1])
    if first > last:
        raise ToolError(f"{path.name} has {total} lines.")
    shown: list[str] = []
    size = 0
    for line in lines[first - 1 : last]:
        size += len(line) + 1
        if shown and (size > VIEW_CHARS or len(shown) == VIEW_LINES):
            break
        shown.append(line)
    end = first + len(shown) - 1
    body, _ = cap("\n".join(shown), RESULT_TOKENS)
    if first > 1 or end < total:
        body += f"\n[lines {first}-{end} of {total}. Use view_range for other parts.]"
    return body


def replace(text: str, old: str, new: str, path: Path) -> str:
    count = text.count(old) if old else 0
    if count == 0:
        raise ToolError(f"old_str not found in {path.name}. View the file and copy the exact text.")
    if count > 1:
        raise ToolError(f"old_str matches {count} places in {path.name}. Add nearby lines to make it unique.")
    return text.replace(old, new, 1)


def insert(text: str, after: int, addition: str, path: Path) -> str:
    lines = LINE.findall(text)
    if not 0 <= after <= len(lines):
        raise ToolError(f"insert_line must be 0 to {len(lines)} for {path.name}.")
    if after and not lines[after - 1].endswith("\n"):
        lines[after - 1] += "\n"
    block = addition if addition.endswith("\n") else addition + "\n"
    return "".join(lines[:after]) + block + "".join(lines[after:])


async def run(ctx: ToolContext, args: dict[str, Any]) -> str:
    command = args["command"]
    path = resolve(ctx, args["path"], write=command != "view")
    if command == "view":
        return view(ctx, path, args.get("view_range"))
    check_fresh(ctx, path)
    if command == "create":
        text = need(args, "file_text")
    elif command == "str_replace":
        text = replace(read(ctx, path, editing=True), need(args, "old_str"), args.get("new_str", ""), path)
    else:
        text = insert(read(ctx, path, editing=True), need(args, "insert_line"), need(args, "insert_text"), path)
    copy = backup(ctx, path)
    with os.fdopen(open_inside(ctx, path, WRITE, write=True), "w", encoding="utf-8") as out:
        out.write(text)
    # recorded only once the write worked, so undo never points at a change that didn't happen
    remember(ctx, path, copy)
    ctx.seen[str(path)] = stamp(path)
    return f"{DONE[command]} {path.relative_to(ctx.workspace)}."


def undo_last(ctx: ToolContext) -> str:
    index = ctx.session_dir / "undo" / "index.jsonl"
    lines = index.read_text().splitlines() if index.is_file() else []
    if not lines:
        return "Nothing to undo."
    record = json.loads(lines[-1])
    path = Path(record["path"])
    if record["backup"]:
        with open(record["backup"], "rb") as source, os.fdopen(open_inside(ctx, path, WRITE, write=True), "wb") as out:
            copy_into(source, out)
    else:
        path.unlink(missing_ok=True)
    index.write_text("".join(line + "\n" for line in lines[:-1]))
    ctx.seen.pop(str(path), None)
    return f"Restored {path}." if record["backup"] else f"Removed {path}."


TOOL = Tool(SPEC, run, lambda args: args.get("command") == "view")
