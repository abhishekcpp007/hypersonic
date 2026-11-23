from __future__ import annotations

import asyncio
import os
import re
import signal
from pathlib import Path
from typing import Any

from hypersonic.context.squeeze import CHARS_PER_TOKEN, clean
from hypersonic.policy.sandbox import HINT, Sandbox, SandboxUnavailable, blocked, wrap
from hypersonic.providers.base import ToolSpec
from hypersonic.tools.base import RESULT_TOKENS, Tool, ToolContext, ToolError

DEFAULT_TIMEOUT = 120
MIN_TIMEOUT = 1
MAX_TIMEOUT = 1800
DRAIN_SECONDS = 2
MAX_SPILL_BYTES = 50 * 1024 * 1024
RESULT_CHARS = RESULT_TOKENS * CHARS_PER_TOKEN
CHUNK_BYTES = 64 * 1024
CWD_ENV = "HYPERSONIC_CWD_FILE"
SECRET_ENV = re.compile(r"KEY|TOKEN|SECRET|PASS|CREDENTIAL|COOKIE|JWT|_DSN|_URI|_PWD|_AUTH(_|$)|_PAT$", re.IGNORECASE)
LOGIN_IN_URL = re.compile(r"://[^/@\s]+:[^/@\s]+@")
CUT = "\n[output cut here: {extra:,} more bytes were not saved]\n"
HELD = "\n[a background job kept this command's output open, and it is closed now. Start such jobs with: cmd > log 2>&1 &]"
STRIP_NAMES = frozenset({"SSH_AUTH_SOCK", "SSH_AGENT_PID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"})
TRAP = f"trap 'pwd -P > \"${CWD_ENV}\"' EXIT\n"
# A helper that waits on a pipe from the server. A word on it means the command ended normally. The pipe closing
# with no word means the server died, and then nothing is left to enforce the timeout, so the helper ends the
# whole command. It is detached (two levels of subshell) so a plain `wait` in the command doesn't wait for it.
WATCH = '( ( {{ : <&{fd}; }} 2>/dev/null || exit; read -r -u {fd} word; [ -z "$word" ] && kill -KILL 0 ) >/dev/null 2>&1 & )\n'
ENDED = b"ended\n"
FAILURE = re.compile(r"\n\[(exit code -?\d+|killed after \d+s timeout)\]")

SPEC = ToolSpec(
    "shell",
    "Run a bash command in the workspace. Each call is a fresh process; only the working "
    f"directory carries over. Killed after {DEFAULT_TIMEOUT}s unless timeout (seconds) is set. Long output "
    "is cut and saved to a file named in the result. Background jobs: cmd > log 2>&1 &. A "
    "sandbox limits writes to the workspace and temp folders and blocks the network.",
    {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "integer"},
        },
        "required": ["command"],
    },
)


def child_env(box: Sandbox) -> dict[str, str]:
    env = {
        name: value for name, value in os.environ.items()
        if name not in STRIP_NAMES and not SECRET_ENV.search(name) and not LOGIN_IN_URL.search(value)
    }
    env[CWD_ENV] = str(box.cwd_file)
    env["TMPDIR"] = str(box.tmp)
    return env


def failed(result: str) -> bool:
    return bool(FAILURE.search("\n" + result))


def kill_group(proc: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


async def pump(stream: asyncio.StreamReader, path: Path) -> None:
    size = 0
    with path.open("wb") as out:
        while chunk := await stream.read(CHUNK_BYTES):
            if size < MAX_SPILL_BYTES:
                out.write(chunk)
            size += len(chunk)
        if size > MAX_SPILL_BYTES:
            out.write(CUT.format(extra=size - out.tell()).encode())


def excerpt(path: Path) -> tuple[str, bool]:
    size = path.stat().st_size
    if size <= RESULT_CHARS:
        return clean(path.read_bytes().decode(errors="replace")), False
    half = RESULT_CHARS // 2
    with path.open("rb") as source:
        head = source.read(half).decode(errors="replace")
        source.seek(-half, os.SEEK_END)
        tail = source.read().decode(errors="replace")
    omitted = size - 2 * half
    marker = f"\n[... {omitted:,} chars omitted. Full output ({size:,} bytes): {path} ...]\n"
    return clean(head) + marker + clean(tail), True


async def run(ctx: ToolContext, args: dict[str, Any]) -> str:
    timeout = max(MIN_TIMEOUT, min(args.get("timeout") or DEFAULT_TIMEOUT, MAX_TIMEOUT))
    if not ctx.cwd.is_dir():
        ctx.cwd = ctx.workspace
    box = Sandbox(ctx.workspace, ctx.session_dir, ctx.network)
    box.tmp.mkdir(parents=True, exist_ok=True)
    cwd_file = box.cwd_file
    still_here, server_end = os.pipe()
    argv = ["bash", "-c", TRAP + WATCH.format(fd=still_here) + args["command"]]
    if ctx.sandbox:
        try:
            argv = wrap(argv, box)
        except SandboxUnavailable as error:
            raise ToolError(str(error))
    spill = ctx.new_file("out", ".txt")
    # The output pipe is opened here, not by asyncio. That way the wait ends when the command does, even if a
    # background job it started still holds the output, and the pipe can be closed afterwards.
    read_end, write_end = os.pipe()
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=ctx.cwd,
            env=child_env(box),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=write_end,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
            pass_fds=(still_here,),
        )
    except BaseException:
        os.close(read_end)
        os.close(server_end)
        raise
    finally:
        os.close(write_end)
        os.close(still_here)
    output = asyncio.StreamReader()
    pipe, _ = await asyncio.get_running_loop().connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(output), os.fdopen(read_end, "rb")
    )
    reader = asyncio.create_task(pump(output, spill))
    held = False
    try:
        code = await asyncio.wait_for(proc.wait(), timeout)
        note = f"\n[exit code {code}]" if code else ""
    except asyncio.TimeoutError:
        kill_group(proc)
        note = f"\n[killed after {timeout}s timeout]"
    except asyncio.CancelledError:
        kill_group(proc)
        raise
    finally:
        await asyncio.wait({reader}, timeout=DRAIN_SECONDS)
        held = not reader.done()
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)
        pipe.close()
        try:
            os.write(server_end, ENDED)
        except OSError:
            pass
        os.close(server_end)
    if held:
        note += HELD

    # empty when the command was killed or replaced itself with exec before it could say where it ended
    moved = cwd_file.read_text().strip() if cwd_file.exists() else ""
    if moved and Path(moved).is_absolute() and Path(moved).is_dir():
        ctx.cwd = Path(moved)
    body, truncated = excerpt(spill)
    if ctx.sandbox and note and blocked(body):
        note += f"\n{HINT}"
    if not truncated:
        spill.unlink()
    return (body + note).strip() or "[no output]"


TOOL = Tool(SPEC, run, lambda args: False)
