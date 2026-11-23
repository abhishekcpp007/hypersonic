import asyncio
import re
from pathlib import Path

import os
import signal
import subprocess
import sys
import time

import pytest
from test_sandbox import sandbox_works

from hypersonic.tools import shell as shell_tool
from hypersonic.tools.shell import RESULT_CHARS, TOOL


def shell(ctx, command, **args):
    return asyncio.run(TOOL.run(ctx, {"command": command, **args}))


def test_output_and_exit_code(ctx):
    assert shell(ctx, "echo hello") == "hello"
    assert shell(ctx, "echo oops >&2; exit 3") == "oops\n[exit code 3]"
    assert shell(ctx, "true") == "[no output]"


def test_working_directory_carries_over(ctx, workspace):
    (workspace / "sub").mkdir()
    shell(ctx, "cd sub")
    assert shell(ctx, "pwd") == str(workspace / "sub")
    assert ctx.cwd == workspace / "sub"


def test_timeout_kills_the_command_and_keeps_partial_output(ctx):
    result = shell(ctx, "echo started; sleep 30; echo never", timeout=1)
    assert result == "started\n[killed after 1s timeout]"


def test_background_job_does_not_block(ctx, workspace):
    assert shell(ctx, "sleep 30 > /dev/null 2>&1 & echo launched") == "launched"


def test_a_background_job_that_keeps_the_output_open_does_not_hold_the_command(ctx):
    began = time.monotonic()
    result = shell(ctx, "sleep 20 & echo started", timeout=15)
    assert result.startswith("started\n[a background job kept") and "cmd > log 2>&1 &" in result
    assert time.monotonic() - began < 10


def test_the_folder_survives_commands_that_never_say_where_they_ended(ctx, workspace):
    (workspace / "sub").mkdir()
    shell(ctx, "cd sub")
    ctx.session_dir.joinpath("cwd").write_text("")
    shell(ctx, "exec true")
    assert ctx.cwd == workspace / "sub"
    shell(ctx, "rmdir ../sub")
    assert shell(ctx, "pwd") == str(workspace)
    assert shell(ctx, "echo quick", timeout=-5) == "quick"


ORPHAN = """
import asyncio, sys
from pathlib import Path
from hypersonic.tools.base import ToolContext
from hypersonic.tools.shell import TOOL
work = Path(sys.argv[1])
asyncio.run(TOOL.run(ToolContext(work, work / "session", None, sandbox=sys.argv[2] == "boxed"), {"command": "sleep 647"}))
"""


@pytest.mark.parametrize("mode", ["plain", "boxed"])
def test_a_command_ends_when_the_server_running_it_is_killed(tmp_path, mode):
    if mode == "boxed" and not sandbox_works():
        pytest.skip("no working sandbox tool on this machine")
    (tmp_path / "session").mkdir()
    napping = lambda: subprocess.run(["pgrep", "-fx", "sleep 647"], capture_output=True, text=True).stdout.split()
    server = subprocess.Popen([sys.executable, "-c", ORPHAN, str(tmp_path), mode])
    try:
        deadline = time.monotonic() + 10
        while not napping() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert napping()
        server.kill()
        deadline = time.monotonic() + 5
        while napping() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not napping()
    finally:
        server.kill()
        for pid in napping():
            os.kill(int(pid), signal.SIGKILL)


def test_the_watcher_does_not_get_in_the_way_of_ordinary_commands(ctx, workspace):
    began = time.monotonic()
    assert shell(ctx, "sleep 0.2 & sleep 0.1 & wait; echo all done") == "all done"
    assert shell(ctx, "sleep 5 > /dev/null 2>&1 & echo left running") == "left running"
    assert shell(ctx, "exec echo replaced") == "replaced"
    assert time.monotonic() - began < 4
    time.sleep(0.3)
    assert subprocess.run(["pgrep", "-fx", "sleep 5"], capture_output=True).stdout.strip()


def test_secret_env_vars_are_stripped(ctx, monkeypatch):
    monkeypatch.setenv("MY_API_KEY", "abc123")
    monkeypatch.setenv("PLAIN_SETTING", "visible")
    assert shell(ctx, 'echo "${MY_API_KEY:-none} $PLAIN_SETTING"') == "none visible"
    hidden = {"DATABASE_URL": "postgres://app:hunter2@db/prod", "MYSQL_PWD": "x", "DB_PASS": "x", "DOCKER_AUTH_CONFIG": "x",
              "GH_PAT": "x", "SESSION_COOKIE": "x", "JWT": "x", "HTTPS_PROXY_LOGIN": "http://me:pw@proxy:8080"}
    kept = {"GIT_AUTHOR_NAME": "A", "PIP_INDEX_URL": "https://pypi.org/simple", "PATTERN": "x"}
    for name, value in {**hidden, **kept}.items():
        monkeypatch.setenv(name, value)
    seen = shell(ctx, "env").split("\n")
    assert not [line for line in seen if line.split("=")[0] in hidden]
    assert {line.split("=")[0] for line in seen} >= set(kept)


def test_output_past_the_save_limit_says_so(ctx, monkeypatch):
    monkeypatch.setattr(shell_tool, "MAX_SPILL_BYTES", 100_000)
    assert "more bytes were not saved]" in shell(ctx, "yes line | head -c 400000")


def test_long_output_is_cut_and_saved(ctx):
    result = shell(ctx, "seq 1 20000")
    assert result.startswith("1\n2\n")
    match = re.search(r"\[\.\.\. ([\d,]+) chars omitted\. Full output \(([\d,]+) bytes\): (\S+) \.\.\.\]", result)
    omitted, size, path = int(match[1].replace(",", "")), int(match[2].replace(",", "")), match[3]
    spill = Path(path)
    assert spill.read_text().splitlines()[-1] == "20000"
    assert size == spill.stat().st_size
    assert omitted == size - RESULT_CHARS  # the TRUE amount dropped, not an excerpt-relative count (the bug)
    assert omitted > 100_000


def test_short_output_leaves_no_spill_file(ctx):
    shell(ctx, "echo small")
    assert list((ctx.session_dir / "out").iterdir()) == []
