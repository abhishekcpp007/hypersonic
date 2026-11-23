import asyncio
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from hypersonic.engine.session import Session, home
from hypersonic.policy import sandbox
from hypersonic.policy.sandbox import BWRAP_LOCKDOWN, HINT, UNAVAILABLE, Sandbox, SandboxUnavailable, blocked, bubblewrap, seatbelt, wrap
from hypersonic.tools.base import ToolContext, ToolError
from hypersonic.tools.shell import TOOL

HERE = Path(__file__).resolve().parent
OUTSIDE = HERE / "red-team-probe.txt"
SECRETS = ("FAKE-PRIVATE-KEY", "SERVER-TOKEN", "OLD-COMMAND-WITH-A-PASSWORD")
RESOLVER = "/private/var/run/mDNSResponder"


def sandbox_works() -> bool:
    """True when this machine can really start the sandbox (macOS Seatbelt, or bubblewrap with namespaces allowed)."""
    with tempfile.TemporaryDirectory(dir=HERE) as scratch:
        try:
            probe = wrap(["true"], Sandbox(Path(scratch), Path(scratch) / "session"))
        except SandboxUnavailable:
            return False
        return subprocess.run(probe, capture_output=True).returncode == 0


needs_sandbox = pytest.mark.skipif(not sandbox_works(), reason="no working sandbox tool on this machine")


@pytest.fixture
def fake_home(monkeypatch):
    # not under /tmp: the Linux sandbox gives every command a private /tmp, which would hide these files by accident
    with tempfile.TemporaryDirectory(prefix="home-", dir=HERE) as made:
        user_home = Path(made).resolve()
        (user_home / ".ssh").mkdir()
        (user_home / ".ssh" / "id_test").write_text("FAKE-PRIVATE-KEY")
        (user_home / ".zsh_history").write_text("OLD-COMMAND-WITH-A-PASSWORD")
        monkeypatch.setenv("HOME", str(user_home))
        yield user_home


@pytest.fixture
def boxed(workspace, store, fake_home):
    session = Session.create(workspace)
    return ToolContext(workspace, session.dir, store, sandbox=True)


def shell(ctx, command):
    return asyncio.run(TOOL.run(ctx, {"command": command}))


@needs_sandbox
def test_normal_work_inside_the_workspace_still_runs(boxed, workspace):
    (workspace / "sub").mkdir()
    assert shell(boxed, "echo hi > made.txt && cat made.txt | tr a-z A-Z") == "HI"
    shell(boxed, "cd sub")
    assert shell(boxed, "pwd") == str(workspace / "sub")
    assert shell(boxed, 'f=$(mktemp) && echo temp > "$f" && cat "$f"') == "temp"
    assert shell(boxed, "python3 -c 'import tempfile; tempfile.NamedTemporaryFile(); print(1)'") == "1"


@needs_sandbox
def test_red_team_reads_and_writes_are_blocked(boxed, workspace, fake_home):
    (workspace / ".git" / "hooks").mkdir(parents=True)
    (home() / "server.json").write_text('{"token": "SERVER-TOKEN"}')
    reads = {
        "read ssh key": "cat ~/.ssh/id_test",
        "list ssh folder": "ls ~/.ssh",
        "read shell history": "cat ~/.zsh_history",
        "read the server token": f"cat {home() / 'server.json'}",
    }
    writes = {
        "write outside the workspace": f"echo x > {OUTSIDE}",
        "write into the home folder": "echo x > ~/planted.txt",
        "plant a git hook": "echo x > .git/hooks/pre-commit",
        "rewrite git config": "echo x > .git/config",
    }
    try:
        for name, command in reads.items():
            assert not any(secret in shell(boxed, command) for secret in SECRETS), name
        assert "id_test" not in shell(boxed, "ls ~/.ssh")
        for name, command in writes.items():
            result = shell(boxed, command)
            assert blocked(result) and result.endswith(HINT), name
        assert not OUTSIDE.exists()
    finally:
        OUTSIDE.unlink(missing_ok=True)
    assert not (fake_home / "planted.txt").exists()
    assert not (workspace / ".git" / "hooks" / "pre-commit").exists()
    assert not (workspace / ".git" / "config").exists()


@needs_sandbox
def test_a_command_cannot_take_the_sandbox_apart(boxed, workspace, fake_home):
    (workspace / ".git" / "hooks").mkdir(parents=True)
    (home() / "server.json").write_text('{"token": "SERVER-TOKEN"}')
    attempts = [
        f"mount -o remount,bind,rw / ; echo x > {OUTSIDE}",
        f"umount -l {workspace / '.git'} ; echo x > {workspace / '.git/hooks/pre-commit'}",
        f"umount -l {fake_home / '.ssh'} ; cat {fake_home / '.ssh/id_test'}",
        f"umount -l {fake_home / '.zsh_history'} ; cat {fake_home / '.zsh_history'}",
        f"umount -l {home() / 'server.json'} ; cat {home() / 'server.json'}",
    ]
    try:
        for command in attempts:
            assert not any(secret in shell(boxed, command) for secret in SECRETS), command
        assert not OUTSIDE.exists()
    finally:
        OUTSIDE.unlink(missing_ok=True)
    assert not (workspace / ".git" / "hooks" / "pre-commit").exists()


@needs_sandbox
def test_a_command_cannot_signal_processes_outside_the_sandbox(boxed):
    if sys.platform.startswith("linux") and not sandbox.private_pids():
        pytest.skip("this Linux cannot give the sandbox a private process list (usual inside containers)")
    assert "reached" not in shell(boxed, f"kill -0 {os.getpid()} && echo reached")
    assert shell(boxed, "sleep 30 & kill $! && wait $! 2>/dev/null; echo own-child-stopped") == "own-child-stopped"


@needs_sandbox
def test_network_is_off_unless_allowed(boxed):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        connect = f"exec 3<>/dev/tcp/127.0.0.1/{listener.getsockname()[1]} && echo connected"
        assert "connected" not in shell(boxed, connect)
        boxed.network = True
        assert shell(boxed, connect) == "connected"


@needs_sandbox
def test_folders_with_names_outside_english_work_and_stay_protected(store, monkeypatch):
    with tempfile.TemporaryDirectory(prefix="h\u00f4m\u00e9-", dir=HERE) as made:
        user_home = Path(made).resolve()
        (user_home / ".ssh").mkdir()
        (user_home / ".ssh" / "id_test").write_text("FAKE-PRIVATE-KEY")
        monkeypatch.setenv("HOME", str(user_home))
        workspace = user_home / "\u092a\u094d\u0930\u094b\u091c\u0947\u0915\u094d\u091f w\u00f6rk"
        workspace.mkdir()
        ctx = ToolContext(workspace, Session.create(workspace).dir, store, sandbox=True)
        assert shell(ctx, "echo hi > made.txt && cat made.txt") == "hi"
        assert "FAKE-PRIVATE-KEY" not in shell(ctx, "cat ~/.ssh/id_test")


def dial(path):
    return f"python3 -c \"import socket; socket.socket(socket.AF_UNIX).connect('{path}'); print('reached')\""


@needs_sandbox
@pytest.mark.skipif(sys.platform != "darwin", reason="these rules are in the macOS profile")
def test_with_the_network_on_local_sockets_stay_shut_but_name_lookups_work(boxed):
    boxed.network = True
    private = HERE / "probe.sock"
    with socket.socket(socket.AF_UNIX) as listener:
        listener.bind(str(private))
        listener.listen()
        try:
            assert blocked(shell(boxed, dial(private)))
        finally:
            private.unlink()
    assert shell(boxed, dial(RESOLVER)) == "reached"


@needs_sandbox
@pytest.mark.skipif(sys.platform != "darwin", reason="these rules are in the macOS profile")
def test_a_command_cannot_launch_an_app(boxed, workspace):
    (workspace / "run.command").write_text("#!/bin/sh\ntouch launched.txt\n")
    (workspace / "run.command").chmod(0o755)
    shell(boxed, "open -g run.command; sleep 1")
    assert not (workspace / "launched.txt").exists()


def test_shell_fails_closed_without_a_sandbox_tool(boxed, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(ToolError, match="No sandbox tool found"):
        shell(boxed, "echo hi")
    assert "--mode yolo" in UNAVAILABLE


def test_profiles_name_the_right_paths(workspace, tmp_path, fake_home, monkeypatch):
    box = Sandbox(workspace, tmp_path / "session")
    home().mkdir(parents=True, exist_ok=True)
    for own_secret in ("server.json", "keys.json", "anything-a-provider-saved.json"):
        (home() / own_secret).write_text("{}")
    profile = seatbelt(box, fake_home)
    assert f'(subpath "{workspace}")' in profile
    assert f'(subpath "{workspace / ".git"}")' in profile
    assert f'(subpath "{fake_home / ".ssh"}")' in profile
    for own_secret in ("server.json", "keys.json", "anything-a-provider-saved.json"):
        assert f'(literal "{home().resolve() / own_secret}")' in profile
    assert f'(literal "{home().resolve() / "index.db"}")' in profile
    assert profile.endswith("(deny network*)") and "(deny lsopen)" in profile and "(deny appleevent-send)" in profile
    networked = seatbelt(Sandbox(workspace, tmp_path / "session", network=True), fake_home)
    assert "(deny network*)" not in networked and f'(literal "{RESOLVER}")' in networked
    assert '(subpath "/x/w\u00f6rk")' in seatbelt(Sandbox(Path("/x/w\u00f6rk"), tmp_path / "session"), fake_home)

    (workspace / ".git" / "hooks").mkdir(parents=True)
    monkeypatch.setattr(sandbox, "private_pids", lambda: True)
    assert bubblewrap(box, fake_home)[len(BWRAP_LOCKDOWN) + 1] == "--unshare-pid"
    monkeypatch.setattr(sandbox, "private_pids", lambda: False)
    args = bubblewrap(box, fake_home)
    assert args[: len(BWRAP_LOCKDOWN) + 4] == ["bwrap", *BWRAP_LOCKDOWN, "--ro-bind", "/", "/"]
    assert BWRAP_LOCKDOWN[-2:] == ("--cap-drop", "ALL") and "--unshare-user-try" in BWRAP_LOCKDOWN
    assert ["--tmpfs", "/tmp"] == args[args.index("--tmpfs") : args.index("--tmpfs") + 2]
    assert ["--bind", str(workspace), str(workspace)] == args[args.index(str(workspace)) - 1 : args.index(str(workspace)) + 2]
    guard, hidden, history = str(workspace / ".git"), str(fake_home / ".ssh"), str(fake_home / ".zsh_history")
    assert args[args.index(guard) - 1 : args.index(guard) + 2] == ["--ro-bind", guard, guard]
    assert args[args.index(hidden) - 1] == "--tmpfs"
    assert args[args.index(history) - 2 : args.index(history) + 1] == ["--ro-bind", "/dev/null", history]
    assert args[-1] == "--unshare-net" and args.index(guard) > args.index(str(workspace))
