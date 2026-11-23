from __future__ import annotations

import functools
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from hypersonic.engine.session import home as hypersonic_home
from hypersonic.policy.rules import GUARDED

SECRET_DIRS = (
    ".ssh", ".aws", ".gnupg", ".kube", ".docker", ".azure", ".oci", ".terraform.d",
    ".config/gh", ".config/gcloud", ".config/gh-copilot",
)
SECRET_FILES = (
    ".netrc", ".npmrc", ".pypirc", ".git-credentials", ".pgpass", ".boto", ".s3cfg",
    ".zsh_history", ".bash_history", ".python_history", ".mysql_history", ".psql_history",
)
DEVICES = ("/dev/null", "/dev/zero", "/dev/stdout", "/dev/stderr")
# Root keeps its capabilities inside bubblewrap unless told otherwise, and with them a command can remount
# the read-only root or unmount the guards. A user namespace is what lets root set up the private network.
BWRAP_LOCKDOWN = ("--die-with-parent", "--unshare-user-try", "--cap-drop", "ALL")
BWRAP_ROOT = ("--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc")
PRIVATE_PIDS = "--unshare-pid"
SEATBELT_SIGNALS = ("(deny signal)", "(allow signal (target same-sandbox))")
# launching an app or scripting one runs code outside the sandbox
SEATBELT_APPS = ("(deny lsopen)", "(deny appleevent-send)")
# with the network on, local sockets (Docker, tmux, the ssh agent) stay shut. Name lookups go through one, so it stays open.
SEATBELT_SOCKETS = (
    "(deny network-outbound (remote unix-socket))",
    '(allow network-outbound (literal "/private/var/run/mDNSResponder"))',
)
SYSTEM_TEMP = ("/private/tmp", "/private/var/tmp")
CS_DARWIN_USER_TEMP_DIR = 65537
CS_DARWIN_USER_CACHE_DIR = 65538
OWN_INDEX = ("index.db", "index.db-wal", "index.db-shm")
BLOCKED_SIGNS = (
    "Operation not permitted",
    "Read-only file system",
    "Could not resolve host",
    "Network is unreachable",
)
HINT = (
    "[sandbox: writes are limited to the workspace and temp folders, some credential folders "
    "can't be read, and the network is off unless the user allowed it]"
)
UNAVAILABLE = (
    "No sandbox tool found (sandbox-exec on macOS, bwrap on Linux). "
    "The user can start the session with --mode yolo to run without one."
)


class SandboxUnavailable(Exception):
    pass


@dataclass(frozen=True)
class Sandbox:
    workspace: Path
    session_dir: Path
    network: bool = False

    @property
    def tmp(self) -> Path:
        return self.session_dir / "tmp"

    @property
    def cwd_file(self) -> Path:
        return self.session_dir / "cwd"


def secrets(user_home: Path) -> tuple[list[Path], list[Path]]:
    folders = [user_home / name for name in SECRET_DIRS]
    # Hypersonic's own settings files hold its token and the user's keys, and a provider plug-in may keep one too
    own = hypersonic_home().resolve()
    files = [user_home / name for name in SECRET_FILES] + [own / name for name in OWN_INDEX] + sorted(own.glob("*.json"))
    return folders, files


def temp_folders() -> list[Path]:
    found = [Path(name) for name in SYSTEM_TEMP]
    if sys.platform != "darwin":
        return found
    for key in (CS_DARWIN_USER_TEMP_DIR, CS_DARWIN_USER_CACHE_DIR):
        try:
            value = os.confstr(key)
        except (ValueError, OSError):
            continue
        if value:
            found.append(Path(value).resolve())
    return found


def quote(path: Path | str) -> str:
    # Seatbelt reads the letters themselves, not \u escapes, so a path like café must stay as written
    return json.dumps(str(path), ensure_ascii=False)


def seatbelt(box: Sandbox, user_home: Path) -> str:
    folders, files = secrets(user_home)
    writable = [f"(subpath {quote(folder)})" for folder in (box.workspace, box.tmp, *temp_folders())]
    writable.append(f"(literal {quote(box.cwd_file)})")
    writable += [f"(literal {quote(device)})" for device in DEVICES]
    writable += ['(regex #"^/dev/tty")', '(regex #"^/dev/fd/")']
    protected = [f"(subpath {quote(box.workspace / name)})" for name in GUARDED]
    hidden = [f"(subpath {quote(folder)})" for folder in folders] + [f"(literal {quote(item)})" for item in files]
    rules = [
        "(version 1)",
        "(allow default)",
        "(deny file-write*)",
        f"(allow file-write* {' '.join(writable)})",
        f"(deny file-write* {' '.join(protected)})",
        f"(deny file-read* {' '.join(hidden)})",
        *SEATBELT_SIGNALS,
        *SEATBELT_APPS,
        *(SEATBELT_SOCKETS if box.network else ("(deny network*)",)),
    ]
    return "\n".join(rules)


@functools.lru_cache(maxsize=1)
def private_pids() -> bool:
    """A private process list stops a command from signalling the harness or the user's other programs.

    It needs a fresh /proc, which most containers refuse to mount, so it is used only where it works.
    """
    if not shutil.which("bwrap"):
        return False
    probe = ["bwrap", *BWRAP_LOCKDOWN, PRIVATE_PIDS, *BWRAP_ROOT, "true"]
    return subprocess.run(probe, capture_output=True).returncode == 0


def bubblewrap(box: Sandbox, user_home: Path) -> list[str]:
    folders, files = secrets(user_home)
    args = ["bwrap", *BWRAP_LOCKDOWN, *([PRIVATE_PIDS] if private_pids() else []), *BWRAP_ROOT]
    args += ["--tmpfs", "/tmp"]
    for path in (box.workspace, box.tmp, box.cwd_file):
        args += ["--bind", str(path), str(path)]
    for name in GUARDED:
        if (box.workspace / name).exists():
            args += ["--ro-bind", str(box.workspace / name), str(box.workspace / name)]
    for folder in folders:
        if folder.is_dir():
            args += ["--tmpfs", str(folder)]
    for item in files:
        if item.is_file():
            args += ["--ro-bind", "/dev/null", str(item)]
    if not box.network:
        args.append("--unshare-net")
    return args


def wrap(argv: list[str], box: Sandbox) -> list[str]:
    box.tmp.mkdir(parents=True, exist_ok=True)
    box.cwd_file.touch()
    user_home = Path.home().resolve()
    if sys.platform == "darwin" and shutil.which("sandbox-exec"):
        return ["sandbox-exec", "-p", seatbelt(box, user_home), *argv]
    if sys.platform.startswith("linux") and shutil.which("bwrap"):
        return [*bubblewrap(box, user_home), *argv]
    raise SandboxUnavailable(UNAVAILABLE)


def blocked(output: str) -> bool:
    return any(sign in output for sign in BLOCKED_SIGNS)
