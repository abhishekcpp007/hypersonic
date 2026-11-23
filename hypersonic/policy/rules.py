from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import bashlex
import bashlex.ast

from hypersonic.providers.base import ToolCall

ALLOW, ASK, DENY = "allow", "ask", "deny"
PLAN, ASK_MODE, AUTO, YOLO = "plan", "ask", "auto", "yolo"
MODES = (PLAN, ASK_MODE, AUTO, YOLO)
NULL_DEVICE = "/dev/null"
GUARDED = (".git", ".hypersonic")
WRAPPERS = frozenset(
    """bash sh zsh ksh dash eval command xargs env timeout nohup nice setsid watch ssh
    exec builtin trap stdbuf script caffeinate arch busybox doas""".split()
)
EXPANSIONS = ("$", "`")
SUBSTITUTIONS = ("$(", "`")
GLOBS = ("{", "*", "?")
FIND_RUNS = ("-exec", "-execdir", "-ok", "-okdir")
LONG_RECURSIVE = "--recursive"
DEFAULT_ASK: tuple[str, ...] = (
    "git push*", "rm -rf*", "rm -fr*", "git reset --hard*", "git clean*",
    "git * push*", "git * clean*", "git reset *--h*", "git * reset *--h*",
)
BAD_RULES = 'In config.toml, {key} must be a list of patterns, like ["git commit*"].'
DEFAULT_DENY: tuple[str, ...] = ("sudo", "sudo *")
PLAN_ONLY = "Plan mode is read-only, and this call could change something."
READ_ONLY = frozenset(
    """ls cat head tail wc pwd cd echo printf which type file stat du df date whoami hostname id
    uname printenv ps grep egrep fgrep rg find fd tree sort uniq cut tr diff cmp basename dirname
    realpath readlink true false test [ seq sleep jq column nl tac rev comm paste""".split()
)
GIT_READ_ONLY = frozenset("status diff log show rev-parse ls-files blame describe shortlog grep".split())
UNSAFE_FLAGS = {
    "find": ("-delete", *FIND_RUNS, "-fprint", "-fprint0", "-fprintf", "-fls"),
    "fd": ("-x", "--exec", "-X", "--exec-batch"),
    "rg": ("--pre", "--hostname-bin"),
    "sort": ("-o", "--output", "--compress-program"),
    "tree": ("-o",),
    "date": ("-s", "--set"),
    "file": ("-C", "--compile"),
    "git": ("--output", "-O", "--open-files-in-pager", "--ext-diff"),
}
OUTPUT_ARGUMENT = {"uniq": 2}


@dataclass(frozen=True)
class Rules:
    allow: tuple[str, ...] = ()
    ask: tuple[str, ...] = DEFAULT_ASK
    deny: tuple[str, ...] = DEFAULT_DENY
    web: tuple[str, ...] = ()


def patterns(config: dict[str, Any], section: str, key: str) -> tuple[str, ...]:
    table = config.get(section, {})
    found = table.get(key, []) if isinstance(table, dict) else None
    # a bare string would be read letter by letter, and the pattern "*" allows everything
    if not isinstance(found, list) or not all(isinstance(item, str) for item in found):
        raise ValueError(BAD_RULES.format(key=f"{section}.{key}"))
    return tuple(found)


def load_rules(path: Path) -> Rules:
    if not path.is_file():
        return Rules()
    config = tomllib.loads(path.read_text())
    return Rules(
        patterns(config, "rules", "allow"),
        DEFAULT_ASK + patterns(config, "rules", "ask"),
        DEFAULT_DENY + patterns(config, "rules", "deny"),
        patterns(config, "web", "allow"),
    )


class Collector(bashlex.ast.nodevisitor):
    def __init__(self) -> None:
        self.commands: list[list[str]] = []
        self.writes = False
        self.opaque = False

    def visitcommand(self, node: Any, parts: Any) -> None:
        words = [part.word for part in parts if part.kind == "word"]
        self.commands.append(words)
        names = [part.word.split("=", 1)[0] for part in parts if part.kind == "assignment"]
        # VAR=x in front of a command, or a new value for a variable the shell already has, can change what runs
        if names and (words or any(name in os.environ for name in names)):
            self.writes = True
        # a flag built when the command runs can't be read here, so it is assumed to be a dangerous one
        built = any(part.kind == "word" and part.parts for part in parts[1:])
        if built and words and words[0] in UNSAFE_FLAGS:
            self.writes = True

    def visitheredoc(self, node: Any, value: str) -> None:
        # bash runs $(...) inside a here-document body and inside ${...}, where the parser shows only text
        self.opaque = self.opaque or any(sign in value for sign in SUBSTITUTIONS)

    visitparameter = visitheredoc

    def visitredirect(self, node: Any, input: Any, type: str, output: Any, heredoc: Any) -> None:
        target = getattr(output, "word", None)
        if ">" in type and target not in (None, NULL_DEVICE):
            self.writes = True


def split(command: str) -> tuple[list[list[str]], bool] | None:
    if not command.strip():
        return [], False
    collector = Collector()
    try:
        for tree in bashlex.parse(command):
            collector.visit(tree)
    except Exception:
        return None
    return None if collector.opaque else (collector.commands, collector.writes)


def matches(text: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatchcase(text, pattern) for pattern in patterns)


def deny_names(patterns: tuple[str, ...]) -> list[str]:
    """The executable names a deny pattern targets, e.g. 'sudo *' -> 'sudo'."""
    return [p.split()[0].rstrip("*") for p in patterns if p.split() and p.split()[0].rstrip("*")]


def bare(words: list[str]) -> list[str]:
    """The same command with its program named plainly, so /bin/rm and RM are judged as rm."""
    return [words[0].rsplit("/", 1)[-1].lower(), *words[1:]]


def wrapped_denied(commands: list[list[str]], deny: tuple[str, ...]) -> bool:
    """Catch a denied command hidden inside a wrapper's arguments (bash -c '...', xargs ...)."""
    names = deny_names(deny)
    return any(
        re.search(rf"(?<![\w.-]){re.escape(name)}(?![\w.-])", " ".join(words))
        for words in commands
        if is_wrapper(words)
        for name in names
    )


def is_wrapper(words: list[str]) -> bool:
    """True for a command that runs another command, or whose own name is only known when it runs."""
    if not words:
        return False
    runs_others = words[0] == "find" and any(flag in words for flag in FIND_RUNS)
    return words[0] in WRAPPERS or runs_others or any(sign in words[0] for sign in (*EXPANSIONS, *GLOBS))


def long_flag(word: str, flag: str) -> bool:
    """GNU tools take any unambiguous start of a long option, with or without =value."""
    stem = word.split("=", 1)[0]
    return len(stem) > len("--") and flag.startswith(stem)


def rm_recursive(words: list[str]) -> bool:
    if not words or words[0] != "rm":
        return False
    return any(
        long_flag(word, LONG_RECURSIVE) or (word.startswith("-") and not word.startswith("--") and ("r" in word or "R" in word))
        for word in words[1:]
    )


def flagged(word: str, flag: str) -> bool:
    if word.startswith("--"):
        return long_flag(word, flag)
    clustered = len(flag) == len("-x") and word.startswith("-") and flag[1] in word
    return word == flag or clustered or (word.startswith(("-", "{")) and "{" in word)


def read_only(words: list[str]) -> bool:
    if not words:
        return True
    name, rest = words[0], words[1:]
    if any(flagged(word, flag) for word in rest for flag in UNSAFE_FLAGS.get(name, ())):
        return False
    plain = [word for word in rest if not word.startswith("-")]
    if name in OUTPUT_ARGUMENT and len(plain) >= OUTPUT_ARGUMENT[name]:
        return False
    if name == "git":
        return bool(rest) and rest[0] in GIT_READ_ONLY
    return name in READ_ONLY


def judge_shell(command: str, mode: str, rules: Rules) -> tuple[str, str]:
    if mode == YOLO:
        return ALLOW, ""
    parsed = split(command)
    if parsed is None:
        return (DENY, PLAN_ONLY) if mode == PLAN else (ASK, "")
    commands, writes = parsed
    named = [bare(words) for words in commands if words]
    texts = [" ".join(words) for words in (*commands, *named) if words]
    for text in texts:
        if matches(text, rules.deny):
            return DENY, f"Blocked by a deny rule: {text}"
    if wrapped_denied(named, rules.deny):
        return DENY, "Blocked: a denied command appears inside a wrapper (bash -c, eval, ssh, …)."
    risky = any(is_wrapper(words) or rm_recursive(words) for words in named)
    flagged = risky or any(matches(text, rules.ask) for text in texts)
    harmless = not writes and not flagged and all(read_only(words) for words in commands)
    if mode == PLAN:
        return (ALLOW, "") if harmless else (DENY, PLAN_ONLY)
    if flagged:
        return ASK, ""
    approved = not writes and all(
        read_only(words) or matches(" ".join(words), rules.allow) for words in commands
    )
    return (ALLOW, "") if mode == AUTO or harmless or approved else (ASK, "")


def guarded(raw: str, workspace: Path, cwd: Path) -> bool:
    path = Path(raw).expanduser()
    path = (path if path.is_absolute() else cwd / path).resolve()
    if not path.is_relative_to(workspace):
        return False
    # the Mac's disk ignores case, so .GIT/hooks is the same folder as .git/hooks
    inside = path.relative_to(workspace).parts
    return bool(inside) and inside[0].casefold() in GUARDED


def judge_files(args: dict[str, Any], mode: str, workspace: Path, cwd: Path) -> tuple[str, str]:
    if args["command"] == "view" or mode == YOLO:
        return ALLOW, ""
    if guarded(args["path"], workspace, cwd):
        return DENY, f"{args['path']} is protected and can't be changed by the agent."
    if mode == PLAN:
        return DENY, PLAN_ONLY
    return (ALLOW, "") if mode == AUTO else (ASK, "")


@dataclass
class Policy:
    mode: str = ASK_MODE
    rules: Rules = field(default_factory=Rules)

    def decide(self, call: ToolCall, workspace: Path, cwd: Path) -> tuple[str, str]:
        if call.name == "shell":
            return judge_shell(call.input["command"], self.mode, self.rules)
        if call.name == "files":
            return judge_files(call.input, self.mode, workspace, cwd)
        if call.name == "web":
            host = urlparse(call.input["url"]).hostname or ""
            return (ALLOW, "") if self.mode == YOLO or matches(host, self.rules.web) else (ASK, "")
        return ALLOW, ""
