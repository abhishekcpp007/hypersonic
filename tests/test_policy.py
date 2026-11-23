import asyncio

import pytest
from conftest import say, use

from hypersonic.engine.loop import DENIED, run_turn
from hypersonic.policy.rules import (
    ALLOW,
    ASK,
    ASK_MODE,
    AUTO,
    DENY,
    PLAN,
    PLAN_ONLY,
    YOLO,
    Policy,
    Rules,
    judge_files,
    judge_shell,
    load_rules,
    split,
)

RULES = Rules()


def verdict(command, mode=ASK_MODE, rules=RULES):
    return judge_shell(command, mode, rules)[0]


def test_split_sees_every_command_and_file_write():
    assert split("ls -la && cat a | grep b") == ([["ls", "-la"], ["cat", "a"], ["grep", "b"]], False)
    assert split("echo $(rm -rf x) `whoami`")[0][1:] == [["rm", "-rf", "x"], ["whoami"]]
    assert split("FOO=1 git push; (cd sub && make)")[0] == [["git", "push"], ["cd", "sub"], ["make"]]
    assert split("grep x f > out.txt")[1] is True
    assert split("for f in *; do wc $f; done > all.txt")[1] is True
    assert split("grep x f 2>/dev/null 2>&1")[1] is False
    assert split("echo 'unterminated") is None
    assert split("[[ -f x ]] && echo y") is None
    assert split("  ") == ([], False)


@pytest.mark.parametrize(
    "command",
    [
        "ls -la",
        "cat a.txt | grep foo | sort | uniq -c",
        "git status && git diff HEAD~1",
        "find . -name '*.py'",
        "rg TODO src 2>/dev/null",
        "cd src && pwd",
        "X=1",
    ],
)
def test_read_only_commands_run_without_asking(command):
    assert verdict(command) == ALLOW
    assert verdict(command, PLAN) == ALLOW


@pytest.mark.parametrize(
    "command",
    [
        "touch a.txt",
        "echo hi > a.txt",
        "cat a.txt | tee b.txt",
        "find . -name '*.pyc' -delete",
        "find . -exec rm {} ;",
        "sort -o out.txt in.txt",
        "git commit -m x",
        "python script.py",
        "ls && npm install",
        "echo $(curl example.com)",
        "$CMD --flag",
        "/bin/ls",
        "sed -i s/a/b/ f",
        "echo 'unterminated",
    ],
)
def test_anything_else_asks_and_is_refused_in_plan_mode(command):
    assert verdict(command) == ASK
    assert judge_shell(command, PLAN, RULES) == (DENY, PLAN_ONLY)


def test_auto_mode_allows_all_but_flagged_and_denied():
    assert verdict("npm install && npm test", AUTO) == ALLOW
    assert verdict("echo 'unterminated", AUTO) == ASK
    for flagged in ("git push origin main", "rm -rf build", "ls && git reset --hard HEAD~3", "echo $(git clean -fd)"):
        assert verdict(flagged, AUTO) == ASK
    for banned in ("sudo rm x", "ls; sudo ls", "echo $(sudo whoami)"):
        assert judge_shell(banned, AUTO, RULES)[0] == DENY
        assert judge_shell(banned, ASK_MODE, RULES)[0] == DENY


def test_wrappers_cannot_hide_a_denied_command():
    # a denied command tucked inside a wrapper's string argument must still be denied, even in auto
    for sneaky in (
        'bash -c "sudo rm -rf /"',
        "eval 'sudo whoami'",
        'sh -c "sudo ls"',
        'ssh localhost "sudo reboot"',
        "xargs sudo rm",
        "env FOO=bar sudo rm x",
    ):
        assert judge_shell(sneaky, AUTO, RULES)[0] == DENY, sneaky
        assert judge_shell(sneaky, ASK_MODE, RULES)[0] == DENY, sneaky


def test_wrappers_are_never_auto_allowed():
    # even a harmless-looking wrapped command prompts rather than running unseen in auto mode
    for wrapped in ('bash -c "ls"', "eval 'echo hi'", 'sh -c "make build"'):
        assert judge_shell(wrapped, AUTO, RULES)[0] == ASK, wrapped


def test_reordered_rm_flags_still_prompt():
    for recursive in ("rm -rf x", "rm -r -f x", "rm -fr x", "rm --recursive x", "rm -r x"):
        assert verdict(recursive, AUTO) == ASK, recursive


def test_a_folder_in_front_of_the_program_changes_nothing():
    assert verdict("/usr/bin/sudo rm x", AUTO) == DENY
    assert verdict('/bin/bash -c "sudo rm x"', AUTO) == DENY
    for hidden in ('/bin/bash -c "make"', "/bin/rm -rf build", '$SHELL -c "make"', '$(which sh) -c "make"'):
        assert verdict(hidden, AUTO) == ASK, hidden
    assert verdict("./ls") == ASK


def test_a_denied_word_as_plain_text_is_not_a_denied_command():
    assert verdict("grep -rn sudo .") == ALLOW
    assert verdict('git commit -m "drop sudo usage"', AUTO) == ALLOW


HIDDEN = [
    "cat <<EOF\n$(touch pwned)\nEOF",
    "echo ${X:-$(touch pwned)}",
    "X=${Y:-$(touch pwned)}",
    'cat < "${X:-$(touch pwned)}"',
    "GIT_EXTERNAL_DIFF='touch pwned' git diff",
    "PATH=$PWD/bin:$PATH ls",
    "PATH=$PWD/bin:$PATH; ls",
    "sort -oout in",
    "sort --output=out in",
    "sort --outp=out in",
    "sort -no out in",
    "sort in --compress-program=./evil.sh",
    "uniq in out",
    "git diff --output=out",
    "git grep -O'touch pwned' hello",
    "file -C -m in",
    "rg --pre=./evil.sh hello tracked.txt",
    "find . $(echo -delete)",
    "a=-delete; find . $a",
    "find . -{delete,print}",
    "find . -fprint0 out",
]


@pytest.mark.parametrize("command", HIDDEN)
def test_a_command_that_only_looks_read_only_asks_and_is_refused_in_plan_mode(command):
    assert verdict(command) == ASK and verdict(command, PLAN) == DENY


def test_the_harmless_forms_of_those_commands_still_run_without_asking():
    harmless = ("X=1", "echo ${HOME:-none}", "cat <<EOF\nplain $HOME text\nEOF", "sort -n in", "uniq -c in",
                "rg 'a{2,3}' src", "git log --oneline -5", "file in", "date +%s", "find . -name '*.py'")
    assert [command for command in harmless if verdict(command) != ALLOW] == []


def test_deny_and_ask_rules_hold_in_auto_mode_however_the_command_is_spelled():
    denied = ("exec sudo ls", "builtin command sudo ls", "find . -exec sudo ls ;", "trap 'sudo ls' EXIT",
              "SUDO ls", "{sudo,ls}", "stdbuf -o0 sudo ls")
    asked = ("/usr/bin/sud? ls", "RM -rf x", "{rm,-rf,x}", "exec rm -rf x", "rm --recurs x", "git -C . push --force",
             "git reset HEAD~3 --hard", "git reset --har", "git -C . clean -fdx")
    assert [command for command in denied if verdict(command, AUTO) != DENY] == []
    assert [command for command in asked if verdict(command, AUTO) != ASK] == []
    assert verdict("rm -- old.txt", AUTO) == ALLOW


def test_protected_folders_are_protected_whatever_the_letter_case(workspace):
    (workspace / ".git" / "hooks").mkdir(parents=True)
    for path in (".GIT/hooks/pre-commit", ".Hypersonic/plugins/y.py", ".git/config"):
        assert judge_files({"command": "create", "path": path}, AUTO, workspace, workspace)[0] == DENY
    assert judge_files({"command": "create", "path": "git/notes.txt"}, AUTO, workspace, workspace)[0] == ALLOW


def test_a_string_where_a_list_of_patterns_belongs_is_refused(tmp_path):
    config = tmp_path / "config.toml"
    for text in ('[rules]\nallow = "git commit*"\n', '[web]\nallow = "*.readthedocs.io"\n', "rules = 5\n"):
        config.write_text(text)
        with pytest.raises(ValueError, match="must be a list"):
            load_rules(config)


def test_yolo_mode_allows_everything():
    assert verdict("sudo rm -rf /", YOLO) == ALLOW
    assert verdict('bash -c "sudo rm -rf /"', YOLO) == ALLOW


def test_user_rules_extend_the_defaults(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('[rules]\nallow = ["git commit*", "npm test"]\nask = ["docker *"]\ndeny = ["curl *"]\n')
    rules = load_rules(config)
    assert verdict("git commit -m 'x'", rules=rules) == ALLOW
    assert verdict("ls && npm test", rules=rules) == ALLOW
    assert verdict("npm test > log.txt", rules=rules) == ASK
    assert verdict("docker ps", AUTO, rules) == ASK
    assert verdict("curl example.com", AUTO, rules) == DENY
    assert verdict("git push", AUTO, rules) == ASK
    assert verdict("sudo ls", AUTO, rules) == DENY
    assert load_rules(tmp_path / "missing.toml") == Rules()


def test_file_writes_follow_the_mode(workspace):
    def judge(command, path, mode):
        return judge_files({"command": command, "path": path}, mode, workspace, workspace)[0]

    assert [judge("view", "a.txt", mode) for mode in (PLAN, ASK_MODE, AUTO, YOLO)] == [ALLOW] * 4
    assert [judge("create", "a.txt", mode) for mode in (PLAN, ASK_MODE, AUTO, YOLO)] == [DENY, ASK, ALLOW, ALLOW]
    for protected in (".git/hooks/pre-commit", ".git/config", ".hypersonic/config.toml", "sub/../.git/hooks/x"):
        assert judge("create", protected, AUTO) == DENY
        assert judge("str_replace", str(workspace / protected), ASK_MODE) == DENY
    assert judge("create", ".git/hooks/pre-commit", YOLO) == ALLOW
    assert judge("create", ".github/workflows/ci.yml", AUTO) == ALLOW


def test_denied_by_policy_never_reaches_the_user_or_the_tool(make_agent, workspace):
    async def explode(call):
        raise AssertionError("asked about a denied call")

    script = [
        use(("shell", {"command": "sudo touch made.txt"}), ("files", {"command": "create", "path": ".git/hooks/pre-commit", "file_text": "x"})),
        say("ok"),
    ]
    agent, _ = make_agent(script, auto=False, ask=explode)
    asyncio.run(run_turn(agent, "go"))
    shell_result, files_result = agent.session.messages[2]["content"]
    assert shell_result["content"] == "Blocked by a deny rule: sudo touch made.txt"
    assert files_result["content"] == ".git/hooks/pre-commit is protected and can't be changed by the agent."
    assert shell_result["is_error"] and files_result["is_error"]
    assert not (workspace / "made.txt").exists()


def test_plan_mode_reads_but_never_writes(make_agent, workspace):
    (workspace / "a.txt").write_text("content")
    script = [
        use(("shell", {"command": "cat a.txt"}), ("shell", {"command": "rm a.txt"}), ("files", {"command": "create", "path": "b.txt", "file_text": "x"})),
        say("ok"),
    ]
    agent, _ = make_agent(script)
    agent.policy = Policy(PLAN)
    asyncio.run(run_turn(agent, "go"))
    read, removed, created = agent.session.messages[2]["content"]
    assert read["content"] == "content"
    assert removed["content"] == created["content"] == PLAN_ONLY
    assert (workspace / "a.txt").exists() and not (workspace / "b.txt").exists()


def test_ask_mode_prompts_only_for_risky_calls(make_agent, workspace):
    asked = []

    async def refuse(call):
        asked.append(call.input["command"])
        return False

    agent, _ = make_agent([use(("shell", {"command": "ls"}), ("shell", {"command": "touch x"})), say("ok")], auto=False, ask=refuse)
    asyncio.run(run_turn(agent, "go"))
    assert asked == ["touch x"]
    assert agent.session.messages[2]["content"][1]["content"] == DENIED
