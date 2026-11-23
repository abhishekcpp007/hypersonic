import asyncio

from conftest import say, use

from hypersonic.engine.events import SESSION_START, Hooks, load_plugins
from hypersonic.engine.loop import run_turn
from hypersonic.engine.session import harness_note, transcript
from hypersonic.plugins import budgets, done_gate, loop_guard

PLUGIN = '''
from hypersonic.engine.events import PRE_TOOL, SESSION_START

def register(hooks):
    async def greet(agent):
        agent.session.queue.append("plugin says hello")

    async def no_curl(agent, call):
        if call.name == "shell" and "curl" in call.input["command"]:
            return "This project forbids curl."

    hooks.on(SESSION_START, greet)
    hooks.on(PRE_TOOL, no_curl)
'''


def turn(agent, text="do it"):
    return asyncio.run(run_turn(agent, text))


def results(agent, index):
    return agent.session.messages[index]["content"]


def test_loop_guard_nudges_after_four_identical_calls(make_agent):
    same = ("shell", {"command": "echo stuck"})
    agent, _ = make_agent([use(same)] * loop_guard.REPEAT_LIMIT + [use(("shell", {"command": "echo new"})), say("ok")])
    loop_guard.register(agent.hooks)
    turn(agent)
    fourth = results(agent, 2 * loop_guard.REPEAT_LIMIT)
    assert fourth[-1] == {"type": "text", "text": harness_note(loop_guard.NUDGE)}
    assert all(len(results(agent, index)) == 1 for index in (2, 4, 6, 10))


def test_budget_warns_then_blocks_tools(make_agent, workspace):
    step = ("shell", {"command": "echo tick >> ticks.txt"})
    agent, _ = make_agent([use(step)] * 6 + [say("summary")])
    budgets.register(agent.hooks, budgets.Budget(steps=5))
    assert turn(agent) == "summary"
    assert (workspace / "ticks.txt").read_text().count("tick") == 4
    assert results(agent, 8)[-1] == {"type": "text", "text": harness_note(budgets.WARNING)}
    assert results(agent, 10)[0]["content"] == budgets.OVER
    assert results(agent, 12)[0]["content"] == budgets.OVER
    assert sum(budgets.WARNING in str(message) for message in agent.session.messages) == 1


def test_a_model_that_ignores_a_spent_budget_is_stopped(make_agent, workspace):
    step = ("shell", {"command": "echo tick >> ticks.txt"})
    agent, events = make_agent([use(step)] * 30)
    budgets.register(agent.hooks, budgets.Budget(steps=2))
    turn(agent)
    assert events[-1]["stop"] == "error" and budgets.SPENT in events[-2]["message"]
    assert agent.session.steps == 2 + budgets.BLOCK_LIMIT
    assert all(entry["role"] != "user" or "budget" not in entry.get("text", "") for entry in transcript(agent.session.messages))


def test_the_check_runs_from_the_workspace_root_and_leaves_the_agent_where_it_was(make_agent, workspace):
    (workspace / "sub").mkdir()
    (workspace / "marker.txt").write_text("here")
    agent, _ = make_agent([use(("shell", {"command": "cd sub && touch made.txt"})), say("done")])
    done_gate.register(agent.hooks)
    agent.check = "test -f marker.txt && cd / && true"
    assert turn(agent) == "done"
    assert agent.ctx.cwd == workspace / "sub" and len(agent.session.messages) == 4


def test_no_budget_means_no_hooks(make_agent):
    hooks = Hooks()
    budgets.register(hooks, budgets.Budget())
    assert not any(hooks.handlers.values())


def test_done_gate_sends_failures_back_until_checks_pass(make_agent, workspace):
    edit = ("files", {"command": "create", "path": "ok.txt", "file_text": "fixed"})
    script = [use(("files", {"command": "create", "path": "a.py", "file_text": "x = 1"})), say("done"), use(edit), say("really done")]
    agent, _ = make_agent(script)
    done_gate.register(agent.hooks)
    agent.check = "test -f ok.txt || { echo 'ok.txt is missing'; exit 1; }"
    assert turn(agent) == "really done"
    feedback = results(agent, 4)[0]["text"]
    assert feedback.startswith(harness_note("The check command `test -f ok.txt")[:40])
    assert "ok.txt is missing\n[exit code 1]" in feedback
    assert len(agent.session.messages) == 8


def test_done_gate_gives_up_after_two_rounds_and_skips_read_only_turns(make_agent):
    edit = ("shell", {"command": "touch a.txt"})
    agent, _ = make_agent([use(edit), say("one"), use(edit), say("two"), use(edit), say("three")])
    done_gate.register(agent.hooks)
    agent.check = "false"
    assert turn(agent) == "three"
    assert sum("The check command" in str(message) for message in agent.session.messages) == done_gate.DONE_ROUNDS

    agent, _ = make_agent([use(("shell", {"command": "ls"})), say("just looked")])
    done_gate.register(agent.hooks)
    agent.check = "false"
    assert turn(agent) == "just looked"
    assert len(agent.session.messages) == 4

    agent, _ = make_agent([use(edit), say("no check set")])
    done_gate.register(agent.hooks)
    assert turn(agent) == "no check set"
    assert len(agent.session.messages) == 4


def test_user_plugins_load_and_can_veto_tools(make_agent, tmp_path, workspace):
    folder = tmp_path / "plugins"
    folder.mkdir()
    (folder / "house_rules.py").write_text(PLUGIN)
    agent, _ = make_agent([use(("shell", {"command": "curl example.com"}), ("shell", {"command": "echo fine"})), say("ok")])
    assert load_plugins(agent.hooks, [folder, tmp_path / "missing"]) == ["house_rules"]
    asyncio.run(agent.hooks.fire(SESSION_START, agent))
    turn(agent)
    vetoed, fine, queued = results(agent, 2)
    assert vetoed["content"] == "This project forbids curl." and vetoed["is_error"]
    assert fine["content"] == "fine"
    assert queued == {"type": "text", "text": "plugin says hello"}
