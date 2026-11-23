import asyncio

import pytest
from conftest import say, use

from hypersonic.context.compact import COMPACT_AT
from hypersonic.context.prefix import INSTRUCTION_TOKENS, PROMPTS, system_prompt
from hypersonic.context.squeeze import CHARS_PER_TOKEN
from hypersonic.engine.commands import CommandError, command_list, run_command
from hypersonic.engine.loop import run_turn
from hypersonic.engine.session import Session, home, transcript
from hypersonic.providers.base import Usage, picked_model
from hypersonic.rag.retrieve import retrieve

BIG = Usage(input=COMPACT_AT, output=10)


def command(agent, line):
    return asyncio.run(run_command(agent, line))


def turn(agent, text="do it"):
    return asyncio.run(run_turn(agent, text))


def test_one_prompt_carries_the_code_rules_and_there_are_no_modes(make_agent):
    prompt = system_prompt()
    assert "find it with rg" in prompt and "only when they type /rag" in prompt and "Keep diffs small." in prompt
    assert "run the project's tests or checks" in prompt and "mode" not in prompt.lower()
    assert sorted(path.name for path in PROMPTS.iterdir()) == ["base.md", "compact.md"]
    agent, _ = make_agent([])
    for gone in ("/code", "/general"):
        with pytest.raises(CommandError, match="Unknown command"):
            command(agent, gone)
    assert not {"code", "general"} & {item["name"] for item in command_list(agent)}


def test_old_session_logs_with_mode_records_still_load(workspace):
    session = Session.create(workspace)
    session.append({"role": "user", "content": []})
    session.write("mode", mode="code")
    session.append({"role": "assistant", "content": []})
    assert len(Session.load(session.id).messages) == 2


def test_index_undo_and_help(make_agent, workspace):
    agent, _ = make_agent([use(("files", {"command": "create", "path": "a.txt", "file_text": "new"})), say("ok")])
    (workspace / "a.txt").write_text("original")
    (workspace / "notes").mkdir()
    (workspace / "notes" / "pets.md").write_text("The walrus eats clams.")
    assert command(agent, "/index notes").notice.endswith("1 indexed, 0 unchanged, 0 removed.")
    assert retrieve(agent.ctx.store, "walrus", agent.ctx.scopes).hits
    turn(agent)
    assert (workspace / "a.txt").read_text() == "new"
    assert command(agent, "/undo").notice == f"Restored {workspace / 'a.txt'}."
    assert (workspace / "a.txt").read_text() == "original"
    assert command(agent, "/undo").notice == "Nothing to undo."
    listed = command(agent, "/help").notice
    assert all(f"/{name}" in listed for name in ("model", "check", "compact", "index", "undo", "remember", "help", "new"))
    with pytest.raises(CommandError, match="Unknown command /nope"):
        command(agent, "/nope")
    with pytest.raises(CommandError, match="doesn't exist"):
        command(agent, "/index missing")


def test_memory_is_shared_across_sessions(make_agent):
    writer, _ = make_agent([])
    assert command(writer, "/remember The deploy server is called marmalade-7.").notice == "Saved to memory."
    reader, _ = make_agent([say("no idea"), say("marmalade-7")])
    turn(reader, "which deploy server name do we use")
    assert "marmalade-7" not in str(reader.session.messages[0])
    turn(reader, command(reader, "/rag which deploy server name do we use").prompt)
    attached = reader.session.messages[2]["content"][-1]["text"]
    assert attached.startswith("<context>\n[memory] ") and "marmalade-7" in attached


def test_custom_commands_need_trust_inside_a_project(make_agent, workspace):
    shared = home() / "commands"
    shared.mkdir(parents=True)
    (shared / "review.md").write_text("Review $ARGUMENTS for bugs.")
    local = workspace / ".hypersonic" / "commands"
    local.mkdir(parents=True)
    (local / "ship.md").write_text("Run the release checklist.")
    agent, _ = make_agent([])
    assert command(agent, "/review src/app.py").prompt == "Review src/app.py for bugs."
    with pytest.raises(CommandError, match="Unknown command /ship"):
        command(agent, "/ship")
    agent.trusted = True
    assert command(agent, "/ship now").prompt == "Run the release checklist.\nnow"
    assert "/ship  custom command" in command(agent, "/help").notice


def test_project_instructions_load_only_when_trusted(make_agent, workspace):
    (workspace / "AGENTS.md").write_text("Always answer in French.\n" + "filler line\n" * 2000)
    agent, _ = make_agent([say("ok")])
    turn(agent, "hi")
    assert len(agent.session.messages[0]["content"]) == 2

    agent, _ = make_agent([say("ok")])
    agent.trusted = True
    turn(agent, "hi")
    env, rules, text = agent.session.messages[0]["content"]
    assert rules["text"].startswith('<instructions source="AGENTS.md">\nAlways answer in French.')
    assert "Read the rest of AGENTS.md at " in rules["text"]
    assert len(rules["text"]) <= INSTRUCTION_TOKENS * CHARS_PER_TOKEN + 300
    assert text == {"type": "text", "text": "hi"}


def test_compact_command(make_agent):
    agent, events = make_agent([say("The code word is marmalade."), say("Goal: keep the code word.")])
    assert command(agent, "/compact").notice == "Nothing to compact yet."
    turn(agent, "remember this")
    assert command(agent, "/compact").notice == "Compacted. The earlier messages are saved in the session folder."
    assert len(agent.session.messages) == 1
    assert "Goal: keep the code word." in agent.session.messages[0]["content"][1]["text"]


def test_model_command_switches_the_provider(make_agent):
    agent, _ = make_agent([])
    assert command(agent, "/model").notice.startswith("Model:")
    assert "ollama/qwen3.5:2b" in command(agent, "/model ollama/qwen3.5:2b").notice
    assert agent.provider.model == "qwen3.5:2b"
    assert agent.model == "ollama/qwen3.5:2b"
    assert command(agent, "/model").notice.startswith("Model: ollama/qwen3.5:2b.")
    assert picked_model(home()) == "ollama/qwen3.5:2b"
    with pytest.raises(CommandError, match="NOT switched to qwen3.5:2b.*pick from the list.*Still using ollama/qwen3.5:2b"):
        command(agent, "/model qwen3.5:2b")
    assert agent.model == "ollama/qwen3.5:2b"


def test_a_switched_model_is_told_its_own_name(make_agent):
    agent, events = make_agent([say("hi"), say("still here")])
    scripted = agent.provider
    command(agent, "/model ollama/qwen3.5:2b")
    command(agent, "/model ollama/qwen3.5:4b")
    agent.provider = scripted
    asyncio.run(run_turn(agent, "which model are you?"))
    asked = agent.session.messages[0]["content"]
    assert asked[-1]["text"].startswith("<harness>") and "say ollama/qwen3.5:4b" in asked[-1]["text"]
    assert "qwen3.5:2b" not in str(asked)
    asyncio.run(run_turn(agent, "and now?"))
    assert agent.session.messages[2]["content"] == [{"type": "text", "text": "and now?"}]
    assert [m["role"] for m in agent.session.messages] == ["user", "assistant", "user", "assistant"]
    assert [row["text"] for row in transcript(agent.session.messages) if row["role"] == "user"] == ["which model are you?", "and now?"]
    assert not [event for event in events if event["type"] == "error"]


def test_mode_command_changes_permission_mode(make_agent):
    agent, _ = make_agent([], auto=False)
    assert "Permission mode: ask" in command(agent, "/mode").notice
    command(agent, "/mode auto")
    assert agent.policy.mode == "auto" and agent.ctx.sandbox is True
    command(agent, "/mode yolo")
    assert agent.policy.mode == "yolo" and agent.ctx.sandbox is False


def test_the_agent_is_told_when_the_permission_mode_changes(make_agent, workspace):
    write = use(("files", {"command": "create", "path": "note.txt", "file_text": "x"}))
    agent, _ = make_agent([write, say("plan mode blocked it"), write, say("written")], auto=False)
    command(agent, "/mode plan")
    turn(agent, "create note.txt")
    assert not (workspace / "note.txt").exists()
    command(agent, "/mode plan")
    command(agent, "/mode auto")
    turn(agent, "create note.txt")
    assert (workspace / "note.txt").read_text() == "x"
    told = [message["content"][-1]["text"] for message in agent.session.messages if message["role"] == "user" and "<harness>" in str(message["content"][-1])]
    assert len(told) == 2 and "now plan" in told[0] and "now auto" in told[1]
    assert not [message for message in agent.session.messages if message["role"] == "system"]
