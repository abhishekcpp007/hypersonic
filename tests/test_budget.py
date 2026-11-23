import json
from pathlib import Path

import hypersonic
from hypersonic.context.prefix import env_block, system_prompt
from hypersonic.context.squeeze import CHARS_PER_TOKEN
from hypersonic.tools import TOOLS

SYSTEM_TOKENS = 410
TOOL_TOKENS = 500
ENV_TOKENS = 80
CORE_LINES = 4000
# the core is the harness itself: the terminal and desktop clients sit beside it, like the web page
NOT_CORE = ("tui.py", "app.py")
PACKAGE = Path(hypersonic.__file__).parent


def tokens(text: str) -> float:
    return len(text) / CHARS_PER_TOKEN


def test_system_prompt_stays_in_budget():
    assert tokens(system_prompt()) <= SYSTEM_TOKENS


def test_system_prompt_has_nothing_that_changes(tmp_path):
    assert system_prompt() == system_prompt()
    assert str(tmp_path) not in system_prompt()
    assert str(Path.home()) not in system_prompt()


def test_tool_schemas_stay_in_budget():
    specs = [
        {"name": tool.spec.name, "description": tool.spec.description, "input_schema": tool.spec.schema}
        for tool in TOOLS.values()
    ]
    assert tokens(json.dumps(specs)) <= TOOL_TOKENS


def test_env_block_is_small(tmp_path):
    assert tokens(env_block(tmp_path)) <= ENV_TOKENS


def test_core_stays_small():
    lines = sum(
        1 for path in PACKAGE.rglob("*.py") if path.name not in NOT_CORE
        for line in path.read_text().splitlines() if line.strip()
    )
    assert lines <= CORE_LINES
