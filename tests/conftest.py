from __future__ import annotations

import copy
import itertools

import pytest

from hypersonic.engine.loop import Agent
from hypersonic.engine.session import Session
from hypersonic.engine.session import home
from hypersonic.policy.rules import ASK_MODE, AUTO, Policy
from hypersonic.providers import catalog
from hypersonic.providers.base import STOP_END, STOP_TOOLS, Reply, ToolCall, Usage
from hypersonic.rag.store import Store
from hypersonic.tools import TOOLS
from hypersonic.tools.base import ToolContext

ids = itertools.count(1)


class FakeProvider:
    def __init__(self, script):
        self.script = list(script)
        self.requests = []

    async def stream(self, system, tools, messages, on_text):
        self.requests.append(copy.deepcopy(messages))
        step = self.script.pop(0)
        reply = step(messages) if callable(step) else step
        if isinstance(reply, Exception):
            raise reply
        if reply.text:
            await on_text(reply.text)
        return reply


def say(text, stop=STOP_END, usage=None):
    return Reply([{"type": "text", "text": text}], text, [], stop, usage or Usage(input=10, output=5))


def use(*calls, stop=STOP_TOOLS, usage=None):
    made = [ToolCall(f"call_{next(ids)}", name, args) for name, args in calls]
    blocks = [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in made]
    return Reply(blocks, "", made, stop, usage or Usage(input=10, cache_read=90, output=5))


@pytest.fixture(autouse=True)
def hypersonic_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERSONIC_HOME", str(tmp_path / "home"))


@pytest.fixture(autouse=True)
def no_live_services(monkeypatch):
    monkeypatch.setattr(catalog, "fetch", lambda url, post=False: None)


@pytest.fixture
def workspace(tmp_path):
    folder = tmp_path / "work"
    folder.mkdir()
    return folder.resolve()


@pytest.fixture
def store():
    return Store(home() / "index.db")


@pytest.fixture
def ctx(workspace, store):
    session = Session.create(workspace)
    return ToolContext(workspace, session.dir, store, sandbox=False)


@pytest.fixture
def make_agent(workspace, store):
    def build(script, auto=True, ask=None):
        session = Session.create(workspace)
        events = []

        async def emit(event):
            events.append(event)

        async def deny(call):
            return False

        provider = FakeProvider(script)
        tools = ToolContext(workspace, session.dir, store, sandbox=False)
        policy = Policy(AUTO if auto else ASK_MODE)
        agent = Agent(provider, session, tools, TOOLS, "system", emit, ask or deny, policy)
        return agent, events

    return build
