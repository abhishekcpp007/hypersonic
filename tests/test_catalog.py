import stat

import pytest

from hypersonic import cli
from hypersonic.engine.session import home
from hypersonic.providers import base, catalog
from hypersonic.engine.events import register_files
from hypersonic.providers.base import KEYS_NAME, PROVIDER_FOLDER, api_key, make_provider, save_key

LOCAL = {"models": [
    {"name": "qwen3.5:2b", "size": 3_100_000_000, "capabilities": ["completion", "tools"]},
    {"name": "embedder:1b", "size": 500_000_000, "capabilities": ["embedding"]},
]}
CLOUD = {"models": [
    {"name": "nemotron-3-ultra", "capabilities": ["completion", "tools"]},
    {"name": "deepseek-v4-pro:0813", "capabilities": ["completion", "tools"]},
    {"name": "speech-only", "capabilities": ["completion"]},
]}
ROUTER = {"data": [
    {"id": "nvidia/nemotron:free", "pricing": {"prompt": "0", "completion": "0"}, "supported_parameters": ["tools"]},
    {"id": "paid/model", "pricing": {"prompt": "0.000001", "completion": "0"}, "supported_parameters": ["tools"]},
    {"id": "free/no-tools:free", "pricing": {"prompt": "0", "completion": "0"}, "supported_parameters": []},
]}


@pytest.fixture
def services(monkeypatch):
    answers = {
        "http://localhost:11434/api/tags": LOCAL,
        catalog.OLLAMA_CLOUD_LIST: CLOUD,
        catalog.OPENROUTER_LIST: ROUTER,
    }
    monkeypatch.setattr(catalog, "fetch", lambda url, post=False: answers.get(url))
    for name in ("OLLAMA_URL", "OPENROUTER_API_KEY", *base.API_KEY_ENVS):
        monkeypatch.delenv(name, raising=False)
    return answers


def names(rows):
    return [item["name"] for item in rows]


def test_cloud_models_get_the_name_the_local_ollama_expects():
    assert catalog.cloud_name("nemotron-3-ultra") == "nemotron-3-ultra:cloud"
    assert catalog.cloud_name("gpt-oss:120b") == "gpt-oss:120b-cloud"


def test_list_holds_only_models_that_can_call_tools(services, tmp_path):
    rows = catalog.list_models(tmp_path)
    assert names(rows) == [
        "ollama/qwen3.5:2b",
        "ollama/deepseek-v4-pro:0813-cloud",
        "ollama/nemotron-3-ultra:cloud",
        "openrouter/nvidia/nemotron:free",
    ]
    assert rows[0]["summary"] == "on this computer, 3.1 GB"


def test_each_row_says_what_is_still_missing(services, tmp_path):
    notes = {item["name"]: item["summary"] for item in catalog.list_models(tmp_path)}
    assert "ollama signin" in notes["ollama/nemotron-3-ultra:cloud"]
    assert "hypersonic key openrouter" in notes["openrouter/nvidia/nemotron:free"]
    services["http://localhost:11434/api/me"] = {"name": "someone"}
    save_key("openrouter", "sk-test", tmp_path)
    notes = {item["name"]: item["summary"] for item in catalog.list_models(tmp_path)}
    assert notes["ollama/nemotron-3-ultra:cloud"] == "Ollama cloud"
    assert notes["openrouter/nvidia/nemotron:free"] == "OpenRouter, free"


def test_a_source_that_is_down_is_left_out(services, tmp_path):
    del services["http://localhost:11434/api/tags"]
    del services[catalog.OPENROUTER_LIST]
    assert catalog.list_models(tmp_path) == []


def test_saved_key_is_private_and_used(services, tmp_path):
    with pytest.raises(ValueError, match="hypersonic key openrouter"):
        make_provider("openrouter/nvidia/nemotron:free", "medium", tmp_path)
    save_key("openrouter", "sk-test", tmp_path)
    save_key("openai", "sk-other", tmp_path)
    assert stat.S_IMODE((tmp_path / KEYS_NAME).stat().st_mode) == base.OWNER_ONLY
    provider = make_provider("openrouter/nvidia/nemotron:free", "medium", tmp_path)
    assert provider.model == "nvidia/nemotron:free"
    assert provider.client.api_key == "sk-test"
    assert str(provider.client.base_url).startswith("https://openrouter.ai/api/v1")


def test_cloud_model_needs_the_ollama_sign_in(services, tmp_path):
    with pytest.raises(ValueError, match="ollama signin"):
        make_provider("ollama/nemotron-3-ultra:cloud", "medium", tmp_path)
    assert make_provider("ollama/qwen3.5:2b", "medium", tmp_path).model == "qwen3.5:2b"
    services["http://localhost:11434/api/me"] = {"name": "someone"}
    assert make_provider("ollama/nemotron-3-ultra:cloud", "medium", tmp_path).model == "nemotron-3-ultra:cloud"


def test_a_local_model_that_was_never_pulled_is_refused_up_front(services, tmp_path):
    with pytest.raises(ValueError, match="ollama pull typo:1b"):
        make_provider("ollama/typo:1b", "medium", home=tmp_path)
    assert make_provider("ollama/qwen3.5:2b", "medium", home=tmp_path).model == "qwen3.5:2b"
    services.pop("http://localhost:11434/api/tags")
    assert make_provider("ollama/typo:1b", "medium", home=tmp_path).model == "typo:1b"


def test_both_cloud_name_endings_need_the_sign_in(services, tmp_path):
    for name in ("ollama/nemotron-3-ultra:cloud", "ollama/deepseek-v4-pro:0813-cloud"):
        with pytest.raises(ValueError, match="ollama signin"):
            make_provider(name, "medium", home=tmp_path)


def test_the_loaded_window_comes_from_ollama(services):
    assert catalog.loaded_window("qwen3.5:2b") == 0
    services["http://localhost:11434/api/ps"] = {"models": [{"name": "qwen3.5:2b", "model": "qwen3.5:2b", "context_length": 4096}]}
    assert catalog.loaded_window("qwen3.5:2b") == 4096 and catalog.loaded_window("other") == 0


def test_one_broken_source_does_not_empty_the_list(services, tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "openrouter", lambda home: 1 / 0)
    assert "ollama/qwen3.5:2b" in names(catalog.list_models(tmp_path))


def test_a_damaged_keys_file_is_treated_as_empty(services, tmp_path):
    (tmp_path / KEYS_NAME).write_text('"just a string"')
    assert api_key("openrouter", tmp_path) == ""
    save_key("openrouter", "sk-test", tmp_path)
    assert api_key("openrouter", tmp_path) == "sk-test"
    assert [entry.name for entry in tmp_path.iterdir()] == [KEYS_NAME]


def test_a_provider_file_in_the_home_folder_adds_its_own_models(services, tmp_path, monkeypatch):
    (tmp_path / PROVIDER_FOLDER).mkdir()
    (tmp_path / PROVIDER_FOLDER / "mine.py").write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "from hypersonic.providers.base import Vendor\n"
        "@dataclass\n"
        "class Settings:\n"
        "    url: str = 'http://localhost:1'\n"
        "def register(vendors):\n"
        "    vendors['mine'] = Vendor(lambda name, effort, home: ('made', name, effort),\n"
        "                             lambda home: [{'name': 'mine/fast', 'summary': 'my own server'}], lambda: 'mine/fast')\n"
    )
    (tmp_path / PROVIDER_FOLDER / "broken.py").write_text("raise RuntimeError('boom')\n")
    monkeypatch.setitem(base.VENDORS, "mine", None)
    loaded, broken = register_files(base.VENDORS, [tmp_path / PROVIDER_FOLDER], "Provider")
    assert loaded == ["mine"] and broken == ["Provider broken.py was skipped. RuntimeError: boom"]
    assert names(catalog.list_models(tmp_path))[0] == "mine/fast"
    assert make_provider("mine/fast", "medium", tmp_path) == ("made", "fast", "medium")
    assert base.default_model() == "mine/fast" and base.default_model("ollama/qwen3.5:2b") == "ollama/qwen3.5:2b"
    with pytest.raises(ValueError, match="mine/"):
        make_provider("nowhere/x", "medium", tmp_path)


def test_unknown_provider_points_to_the_picker(tmp_path):
    with pytest.raises(ValueError, match="pick from the list.*ollama/.*openrouter/"):
        make_provider("qwen3.5:2b", "medium", tmp_path)


def test_key_command_saves_what_the_user_types(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt: " sk-typed ")
    assert cli.main(["key", "openrouter"]) == 0
    assert api_key("openrouter", home()) == "sk-typed"
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt: "")
    assert cli.main(["key", "openrouter"]) == 1
    assert api_key("openrouter", home()) == "sk-typed"
    with pytest.raises(SystemExit):
        cli.main(["key", "ollama"])
