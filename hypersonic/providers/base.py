from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

Message = dict[str, Any]
OnText = Callable[[str], Awaitable[None]]

STOP_END = "end"
STOP_TOOLS = "tools"
STOP_LENGTH = "length"
STOP_PAUSE = "pause"
STOP_REFUSAL = "refusal"
API_MODEL = "anthropic/claude-opus-5-5"
PROVIDER_FOLDER = "providers"
MODEL_ENV = "HYPERSONIC_MODEL"
API_KEY_ENVS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
KEYS_NAME = "keys.json"
PICKED_NAME = "model"
OWNER_ONLY = 0o600
CLOUD_TAG = "cloud"
CLOUD_ENDINGS = (f":{CLOUD_TAG}", f"-{CLOUD_TAG}")
# vendor: (default URL, env var that overrides the URL, env var holding the key or "" when none is needed)
OPENAI_STYLE = {
    "ollama": ("http://localhost:11434/v1", "OLLAMA_URL", ""),
    "openai": ("https://api.openai.com/v1", "OPENAI_BASE_URL", "OPENAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_BASE_URL", "OPENROUTER_API_KEY"),
}
NO_KEY = "{vendor} needs an API key. Run `hypersonic key {vendor}` in a terminal and paste it there."
NO_PROVIDER = "No provider for {model!r}. Type /model and a space to pick from the list, or start the name with one of: {vendors}."


@dataclass
class Usage:
    input: int = 0
    cache_read: int = 0
    cache_write: int = 0
    output: int = 0

    @property
    def prompt(self) -> int:
        return self.input + self.cache_read + self.cache_write


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class Reply:
    content: list[dict[str, Any]]
    text: str
    tool_calls: list[ToolCall]
    stop: str
    usage: Usage = field(default_factory=Usage)
    model: str = ""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    schema: dict[str, Any]


def text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


class Provider(Protocol):
    async def stream(
        self, system: str, tools: list[ToolSpec], messages: list[Message], on_text: OnText
    ) -> Reply: ...


@dataclass
class Vendor:
    """A model provider the user added with a file in ~/.hypersonic/providers.

    The file defines register(vendors) and puts one of these in the table under the name used before the slash.
    """

    make: Callable[[str, str, Path], Provider]
    models: Callable[[Path], list[dict[str, str]]] = lambda home: []
    default: Callable[[], str] = lambda: ""


VENDORS: dict[str, Vendor] = {}


def default_model(configured: str = "") -> str:
    chosen = os.environ.get(MODEL_ENV) or configured
    if chosen:
        return chosen
    if any(os.environ.get(name) for name in API_KEY_ENVS):
        return API_MODEL
    return next((pick for vendor in VENDORS.values() if (pick := vendor.default())), API_MODEL)


def picked_model(home: Path) -> str:
    """The model the user last switched to, kept as the default for new sessions."""
    try:
        return (home / PICKED_NAME).read_text().strip()
    except OSError:
        return ""


def write_private(path: Path, text: str) -> None:
    """Replace a file in one step, readable by its owner only, so a crash or a second writer never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with os.fdopen(os.open(fresh, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, OWNER_ONLY), "w") as out:
        out.write(text)
    os.replace(fresh, path)


def read_table(path: Path) -> dict[str, Any]:
    """A small JSON settings file as a dict, or an empty one when it is missing or damaged."""
    try:
        found = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return found if isinstance(found, dict) else {}


def keep_model(model: str, home: Path) -> None:
    write_private(home / PICKED_NAME, model)


def api_key(vendor: str, home: Path) -> str:
    return os.environ.get(OPENAI_STYLE[vendor][2]) or str(read_table(home / KEYS_NAME).get(vendor, ""))


def save_key(vendor: str, secret: str, home: Path) -> None:
    path = home / KEYS_NAME
    write_private(path, json.dumps({**read_table(path), vendor: secret}))


def make_provider(model: str, effort: str, home: Path | None = None) -> Provider:
    home = home or Path.home() / ".hypersonic"
    vendor, slash, name = model.partition("/")
    if vendor == "anthropic" and name:
        from hypersonic.providers.claude import ClaudeProvider

        return ClaudeProvider(name, effort)
    if vendor in VENDORS and slash:
        return VENDORS[vendor].make(name, effort, home)
    if vendor in OPENAI_STYLE and name:
        from hypersonic.providers import catalog
        from hypersonic.providers.openai_compat import OpenAICompatProvider

        url, url_env, key_env = OPENAI_STYLE[vendor]
        key = api_key(vendor, home)
        if key_env and not key:
            raise ValueError(NO_KEY.format(vendor=vendor))
        hosted = name.endswith(CLOUD_ENDINGS)
        if vendor == "ollama" and hosted and not catalog.signed_in():
            raise ValueError(catalog.SIGN_IN)
        if vendor == "ollama" and not hosted and catalog.not_pulled(name):
            raise ValueError(catalog.NOT_PULLED.format(name=name))
        window = (lambda: catalog.loaded_window(name)) if vendor == "ollama" else None
        return OpenAICompatProvider(name, os.environ.get(url_env) or url, key or vendor, window=window)
    vendors = ", ".join(f"{known}/" for known in ("anthropic", *VENDORS, *OPENAI_STYLE))
    raise ValueError(NO_PROVIDER.format(model=model, vendors=vendors))
