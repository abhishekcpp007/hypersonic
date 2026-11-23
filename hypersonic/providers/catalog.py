from __future__ import annotations

import http.client
import json
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from hypersonic.providers.base import API_KEY_ENVS, CLOUD_TAG, OPENAI_STYLE, VENDORS, api_key

TIMEOUT = 4
GIGABYTE = 1_000_000_000
TOOLS = "tools"
FREE = "0"
USER_AGENT = "hypersonic"
CLAUDE_MODELS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5")
OLLAMA_CLOUD_LIST = "https://ollama.com/api/tags"
OPENROUTER_LIST = "https://openrouter.ai/api/v1/models"
SIGN_IN = "Ollama cloud models need a one-time sign-in. Run `ollama signin` in a terminal, then try again."
NOT_PULLED = "Ollama has no model called {name}. Get it with `ollama pull {name}`, or pick one from the /model list."

Row = dict[str, str]


def row(name: str, summary: str) -> Row:
    return {"name": name, "summary": summary}


def fetch(url: str, post: bool = False) -> Any:
    """The JSON at a URL, or None when the service is down, says no, or we are offline."""
    request = urllib.request.Request(url, data=b"{}" if post else None, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return json.load(response)
    except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError):
        return None


def ollama_root() -> str:
    url, url_env, _ = OPENAI_STYLE["ollama"]
    return (os.environ.get(url_env) or url).removesuffix("/v1")


def signed_in() -> bool:
    return fetch(f"{ollama_root()}/api/me", post=True) is not None


def not_pulled(name: str) -> bool:
    """True only when Ollama answers and does not list the model. When Ollama is unreachable, nothing is assumed."""
    local = fetch(f"{ollama_root()}/api/tags")
    names = [model.get("name", "") for model in local.get("models", [])] if isinstance(local, dict) else None
    return names is not None and name not in names and f"{name}:latest" not in names


def loaded_window(name: str) -> int:
    """How many tokens Ollama really gives the loaded model. Its default is small, and it drops older messages without a word."""
    running = fetch(f"{ollama_root()}/api/ps")
    for model in running.get("models", []) if isinstance(running, dict) else []:
        if name in (model.get("name"), model.get("model")):
            return int(model.get("context_length") or 0)
    return 0


def cloud_name(name: str) -> str:
    return f"{name}-{CLOUD_TAG}" if ":" in name else f"{name}:{CLOUD_TAG}"


def with_tools(listing: Any) -> list[dict[str, Any]]:
    return [model for model in (listing or {}).get("models", []) if TOOLS in model.get("capabilities", [TOOLS])]


def added(home: Path) -> list[Row]:
    return [item for vendor in VENDORS.values() for item in vendor.models(home)]


def claude(home: Path) -> list[Row]:
    has_key = any(os.environ.get(name) for name in API_KEY_ENVS)
    return [row(f"anthropic/{name}", "Claude API") for name in CLAUDE_MODELS] if has_key else []


def ollama(home: Path) -> list[Row]:
    local = fetch(f"{ollama_root()}/api/tags")
    if local is None:
        return []
    found = [
        row(f"ollama/{model['name']}", f"on this computer, {model.get('size', 0) / GIGABYTE:.1f} GB")
        for model in with_tools(local)
    ]
    note = "Ollama cloud" if signed_in() else "Ollama cloud, run `ollama signin` first"
    hosted = sorted(model["name"] for model in with_tools(fetch(OLLAMA_CLOUD_LIST)))
    return found + [row(f"ollama/{cloud_name(name)}", note) for name in hosted]


def free_with_tools(model: dict[str, Any]) -> bool:
    pricing = model.get("pricing", {})
    costs_nothing = pricing.get("prompt") == FREE and pricing.get("completion") == FREE
    return costs_nothing and TOOLS in (model.get("supported_parameters") or [])


def openrouter(home: Path) -> list[Row]:
    listing = fetch(OPENROUTER_LIST) or {}
    note = "OpenRouter, free" if api_key("openrouter", home) else "OpenRouter, free, run `hypersonic key openrouter` first"
    return [row(f"openrouter/{name}", note) for name in sorted(m["id"] for m in listing.get("data", []) if free_with_tools(m))]


def list_models(home: Path) -> list[Row]:
    """Every model that can be picked right now, read live so no model name is guessed."""
    def safely(source: Any) -> list[Row]:
        try:
            return source(home)
        except Exception:
            return []

    with ThreadPoolExecutor() as pool:
        groups = list(pool.map(safely, (added, claude, ollama, openrouter)))
    unique: dict[str, Row] = {}
    for group in groups:
        for item in group:
            unique.setdefault(item["name"], item)
    return list(unique.values())
