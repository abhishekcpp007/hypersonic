from __future__ import annotations

import asyncio
import http.client
import ipaddress
import socket
import urllib.error
import urllib.request
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse

from hypersonic.context.squeeze import CHARS_PER_TOKEN, cap, clean
from hypersonic.providers.base import ToolSpec
from hypersonic.rag.chunk import chunk
from hypersonic.rag.tokens import tokenize
from hypersonic.tools.base import Tool, ToolContext, ToolError

WEB_TOKENS = 1500
MAX_BYTES = 2_000_000
TIMEOUT = 20
USER_AGENT = "hypersonic"
MOVED = "The page redirects to another site: {url}. Fetch that address if you want it, so the user can approve the new site."
SCHEMES = ("http", "https")
NAT64 = ipaddress.ip_network("64:ff9b::/96")
IPV4_BYTES = 4
SKIPPED = frozenset({"script", "style", "noscript", "svg", "template"})
BREAKS = frozenset("p div br li tr section article pre table ul ol blockquote header footer main".split())
HEADINGS = {f"h{level}": "#" * level + " " for level in range(1, 7)}

SPEC = ToolSpec(
    "web",
    "Fetch a web page as text. With query, only matching passages come back. Long pages are "
    "cut and saved to a file. Each new site needs the user's approval.",
    {
        "type": "object",
        "properties": {"url": {"type": "string"}, "query": {"type": "string"}},
        "required": ["url"],
    },
)


class Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skipping = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in SKIPPED:
            self.skipping += 1
        elif tag in HEADINGS:
            self.parts.append("\n\n" + HEADINGS[tag])
        elif tag in BREAKS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIPPED:
            self.skipping = max(0, self.skipping - 1)
        elif tag in HEADINGS or tag in BREAKS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skipping:
            self.parts.append(data)


def to_text(body: str, kind: str) -> str:
    if "html" not in kind:
        return body
    parser = Text()
    parser.feed(body)
    return "".join(parser.parts)


def public(address: str) -> bool:
    parsed = ipaddress.ip_address(address)
    if parsed in NAT64:
        parsed = ipaddress.ip_address(parsed.packed[-IPV4_BYTES:])
    return parsed.is_global


def check_address(url: str, local_ok: bool) -> list[str]:
    """Refuses a URL that may not be fetched. Returns the addresses to dial, or none when any will do."""
    parsed = urlparse(url)
    if parsed.scheme not in SCHEMES or not parsed.hostname:
        raise ToolError("Only http and https URLs can be fetched.")
    if local_ok:
        return []
    try:
        found = socket.getaddrinfo(parsed.hostname, parsed.port, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise ToolError(f"Can't resolve {parsed.hostname}.")
    addresses = [item[4][0] for item in found]
    if not all(public(address) for address in addresses):
        raise ToolError(f"{parsed.hostname} points to a private or local address, which is blocked.")
    return addresses


def proxied(url: str) -> bool:
    """True when this address goes through a proxy. The proxy looks the name up itself, so there is nothing to pin."""
    parts = urlparse(url)
    return parts.scheme in urllib.request.getproxies() and not urllib.request.proxy_bypass(parts.hostname or "")


def pinned(base: Any, addresses: list[str]) -> Any:
    """A connection class that dials only the addresses already checked.

    A second DNS lookup at connect time could return a different, private address. TLS still
    verifies the host name. This relies on http.client keeping its dialer in _create_connection.
    """
    if not addresses:
        return base

    def dial(target: tuple[str, int], *rest: Any, **more: Any) -> socket.socket:
        for address in addresses[:-1]:
            try:
                return socket.create_connection((address, target[1]), *rest, **more)
            except OSError:
                continue
        return socket.create_connection((addresses[-1], target[1]), *rest, **more)

    class Pinned(base):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._create_connection = dial

    return Pinned


class PinnedHTTP(urllib.request.HTTPHandler):
    def __init__(self, local_ok: bool) -> None:
        super().__init__()
        self.local_ok = local_ok

    def http_open(self, req: Any) -> Any:
        if proxied(req.full_url):
            return super().http_open(req)
        return self.do_open(pinned(http.client.HTTPConnection, check_address(req.full_url, self.local_ok)), req)


class PinnedHTTPS(urllib.request.HTTPSHandler):
    def __init__(self, local_ok: bool) -> None:
        super().__init__()
        self.local_ok = local_ok

    def https_open(self, req: Any) -> Any:
        if proxied(req.full_url):
            return super().https_open(req)
        dialer = pinned(http.client.HTTPSConnection, check_address(req.full_url, self.local_ok))
        return self.do_open(dialer, req, context=self._context)


class GuardedRedirects(urllib.request.HTTPRedirectHandler):
    def __init__(self, local_ok: bool) -> None:
        self.local_ok = local_ok

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Any:
        # the user approved one site, so a move to another needs its own approval
        if urlparse(newurl).hostname != urlparse(req.full_url).hostname:
            raise ToolError(MOVED.format(url=newurl))
        check_address(newurl, self.local_ok)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch(url: str, local_ok: bool) -> tuple[str, str]:
    check_address(url, local_ok)
    opener = urllib.request.build_opener(GuardedRedirects(local_ok), PinnedHTTP(local_ok), PinnedHTTPS(local_ok))
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with opener.open(request, timeout=TIMEOUT) as response:
            raw = response.read(MAX_BYTES)
            charset = response.headers.get_content_charset() or "utf-8"
            return raw.decode(charset, errors="replace"), response.headers.get_content_type()
    except (urllib.error.URLError, OSError, LookupError) as error:
        raise ToolError(f"Fetch failed. {error}")


def passages(text: str, query: str) -> str:
    wanted = set(tokenize(query))
    ranked = sorted(
        ((len(wanted & set(tokenize(f"{part.heading} {part.text}"))), -part.ordinal, part) for part in chunk(text)),
        reverse=True,
    )
    picked, used = [], 0
    for score, _, part in ranked:
        if not score or (picked and used + len(part.text) > WEB_TOKENS * CHARS_PER_TOKEN):
            break
        picked.append(part)
        used += len(part.text)
    picked.sort(key=lambda part: part.ordinal)
    return "\n\n".join((f"({part.heading})\n" if part.heading else "") + part.text for part in picked)


async def run(ctx: ToolContext, args: dict[str, Any]) -> str:
    body, kind = await asyncio.to_thread(fetch, args["url"], ctx.local_web)
    text = clean(to_text(body, kind))
    if not text:
        return "[empty page]"
    shown = passages(text, args["query"]) if args.get("query") else ""
    if not shown:
        shown, _ = cap(text, WEB_TOKENS)
    if len(shown) < len(text):
        spill = ctx.new_file("out", ".txt")
        spill.write_text(text)
        shown += f"\n[page cut. Full text: {spill} ({len(text):,} chars)]"
    return shown


TOOL = Tool(SPEC, run, lambda args: False)
