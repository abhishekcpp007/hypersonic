import asyncio
import re
import socket
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from conftest import say, use

from hypersonic.engine.loop import DENIED, run_turn
from hypersonic.policy.rules import ALLOW, ASK, ASK_MODE, AUTO, YOLO, Policy, Rules
from hypersonic.providers.base import ToolCall
from hypersonic.tools.base import ToolError
from hypersonic.tools.web import TOOL, WEB_TOKENS, proxied, public
from hypersonic.context.squeeze import CHARS_PER_TOKEN

PAGE = """<html><head><title>Zoo</title><style>body{color:red}</style></head><body>
<script>alert("tracking")</script>
<h1>Zoo guide</h1><p>Welcome to the zoo.</p>
<h2>Walrus</h2><p>The walrus eats clams every morning.</p>
<h2>Penguin</h2><p>Penguins huddle for warmth.</p>
</body></html>"""
PUBLIC_ADDRESS = "93.184.216.34"
REBINDING_HOST = "rebind.test"
LONG = "<html><body>" + "".join(f"<h2>Animal {n}</h2><p>{'filler words here ' * 60}</p>" for n in range(60)) + "<h2>Okapi</h2><p>The okapi has a striped rump.</p></body></html>"


class Pages(BaseHTTPRequestHandler):
    asked: list[str] = []

    def do_GET(self):
        if self.path in ("/moved", "/elsewhere"):
            there = "/zoo" if self.path == "/moved" else f"http://localhost:{self.server.server_port}/zoo?leak=SECRET"
            self.send_response(302)
            self.send_header("Location", there)
            self.end_headers()
            return
        Pages.asked.append(self.path)
        body = {"/zoo": PAGE, "/long": LONG, "/plain": "just text"}.get(self.path)
        self.send_response(200 if body else 404)
        self.send_header("Content-Type", "text/plain" if self.path == "/plain" else "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write((body or "missing").encode())

    def log_message(self, *args):
        pass


@pytest.fixture
def site():
    server = HTTPServer(("127.0.0.1", 0), Pages)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def web(ctx, **args):
    return asyncio.run(TOOL.run(ctx, args))


def test_page_becomes_clean_text(ctx, site):
    ctx.local_web = True
    text = web(ctx, url=f"{site}/zoo")
    assert text.startswith("Zoo\n\n# Zoo guide\n\nWelcome to the zoo.")
    assert "## Walrus\n\nThe walrus eats clams every morning." in text
    assert "alert" not in text and "color:red" not in text
    assert web(ctx, url=f"{site}/moved") == text
    assert web(ctx, url=f"{site}/plain") == "just text"


def test_query_returns_only_matching_passages(ctx, site):
    ctx.local_web = True
    assert web(ctx, url=f"{site}/zoo", query="what does the walrus eat").startswith("(Zoo guide > Walrus)\nThe walrus eats clams every morning.")
    focused = web(ctx, url=f"{site}/long", query="okapi striped rump")
    assert focused.startswith("(Okapi)\nThe okapi has a striped rump.")
    assert len(focused) < 400


def test_long_pages_are_cut_and_saved(ctx, site):
    ctx.local_web = True
    text = web(ctx, url=f"{site}/long")
    assert len(text) < WEB_TOKENS * CHARS_PER_TOKEN + 300
    spill = Path(re.search(r"Full text: (\S+)", text).group(1))
    assert "The okapi has a striped rump." in spill.read_text()


def test_local_addresses_and_odd_schemes_are_refused(ctx, site):
    with pytest.raises(ToolError, match="private or local address"):
        web(ctx, url=f"{site}/zoo")
    with pytest.raises(ToolError, match="private or local address"):
        web(ctx, url="http://169.254.169.254/latest/meta-data/")
    with pytest.raises(ToolError, match="Only http and https"):
        web(ctx, url="file:///etc/passwd")
    ctx.local_web = True
    with pytest.raises(ToolError, match="Fetch failed"):
        web(ctx, url=f"{site}/nope")


def test_fetch_dials_the_address_it_checked(ctx, site, monkeypatch):
    port = int(site.rsplit(":", 1)[1])
    lookup, connect, dialed = socket.getaddrinfo, socket.create_connection, []

    def one_public_answer(host, *args, **kwargs):
        if host != REBINDING_HOST:
            return lookup(host, *args, **kwargs)
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (PUBLIC_ADDRESS, port))]

    def to_the_test_site(target, *args, **kwargs):
        dialed.append(target[0])
        return connect(("127.0.0.1", target[1]), *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", one_public_answer)
    monkeypatch.setattr(socket, "create_connection", to_the_test_site)
    monkeypatch.setattr(urllib.request, "getproxies", dict)
    assert "The walrus eats clams" in web(ctx, url=f"http://{REBINDING_HOST}:{port}/moved")
    assert dialed == [PUBLIC_ADDRESS, PUBLIC_ADDRESS]


def test_a_redirect_to_another_site_stops_and_names_it(ctx, site):
    ctx.local_web = True
    Pages.asked.clear()
    with pytest.raises(ToolError, match="redirects to another site: http://localhost"):
        web(ctx, url=f"{site}/elsewhere")
    assert Pages.asked == []
    assert "The walrus eats clams" in web(ctx, url=f"{site}/moved")


def test_only_a_proxy_that_really_applies_switches_pinning_off(monkeypatch):
    monkeypatch.setattr(urllib.request, "getproxies", lambda: {"no": "localhost", "all": "socks5://127.0.0.1:1"})
    assert proxied("https://example.com/x") is False
    monkeypatch.setattr(urllib.request, "getproxies", lambda: {"https": "http://proxy:8080"})
    monkeypatch.setattr(urllib.request, "proxy_bypass", lambda host: host == "inside.example")
    assert proxied("https://example.com/x") is True
    assert proxied("http://example.com/x") is False and proxied("https://inside.example/x") is False


def test_nat64_wrapping_does_not_hide_a_local_address():
    assert public(PUBLIC_ADDRESS) and public("64:ff9b::808:808")
    assert not public("64:ff9b::7f00:1") and not public("127.0.0.1")


def test_each_site_needs_approval_unless_listed(workspace):
    call = ToolCall("1", "web", {"url": "https://docs.python.org/3/library/sqlite3.html"})
    for mode in (ASK_MODE, AUTO):
        assert Policy(mode).decide(call, workspace, workspace)[0] == ASK
    assert Policy(YOLO).decide(call, workspace, workspace)[0] == ALLOW
    listed = Policy(AUTO, Rules(web=("docs.python.org", "*.readthedocs.io")))
    assert listed.decide(call, workspace, workspace)[0] == ALLOW
    other = ToolCall("2", "web", {"url": "https://evil.example/?leak=secret"})
    assert listed.decide(other, workspace, workspace)[0] == ASK


def test_unapproved_fetch_never_leaves_the_machine(make_agent, site):
    agent, _ = make_agent([use(("web", {"url": f"{site}/zoo"})), say("ok")])
    agent.ctx.local_web = True
    asyncio.run(run_turn(agent, "look it up"))
    assert agent.session.messages[2]["content"][0]["content"] == DENIED
