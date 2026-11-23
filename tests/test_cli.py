import json
import socket
import subprocess
import sys
import threading
from argparse import Namespace

from test_server import served  # noqa: F401
from conftest import say

import hypersonic
from hypersonic import cli
from hypersonic.engine.session import home
from hypersonic.server import HOST, INFO_NAME

NAP = "import time; time.sleep(60)"


def save_pid(pid):
    home().mkdir(parents=True, exist_ok=True)
    (home() / INFO_NAME).write_text(json.dumps({"port": 1, "token": "t", "pid": pid}))


def test_stop_never_kills_a_process_that_is_not_the_server():
    bystander = subprocess.Popen([sys.executable, "-c", NAP])
    try:
        save_pid(bystander.pid)
        assert cli.terminate_saved() is False
        assert bystander.poll() is None and not (home() / INFO_NAME).exists()
    finally:
        bystander.kill()


def test_stop_ends_the_saved_server_and_reports_it(capsys):
    server = subprocess.Popen([sys.executable, "-c", NAP, "hypersonic", "serve"])
    save_pid(server.pid)
    assert cli.stop(Namespace()) == 0 and server.wait(timeout=10) != 0
    assert cli.stop(Namespace()) == 0
    assert capsys.readouterr().out.split("\n")[:2] == ["Server stopped.", "No server running."]


def test_a_port_now_used_by_something_else_reads_as_no_server():
    with socket.socket() as other:
        other.bind((HOST, 0))
        other.listen()

        def babble():
            taken, _ = other.accept()
            taken.recv(1024)
            taken.sendall(b"not http at all\r\n\r\n")
            taken.close()

        threading.Thread(target=babble, daemon=True).start()
        assert cli.Client(other.getsockname()[1], "t").alive() is False


def test_run_fails_when_its_slash_command_fails_and_frees_the_session(served, workspace, monkeypatch, capsys):
    client, scripts = served
    monkeypatch.setattr(cli, "connect", lambda: client)
    monkeypatch.delenv("HYPERSONIC_MODEL", raising=False)
    scripts.extend([[], [], [say("done")]])
    assert cli.run(Namespace(task="/nosuchcommand", cwd=str(workspace))) == 1
    assert cli.run(Namespace(task="/help", cwd=str(workspace))) == 0
    assert cli.run(Namespace(task="do it", cwd=str(workspace))) == 0
    assert client.call("GET", "/session") == [] and "done" in capsys.readouterr().out


def test_the_model_named_in_this_shell_is_sent_to_the_server(workspace, monkeypatch):
    monkeypatch.setenv("HYPERSONIC_MODEL", "ollama/from-this-shell")
    assert cli.session_options(Namespace(cwd=str(workspace)), interactive=False)["model"] == "ollama/from-this-shell"
    assert cli.session_options(Namespace(cwd=str(workspace), model="anthropic/claude-opus-5-5"), interactive=False)["model"] == "anthropic/claude-opus-5-5"
    assert cli.folder_of(Namespace(cwd=".")) == cli.folder_of(Namespace())


def test_an_uploads_folder_alone_does_not_ask_for_trust(workspace, monkeypatch):
    asked = []
    monkeypatch.setattr(cli, "confirm", lambda question="": asked.append(question) or False)
    (workspace / ".hypersonic" / "uploads").mkdir(parents=True)
    assert cli.is_trusted(str(workspace), Namespace(), interactive=True) is False and asked == []
    (workspace / ".hypersonic" / "commands").mkdir()
    assert cli.is_trusted(str(workspace), Namespace(), interactive=True) is False and len(asked) == 1


def test_the_fingerprint_survives_a_dangling_link(tmp_path, monkeypatch):
    (tmp_path / "real.py").write_text("x = 1")
    (tmp_path / ".#lock.py").symlink_to(tmp_path / "gone.py")
    monkeypatch.setattr(hypersonic, "__file__", str(tmp_path / "__init__.py"))
    assert hypersonic.code_fingerprint().startswith(hypersonic.__version__)


def test_a_missing_window_library_gives_the_install_hint(monkeypatch, capsys, tmp_path):
    from hypersonic import app

    def missing(url):
        raise ModuleNotFoundError("No module named 'webview'")

    home().mkdir(parents=True, exist_ok=True)
    (home() / INFO_NAME).write_text(json.dumps({"port": 1, "token": "t", "pid": 1}))
    monkeypatch.setattr(cli, "connect", lambda: None)
    monkeypatch.setattr(app, "run_app", missing)
    assert cli.desktop(Namespace(cwd=str(tmp_path))) == 1
    assert "pywebview" in capsys.readouterr().err
