import asyncio
import json

import pytest

import stat

from hypersonic.tools import files as files_tool
from hypersonic.tools.base import ToolError
from hypersonic.tools.files import TOOL, VIEW_LINES, undo_last


def files(ctx, **args):
    return asyncio.run(TOOL.run(ctx, args))


def test_create_then_view(ctx, workspace):
    assert files(ctx, command="create", path="notes/a.txt", file_text="one\ntwo\n") == "Wrote notes/a.txt."
    assert (workspace / "notes/a.txt").read_text() == "one\ntwo\n"
    assert files(ctx, command="view", path="notes/a.txt") == "one\ntwo"


def test_view_range_and_partial_note(ctx, workspace):
    (workspace / "long.txt").write_text("\n".join(f"row {n}" for n in range(1, 1001)))
    part = files(ctx, command="view", path="long.txt", view_range=[10, 12])
    assert part == "row 10\nrow 11\nrow 12\n[lines 10-12 of 1000. Use view_range for other parts.]"
    whole = files(ctx, command="view", path="long.txt")
    assert f"[lines 1-{VIEW_LINES} of 1000." in whole
    tail = files(ctx, command="view", path="long.txt", view_range=[999, -1])
    assert tail.startswith("row 999\nrow 1000\n")


def test_view_directory_and_empty_file(ctx, workspace):
    (workspace / "sub").mkdir()
    (workspace / "b.txt").write_text("")
    assert files(ctx, command="view", path=".") == "b.txt\nsub/"
    assert files(ctx, command="view", path="b.txt") == "[empty file]"


def test_str_replace_needs_exactly_one_match(ctx, workspace):
    target = workspace / "a.py"
    target.write_text("x = 1\ny = 1\n")
    assert files(ctx, command="str_replace", path="a.py", old_str="x = 1", new_str="x = 2") == "Edited a.py."
    assert target.read_text() == "x = 2\ny = 1\n"
    with pytest.raises(ToolError, match="not found"):
        files(ctx, command="str_replace", path="a.py", old_str="z = 9", new_str="")
    with pytest.raises(ToolError, match="matches 2 places"):
        files(ctx, command="str_replace", path="a.py", old_str=" = ", new_str=" == ")


def test_insert(ctx, workspace):
    target = workspace / "a.txt"
    target.write_text("one\ntwo")
    files(ctx, command="insert", path="a.txt", insert_line=0, insert_text="zero")
    files(ctx, command="insert", path="a.txt", insert_line=3, insert_text="three")
    assert target.read_text() == "zero\none\ntwo\nthree\n"
    with pytest.raises(ToolError, match="insert_line must be"):
        files(ctx, command="insert", path="a.txt", insert_line=99, insert_text="x")


def test_paths_are_confined_to_the_workspace(ctx, workspace, tmp_path):
    outside = tmp_path / "secret.txt"
    outside.write_text("secret")
    for command in ("view", "create"):
        with pytest.raises(ToolError, match="outside the workspace"):
            files(ctx, command=command, path=str(outside), file_text="x")
    with pytest.raises(ToolError, match="outside the workspace"):
        files(ctx, command="view", path="../secret.txt")
    link = workspace / "link.txt"
    link.symlink_to(outside)
    with pytest.raises(ToolError, match="outside the workspace"):
        files(ctx, command="view", path="link.txt")


def test_a_link_swapped_in_after_the_path_check_is_never_followed(ctx, workspace, tmp_path, monkeypatch):
    outside = tmp_path / "private"
    outside.mkdir()
    (outside / "key.txt").write_text("secret")
    (workspace / "a.txt").write_text("ok")
    (workspace / "sub").mkdir()
    checked = files_tool.resolve

    def then_swap(ctx, raw, write):
        path = checked(ctx, raw, write)
        if raw == "a.txt":
            path.unlink()
            path.symlink_to(outside / "key.txt")
        else:
            (workspace / "sub").rmdir()
            (workspace / "sub").symlink_to(outside, target_is_directory=True)
        return path

    monkeypatch.setattr(files_tool, "resolve", then_swap)
    for attempt in ({"command": "view", "path": "a.txt"}, {"command": "create", "path": "a.txt", "file_text": "pwned"}):
        (workspace / "a.txt").unlink()
        (workspace / "a.txt").write_text("ok")
        with pytest.raises(ToolError, match="Can't open a.txt"):
            files(ctx, **attempt)
    for attempt in ({"command": "view", "path": "sub/key.txt"}, {"command": "create", "path": "sub/new.txt", "file_text": "pwned"}):
        if (workspace / "sub").is_symlink():
            (workspace / "sub").unlink()
            (workspace / "sub").mkdir()
        with pytest.raises(ToolError, match="outside the workspace"):
            files(ctx, **attempt)
    assert (outside / "key.txt").read_text() == "secret" and [entry.name for entry in outside.iterdir()] == ["key.txt"]
    assert not list((ctx.session_dir / "undo").glob("*")) if (ctx.session_dir / "undo").exists() else True


def test_an_edit_never_damages_a_file_that_is_not_utf8(ctx, workspace):
    raw = b"# caf\xe9 menu\nprice = 1\n"
    (workspace / "menu.py").write_bytes(raw)
    assert "price = 1" in files(ctx, command="view", path="menu.py")
    with pytest.raises(ToolError, match="isn't UTF-8"):
        files(ctx, command="str_replace", path="menu.py", old_str="price = 1", new_str="price = 2")
    assert (workspace / "menu.py").read_bytes() == raw


def test_undo_restores_the_file_and_skips_a_write_that_failed(ctx, workspace):
    script = workspace / "run.sh"
    script.write_text("v1")
    script.chmod(0o755)
    files(ctx, command="create", path="run.sh", file_text="v2")
    with pytest.raises(ToolError, match="Can't open child.txt"):
        files(ctx, command="create", path="run.sh/child.txt", file_text="x")
    assert undo_last(ctx) == f"Restored {script}." and undo_last(ctx) == "Nothing to undo."
    assert script.read_text() == "v1" and stat.S_IMODE(script.stat().st_mode) == 0o755


def test_line_numbers_match_other_tools_when_a_file_has_unusual_breaks(ctx, workspace):
    (workspace / "old.c").write_text("one\n\x0ctwo\u2028still two\r\nthree\n")
    assert files(ctx, command="view", path="old.c", view_range=[3, 3]).startswith("three\n[lines 3-3 of 3")
    files(ctx, command="insert", path="old.c", insert_line=2, insert_text="added")
    assert (workspace / "old.c").read_text().split("\n")[2:4] == ["added", "three"]


def test_a_huge_file_is_refused_instead_of_loaded(ctx, workspace, monkeypatch):
    monkeypatch.setattr(files_tool, "MAX_FILE_BYTES", 10)
    (workspace / "big.txt").write_text("x" * 11)
    with pytest.raises(ToolError, match="too big to load"):
        files(ctx, command="view", path="big.txt")


def test_session_folder_is_readable_but_not_writable(ctx):
    spill = ctx.new_file("out", ".txt")
    spill.write_text("full output")
    assert files(ctx, command="view", path=str(spill)) == "full output"
    with pytest.raises(ToolError, match="outside the workspace"):
        files(ctx, command="create", path=str(spill), file_text="x")


def test_edit_is_rejected_when_file_changed_on_disk(ctx, workspace):
    target = workspace / "a.txt"
    target.write_text("old")
    files(ctx, command="view", path="a.txt")
    target.write_text("changed by someone else")
    with pytest.raises(ToolError, match="changed on disk"):
        files(ctx, command="str_replace", path="a.txt", old_str="changed", new_str="x")
    files(ctx, command="view", path="a.txt")
    files(ctx, command="str_replace", path="a.txt", old_str="changed", new_str="edited")
    assert target.read_text() == "edited by someone else"


def test_pre_image_is_saved_before_a_change(ctx, workspace):
    target = workspace / "a.txt"
    target.write_text("original")
    files(ctx, command="create", path="a.txt", file_text="new")
    files(ctx, command="create", path="fresh.txt", file_text="x")
    records = [json.loads(line) for line in (ctx.session_dir / "undo" / "index.jsonl").read_text().splitlines()]
    assert open(records[0]["backup"]).read() == "original"
    assert records[1] == {"path": str(workspace / "fresh.txt"), "backup": None}


def test_binary_and_missing_files(ctx, workspace):
    (workspace / "blob.bin").write_bytes(b"\x00\x01\x02")
    with pytest.raises(ToolError, match="binary"):
        files(ctx, command="view", path="blob.bin")
    with pytest.raises(ToolError, match="doesn't exist"):
        files(ctx, command="view", path="nope.txt")
    with pytest.raises(ToolError, match="needs file_text"):
        files(ctx, command="create", path="x.txt")
