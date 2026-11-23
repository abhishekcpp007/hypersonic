from hypersonic.context.squeeze import CHARS_PER_TOKEN, cap, clean


def test_clean_strips_ansi_and_progress_bars():
    assert clean("\x1b[31mred\x1b[0m") == "red"
    assert clean("10%\r50%\r100%") == "100%"
    assert clean("windows line\r\nnext") == "windows line\nnext"


def test_clean_collapses_repeats_and_blank_runs():
    assert clean("a\na\na\na\nb") == "a  [repeated 4x]\nb"
    assert clean("a\na\nb") == "a\na\nb"
    assert clean("x\n\n\n\ny") == "x\n\ny"


def test_cap_leaves_short_text_alone():
    assert cap("short", 10) == ("short", 0)


def test_cap_keeps_head_and_tail():
    text = "\n".join(f"line {number}" for number in range(1000))
    budget = 50
    body, cut = cap(text, budget)
    assert cut > 0
    assert body.startswith("line 0\n")
    assert body.endswith("line 999")
    assert "chars cut" in body
    assert len(body) < budget * CHARS_PER_TOKEN + 100


def test_cap_handles_one_huge_line():
    body, cut = cap("x" * 10_000, 100)
    assert cut == 10_000 - 100 * CHARS_PER_TOKEN
    assert len(body) < 100 * CHARS_PER_TOKEN + 100


def test_clean_then_cap_collapses_repeats():
    noisy = "\n".join(["same"] * 5000)
    body, cut = cap(clean(noisy), 100)
    assert body == "same  [repeated 5000x]"
    assert cut == 0
