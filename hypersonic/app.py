from __future__ import annotations

WIDTH = 1120
HEIGHT = 840
MIN_WIDTH = 720
MIN_HEIGHT = 540
BG = "#0b0e14"
MISSING = (
    "The desktop app needs pywebview. Install it with:\n"
    "  uv tool install --editable '.[app]'\n"
    "or, in the project's environment:  uv pip install pywebview"
)


def run_app(url: str) -> None:
    """Open the Hypersonic UI in a native OS window (WebKit on macOS, no Chromium)."""
    import webview

    webview.create_window(
        "Hypersonic",
        url,
        width=WIDTH,
        height=HEIGHT,
        min_size=(MIN_WIDTH, MIN_HEIGHT),
        background_color=BG,
        text_select=True,
    )
    webview.start()
