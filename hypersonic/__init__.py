from pathlib import Path

__version__ = "0.1.0"


def code_fingerprint() -> str:
    """Version plus the newest source mtime, so a running server is detected as stale after an edit."""
    root = Path(__file__).resolve().parent
    latest = max((path.stat().st_mtime_ns for path in root.rglob("*.py") if path.exists()), default=0)
    return f"{__version__}:{latest}"
