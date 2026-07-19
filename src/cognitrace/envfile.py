"""Minimal `.env` loader -- stdlib only, no new dependency for reading
KEY=VALUE lines. Real environment variables always win; `.env` only fills
in what's missing, so a session that already exported a key isn't
silently overridden by a stale `.env` value. `.env` itself is gitignored
(never committed) and lives at the repo root.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_loaded = False


def load_dotenv(path: Path | None = None) -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    candidate = path or (_REPO_ROOT / ".env")
    if not candidate.exists():
        return
    for line in candidate.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)
