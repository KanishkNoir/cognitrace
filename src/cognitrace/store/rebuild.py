"""`rebuild --from-raw`: bit-identical replay closure (S5, invariant I7).

Drops and reconstructs memory_records/supersessions purely from the
recorded memory_events -- never by re-running the extractor over
raw_evidence. The FIRST rebuild after ingest records a baseline hash in
`meta`; every subsequent rebuild must reproduce it exactly, or the store's
central claim (raw_evidence + memory_events fully determine everything
else) is false and CI must say so.
"""

from __future__ import annotations

import hashlib
import sqlite3

from cognitrace.store.derive import canonical_dump, derive_records_from_events

_BASELINE_KEY = "replay_baseline_sha256"


def rebuild_from_raw(conn: sqlite3.Connection) -> str:
    """Rebuilds derived tables and returns the canonical-dump SHA-256."""
    derive_records_from_events(conn)
    return hashlib.sha256(canonical_dump(conn).encode("utf-8")).hexdigest()


def record_replay_baseline(conn: sqlite3.Connection) -> str:
    sha = rebuild_from_raw(conn)
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (_BASELINE_KEY, sha),
    )
    return sha


def replay_baseline(conn: sqlite3.Connection) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (_BASELINE_KEY,)).fetchone()
    return row["value"] if row is not None else None
