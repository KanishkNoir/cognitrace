"""`rebuild --from-raw`: bit-identical replay closure (S5, invariant I7).

`rebuild_from_raw` drops and reconstructs memory_records/supersessions
purely from the recorded memory_events -- never by re-running the
extractor. It is an explicit, deliberate, MUTATING operation (the CLI
`rebuild --from-raw` command, or a one-time baseline recording) -- it must
never run as a side effect of a routine health check. `doctor`'s I7 check
only hashes the already-materialized state and compares it to the
recorded baseline; it does not call this function.

Note on S5 wording: raw_evidence is the ultimate ground truth, but
memory_events (recorded encoder outputs) are also never recomputed on
replay (S6) -- they are what this function replays FROM. Only
memory_records/supersessions/FTS/embeddings are the droppable views this
module regenerates.
"""

from __future__ import annotations

import hashlib
import sqlite3

from cognitrace.store.derive import canonical_dump, derive_records_from_events

_BASELINE_KEY = "replay_baseline_sha256"


def rebuild_from_raw(conn: sqlite3.Connection) -> str:
    """MUTATES the store: drops and reconstructs derived tables, returns the
    resulting canonical-dump SHA-256. Call deliberately, never from `doctor`."""
    derive_records_from_events(conn)
    return current_canonical_sha(conn)


def current_canonical_sha(conn: sqlite3.Connection) -> str:
    """Read-only: hashes whatever is currently materialized. No re-derive."""
    return hashlib.sha256(canonical_dump(conn).encode("utf-8")).hexdigest()


def record_replay_baseline(conn: sqlite3.Connection) -> str:
    """Deliberately re-derives once (proving derive-from-events reproduces
    the current state) and freezes the result as the reference for future
    I7 checks. Must be re-recorded after any ingest that should become the
    new 'known good' -- the baseline is a frozen snapshot, not a live
    recomputation (see CONTRACTS.md)."""
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
