"""Deterministic derivation of memory_records/supersessions from memory_events.

This is the ONE function that produces the droppable materialized views
(S5). It is called both at the end of every `ingest_turn` and by
`rebuild_from_raw` -- using the identical function in both places is what
makes replay bit-identical by construction rather than by hoping a second
implementation stays in sync (S6's replay-determinism contract).

Supersession here is deliberately the simplest possible mechanism: shared
subject_key on a supersedable type closes the prior live record (S1 - the
edge is a mechanism, not a classifier label). The real subject-key
normalizer and confidence-threshold linkage are Phase 1a Sprint 4 work
(4.2/4.3); this exact-key rule is enough to prove the store's invariants
now without waiting on it.
"""

from __future__ import annotations

import sqlite3


def derive_records_from_events(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM supersessions")
    conn.execute("DELETE FROM memory_records")
    live: dict[str, int] = {}
    for event in conn.execute("SELECT * FROM memory_events ORDER BY id").fetchall():
        supersedable = 0 if event["type"] == "EVENT_DATED" else 1
        valid_from = event["mention_time"] or event["recorded_at"]
        prev_id = live.get(event["subject_key"]) if supersedable else None
        if prev_id is not None:
            # Close the prior live row FIRST: the partial-unique index forbids
            # a second live row for this subject_key existing even transiently.
            conn.execute(
                "UPDATE memory_records SET valid_to = ? WHERE id = ?", (valid_from, prev_id)
            )
        cur = conn.execute(
            "INSERT INTO memory_records (source_event_id, type, subject_key, attribute, "
            "value, event_time_lo, event_time_hi, valid_from, valid_to, supersedable, "
            "superseded_by, recorded_at) VALUES (?,?,?,?,?,?,?,?,NULL,?,NULL,?)",
            (event["id"], event["type"], event["subject_key"], event["attribute"],
             event["value"], event["event_time_lo"], event["event_time_hi"], valid_from,
             supersedable, event["recorded_at"]),
        )
        new_id = cur.lastrowid
        if prev_id is not None:
            conn.execute(
                "UPDATE memory_records SET superseded_by = ? WHERE id = ?", (new_id, prev_id)
            )
            prev_source_event_id = conn.execute(
                "SELECT source_event_id FROM memory_records WHERE id = ?", (prev_id,)
            ).fetchone()["source_event_id"]
            conn.execute(
                "INSERT INTO supersessions (superseding_event_id, superseded_event_id, "
                "reason, recorded_at) VALUES (?,?,?,?)",
                (event["id"], prev_source_event_id, "exact-subject-key", event["recorded_at"]),
            )
        if supersedable:
            live[event["subject_key"]] = new_id


def canonical_dump(conn: sqlite3.Connection) -> str:
    """Deterministic text serialization of the derived state, for the
    replay-closure hash (I7). Only covers columns derive_records_from_events
    itself controls -- `recorded_at`/`id` values are inputs it preserves
    verbatim, not degrees of freedom it could drift on."""
    lines: list[str] = []
    for row in conn.execute(
        "SELECT id, source_event_id, type, subject_key, attribute, value, event_time_lo, "
        "event_time_hi, valid_from, valid_to, supersedable, superseded_by FROM memory_records "
        "ORDER BY id"
    ):
        lines.append("R\x1f" + "\x1f".join(str(v) for v in tuple(row)))
    for row in conn.execute(
        "SELECT id, superseding_event_id, superseded_event_id, reason FROM supersessions ORDER BY id"
    ):
        lines.append("S\x1f" + "\x1f".join(str(v) for v in tuple(row)))
    return "\n".join(lines)
