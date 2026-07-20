"""Stage 1 of the phased-ranking pipeline (design_scaffold.md S9): the
first-phase candidate pools. Each pool is an independent, cheap SQL query
over the Sprint 3 store; RRF fusion (fusion.py) is what reconciles their
incomparable score distributions, not this module.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

_TOKEN = re.compile(r"[A-Za-z0-9]+")

_SELECT = (
    "SELECT mr.id, mr.subject_key, mr.attribute, mr.value, mr.valid_from, mr.valid_to, "
    "mr.event_time_lo, mr.event_time_hi, re.session_id, re.turn_id "
)
_JOIN = (
    "FROM memory_records mr "
    "JOIN memory_events me ON me.id = mr.source_event_id "
    "JOIN raw_evidence re ON re.id = me.raw_evidence_id "
)


@dataclass
class PoolHit:
    record_id: int
    subject_key: str
    attribute: str
    value: str
    valid_from: str
    valid_to: str | None
    event_time_lo: str | None
    event_time_hi: str | None
    session_id: str
    turn_id: str


def _hit_from_row(row: sqlite3.Row) -> PoolHit:
    return PoolHit(
        record_id=row["id"], subject_key=row["subject_key"], attribute=row["attribute"],
        value=row["value"], valid_from=row["valid_from"], valid_to=row["valid_to"],
        event_time_lo=row["event_time_lo"], event_time_hi=row["event_time_hi"],
        session_id=row["session_id"], turn_id=row["turn_id"],
    )


def _fts_match_expr(text: str) -> str | None:
    tokens = _TOKEN.findall(text)
    return " OR ".join(f'"{t}"' for t in tokens) if tokens else None


def pool_lex(conn: sqlite3.Connection, query_text: str, *, limit: int) -> list[PoolHit]:
    """Stage 1 (S9): FTS5 BM25 over `records_fts`, joined back to
    `memory_records` + provenance. Per A5's accepted default, the FTS
    index carries live AND superseded rows (no UPDATE trigger removes a
    record from it when it's superseded) -- freshness comes from the
    `pool_valid` rescue arm, not from filtering here. O(log R + limit)
    via SQLite's FTS5 inverted index, R = indexed row count."""
    query = _fts_match_expr(query_text)
    if query is None:
        return []
    rows = conn.execute(
        _SELECT + "FROM records_fts JOIN memory_records mr ON mr.id = records_fts.rowid "
        "JOIN memory_events me ON me.id = mr.source_event_id "
        "JOIN raw_evidence re ON re.id = me.raw_evidence_id "
        "WHERE records_fts MATCH ? ORDER BY rank LIMIT ?",
        (query, limit),
    ).fetchall()
    return [_hit_from_row(r) for r in rows]


def pool_valid(conn: sqlite3.Connection, *, limit: int) -> list[PoolHit]:
    """Stage 1 (S9): the freshness rescue arm -- every currently-live
    record (`valid_to IS NULL`), most-recent first, independent of lexical
    match quality. Guarantees a live value survives fusion even when its
    BM25 rank is poor (e.g. it was phrased differently than the query).
    O(N) full scan of `memory_records` plus O(M log M) sort of the M live
    rows; the partial index on `(subject_key) WHERE valid_to IS NULL` does
    not cover this query's `ORDER BY valid_from DESC, id DESC`, so it is
    not used -- acceptable at Phase-1 cardinality (hundreds of rows);
    revisit if this becomes a hot path."""
    rows = conn.execute(
        _SELECT + _JOIN + "WHERE mr.valid_to IS NULL "
        "ORDER BY mr.valid_from DESC, mr.id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [_hit_from_row(r) for r in rows]


def pool_temporal_window(
    conn: sqlite3.Connection, query_text: str, lo: str, hi: str, *, limit: int
) -> list[PoolHit]:
    """Stage 1 (S9/A2), the hard-gate tier's first-phase SQL filter: FTS5
    BM25 (same tokenization as `pool_lex`) restricted to records whose
    `event_time` interval overlaps `[lo, hi]` -- "rank lexically within
    the window" (A2). Records with no `event_time` (NULL) never match --
    a window filter can't include what has no dated interval to compare.
    Overlap, not containment: a record spanning "2023-03-15..2023-04-15"
    matches a "2023-03-01..2023-03-31" window even though it isn't fully
    inside it, the same interval-overlap semantics as any calendar query.

    Stated design choice, not an accident: this filters on the window
    AND the lexical MATCH together (FTS5 MATCH is inherently a filter,
    not just a ranker) -- a defensible SQL-natural reading of "rank
    lexically within the window", but a real tradeoff against the
    alternative (window-only, lexical as ORDER BY not a gate). The
    consequence: an in-window record whose value shares no token with
    the query is excluded here even though `pool_valid`'s lexical-match-
    free rescue would have found it on the soft path -- see
    `tests/test_retrieval.py::test_hard_gate_can_exclude_an_in_window_record_lacking_lexical_overlap`
    for the empirical case. Not revisited without a measured reason to
    prefer the other reading (S10-style: no tuning before evidence).

    The caller (pipeline.py) decides what an empty result means (A2's
    empty-window fallback to the fused path) -- this function only
    filters. O(log R + limit) via SQLite's FTS5 inverted index, R =
    indexed row count, same as `pool_lex`."""
    query = _fts_match_expr(query_text)
    if query is None:
        return []
    rows = conn.execute(
        _SELECT + "FROM records_fts JOIN memory_records mr ON mr.id = records_fts.rowid "
        "JOIN memory_events me ON me.id = mr.source_event_id "
        "JOIN raw_evidence re ON re.id = me.raw_evidence_id "
        "WHERE records_fts MATCH ? AND mr.event_time_lo IS NOT NULL AND mr.event_time_hi IS NOT NULL "
        "AND mr.event_time_lo <= ? AND mr.event_time_hi >= ? "
        "ORDER BY rank LIMIT ?",
        (query, hi, lo, limit),
    ).fetchall()
    return [_hit_from_row(r) for r in rows]


def pool_prot(conn: sqlite3.Connection, *, limit: int) -> list[PoolHit]:
    """Stage 1 (S9): flat dense-vector scan, stubbed. Phase 1a is
    lexical-only (A1); the `embeddings` sidecar table exists in the schema
    but nothing populates it until Phase 1b's judgment-list gate. Returns
    [] so fusion.py proves it tolerates an empty pool without a special
    case. O(1)."""
    del conn, limit  # unused this sprint; kept for interface parity with pool_lex/pool_valid
    return []
