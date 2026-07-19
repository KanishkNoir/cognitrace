"""DDL and connection setup for the per-conversation store.

Table roles (design_scaffold.md S3-S8):
  raw_evidence      -- append-only ground truth; never updated or deleted.
  extractor_versions-- one row per (model, onnx, runtime, thresholds,
                       calibration) tuple; `batch1_parity_verified_at` gates
                       ingestion (S6) -- no row may be used until verified.
  extraction_jobs   -- idempotency ledger; UNIQUE(idempotency_key) makes
                       re-ingesting the same (turn, extractor_version) a
                       no-op instead of a duplicate.
  memory_events     -- recorded encoder outputs (labels, keys, values, raw
                       confidence). Ground truth alongside raw_evidence;
                       never recomputed on replay (S6).
  memory_records    -- derived, tri-temporal, droppable view over events.
  supersessions     -- supersession decisions recorded as events (S1).
  records_fts       -- FTS5 external-content index over memory_records.
  embeddings        -- dense vector sidecar (populated from Phase 1b on).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1

_DDL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS extractor_versions (
    id INTEGER PRIMARY KEY,
    model_sha TEXT NOT NULL,
    onnx_sha TEXT NOT NULL,
    onnxruntime_version TEXT NOT NULL,
    thresholds_json TEXT NOT NULL,
    calibration_json TEXT NOT NULL,
    batch1_parity_verified_at TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (model_sha, onnx_sha, onnxruntime_version, thresholds_json, calibration_json)
);

CREATE TABLE IF NOT EXISTS raw_evidence (
    id INTEGER PRIMARY KEY,
    turn_id TEXT NOT NULL UNIQUE,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    mention_time TEXT,
    recorded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS extraction_jobs (
    id INTEGER PRIMARY KEY,
    raw_evidence_id INTEGER NOT NULL REFERENCES raw_evidence(id),
    extractor_version_id INTEGER NOT NULL REFERENCES extractor_versions(id),
    idempotency_key TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL CHECK (status IN ('ok', 'dead_letter')),
    error TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_events (
    id INTEGER PRIMARY KEY,
    extraction_job_id INTEGER NOT NULL REFERENCES extraction_jobs(id),
    raw_evidence_id INTEGER NOT NULL REFERENCES raw_evidence(id),
    type TEXT NOT NULL CHECK (type IN ('FACT', 'PREFERENCE', 'EVENT_DATED', 'RELATIONSHIP')),
    subject_key TEXT NOT NULL,
    attribute TEXT NOT NULL,
    value TEXT NOT NULL,
    polarity TEXT NOT NULL CHECK (polarity IN ('asserted', 'negated', 'hedged', 'hypothetical')),
    asserts_change INTEGER NOT NULL DEFAULT 0 CHECK (asserts_change IN (0, 1)),
    raw_confidence REAL NOT NULL,
    event_time_lo TEXT,
    event_time_hi TEXT,
    event_time_grain TEXT,
    mention_time TEXT,
    recorded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS supersessions (
    id INTEGER PRIMARY KEY,
    superseding_event_id INTEGER NOT NULL REFERENCES memory_events(id),
    superseded_event_id INTEGER NOT NULL REFERENCES memory_events(id),
    reason TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    UNIQUE (superseding_event_id, superseded_event_id)
);

CREATE TABLE IF NOT EXISTS memory_records (
    id INTEGER PRIMARY KEY,
    source_event_id INTEGER NOT NULL REFERENCES memory_events(id),
    type TEXT NOT NULL,
    subject_key TEXT NOT NULL,
    attribute TEXT NOT NULL,
    value TEXT NOT NULL,
    event_time_lo TEXT,
    event_time_hi TEXT,
    valid_from TEXT NOT NULL,
    valid_to TEXT,
    supersedable INTEGER NOT NULL CHECK (supersedable IN (0, 1)),
    superseded_by INTEGER REFERENCES memory_records(id),
    recorded_at TEXT NOT NULL
);

-- S4: the one-live-record invariant is a write-time constraint, not app logic.
CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_records_one_live
    ON memory_records(subject_key) WHERE valid_to IS NULL AND supersedable = 1;

CREATE VIRTUAL TABLE IF NOT EXISTS records_fts USING fts5(
    value, attribute,
    content='memory_records', content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS records_fts_ai AFTER INSERT ON memory_records BEGIN
    INSERT INTO records_fts(rowid, value, attribute) VALUES (new.id, new.value, new.attribute);
END;

CREATE TRIGGER IF NOT EXISTS records_fts_ad AFTER DELETE ON memory_records BEGIN
    INSERT INTO records_fts(records_fts, rowid, value, attribute) VALUES ('delete', old.id, old.value, old.attribute);
END;

CREATE TABLE IF NOT EXISTS embeddings (
    record_id INTEGER NOT NULL REFERENCES memory_records(id),
    extractor_version_id INTEGER NOT NULL REFERENCES extractor_versions(id),
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (record_id, extractor_version_id)
);
"""


def open_store(path: Path) -> sqlite3.Connection:
    """One connection per conversation file (S7). WAL + NORMAL sync trade a
    tiny durability window for write throughput -- acceptable because
    raw_evidence + memory_events (the ground truth) are re-derivable from
    nothing upstream of the ingest call itself; a torn write loses at most
    the in-flight turn, never corrupts prior history."""
    conn = sqlite3.connect(path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_DDL)
    conn.execute(
        "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    return conn
