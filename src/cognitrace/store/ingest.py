"""Ingest pipeline: one transaction per turn, DLQ on extraction failure.

S8: synchronous single-writer pipeline, jobs/DLQ as tables (not a queue
broker); attempts cap at 1 -- local deterministic encoders have no
transient-failure class, so a failure is dead-lettered immediately with
full context rather than retried. S6: an extractor_version may not be used
until `batch1_parity_verified_at` is set -- ingest refuses outright rather
than silently running an unverified extractor.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field

from cognitrace.store.derive import derive_records_from_events

_EVENT_TYPES = {"FACT", "PREFERENCE", "EVENT_DATED", "RELATIONSHIP"}
_POLARITIES = {"asserted", "negated", "hedged", "hypothetical"}


class UnverifiedExtractorError(RuntimeError):
    """Raised when ingest is attempted against an extractor_version that has
    not passed the batch-1 ONNX parity gate (S6)."""


@dataclass
class ExtractedEvent:
    type: str
    subject_key: str
    attribute: str
    value: str
    polarity: str = "asserted"
    asserts_change: bool = False
    raw_confidence: float = 1.0
    event_time_lo: str | None = None
    event_time_hi: str | None = None
    event_time_grain: str | None = None

    def __post_init__(self) -> None:
        if self.type not in _EVENT_TYPES:
            raise ValueError(f"unknown event type {self.type!r}")
        if self.polarity not in _POLARITIES:
            raise ValueError(f"unknown polarity {self.polarity!r}")


@dataclass
class IngestResult:
    raw_evidence_id: int
    extraction_job_id: int
    status: str  # "ok" | "dead_letter" | "duplicate"
    error: str | None = None
    event_ids: list[int] = field(default_factory=list)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def extractor_version_hash(row: sqlite3.Row) -> str:
    """Covers the entire interpreting configuration -- thresholds and
    calibration have no home outside this versioned hash (S6, the xenc
    triple-trap fix)."""
    parts = "\x1f".join((
        row["model_sha"], row["onnx_sha"], row["onnxruntime_version"],
        row["thresholds_json"], row["calibration_json"],
    ))
    return hashlib.sha256(parts.encode("utf-8")).hexdigest()


def register_extractor_version(
    conn: sqlite3.Connection, *, model_sha: str, onnx_sha: str, onnxruntime_version: str,
    thresholds: dict, calibration: dict, batch1_parity_verified_at: str | None = None,
) -> int:
    thresholds_json = json.dumps(thresholds, sort_keys=True)
    calibration_json = json.dumps(calibration, sort_keys=True)
    conn.execute(
        "INSERT OR IGNORE INTO extractor_versions "
        "(model_sha, onnx_sha, onnxruntime_version, thresholds_json, calibration_json, "
        " batch1_parity_verified_at, created_at) VALUES (?,?,?,?,?,?,?)",
        (model_sha, onnx_sha, onnxruntime_version, thresholds_json, calibration_json,
         batch1_parity_verified_at, _now()),
    )
    row = conn.execute(
        "SELECT id FROM extractor_versions WHERE model_sha=? AND onnx_sha=? AND "
        "onnxruntime_version=? AND thresholds_json=? AND calibration_json=?",
        (model_sha, onnx_sha, onnxruntime_version, thresholds_json, calibration_json),
    ).fetchone()
    return row["id"]


def verify_batch1_parity(conn: sqlite3.Connection, extractor_version_id: int) -> None:
    """Records that this extractor_version has passed the batch-1 ONNX
    parity gate. A separate step from registration on purpose: parity is
    verified by running the actual model, not asserted at registration
    time."""
    conn.execute(
        "UPDATE extractor_versions SET batch1_parity_verified_at = ? WHERE id = ?",
        (_now(), extractor_version_id),
    )


def ingest_turn(
    conn: sqlite3.Connection, *, extractor_version_id: int, turn_id: str, session_id: str,
    role: str, content: str, mention_time: str | None,
    extract_fn,
) -> IngestResult:
    """`extract_fn(content: str) -> list[ExtractedEvent]` may raise; any
    exception dead-letters the turn instead of aborting ingestion. Re-
    ingesting the same (turn_id, extractor_version) is a no-op (idempotency
    key), never a duplicate."""
    ev_row = conn.execute(
        "SELECT id, batch1_parity_verified_at FROM extractor_versions WHERE id = ?",
        (extractor_version_id,),
    ).fetchone()
    if ev_row is None:
        raise ValueError(f"no such extractor_version_id {extractor_version_id}")
    if ev_row["batch1_parity_verified_at"] is None:
        raise UnverifiedExtractorError(
            f"extractor_version {extractor_version_id} has not passed batch-1 parity; refusing to ingest"
        )

    existing = conn.execute(
        "SELECT id FROM raw_evidence WHERE turn_id = ?", (turn_id,)
    ).fetchone()

    conn.execute("BEGIN IMMEDIATE")
    try:
        if existing is not None:
            raw_id = existing["id"]
        else:
            content_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
            cur = conn.execute(
                "INSERT INTO raw_evidence (turn_id, session_id, role, content, content_sha256, "
                "mention_time, recorded_at) VALUES (?,?,?,?,?,?,?)",
                (turn_id, session_id, role, content, content_sha, mention_time, _now()),
            )
            raw_id = cur.lastrowid

        idempotency_key = hashlib.sha256(
            f"{raw_id}\x1f{extractor_version_id}".encode("utf-8")
        ).hexdigest()
        dup = conn.execute(
            "SELECT id, status, error FROM extraction_jobs WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        if dup is not None:
            conn.execute("COMMIT")
            event_ids = [
                r["id"] for r in conn.execute(
                    "SELECT id FROM memory_events WHERE extraction_job_id = ?", (dup["id"],)
                ).fetchall()
            ]
            return IngestResult(raw_id, dup["id"], "duplicate", dup["error"], event_ids)

        try:
            events = extract_fn(content)
            status, error = "ok", None
        except Exception as exc:  # noqa: BLE001 - deterministic failure -> DLQ, never a crash
            events, status, error = [], "dead_letter", f"{type(exc).__name__}: {exc}"

        job_cur = conn.execute(
            "INSERT INTO extraction_jobs (raw_evidence_id, extractor_version_id, "
            "idempotency_key, status, error, created_at) VALUES (?,?,?,?,?,?)",
            (raw_id, extractor_version_id, idempotency_key, status, error, _now()),
        )
        job_id = job_cur.lastrowid

        event_ids: list[int] = []
        for ev in events:
            cur = conn.execute(
                "INSERT INTO memory_events (extraction_job_id, raw_evidence_id, type, "
                "subject_key, attribute, value, polarity, asserts_change, raw_confidence, "
                "event_time_lo, event_time_hi, event_time_grain, mention_time, recorded_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (job_id, raw_id, ev.type, ev.subject_key, ev.attribute, ev.value, ev.polarity,
                 int(ev.asserts_change), ev.raw_confidence, ev.event_time_lo, ev.event_time_hi,
                 ev.event_time_grain, mention_time, _now()),
            )
            event_ids.append(cur.lastrowid)

        if event_ids:
            derive_records_from_events(conn)
        conn.execute("COMMIT")
        return IngestResult(raw_id, job_id, status, error, event_ids)
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def list_dead_letters(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT j.id AS job_id, j.error, j.created_at, r.turn_id, r.session_id, r.content "
        "FROM extraction_jobs j JOIN raw_evidence r ON r.id = j.raw_evidence_id "
        "WHERE j.status = 'dead_letter' ORDER BY j.id"
    ).fetchall()
    return [dict(row) for row in rows]
