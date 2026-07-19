"""Invariant suite I1-I7 (design_scaffold.md S3-S6).

I1 idempotency        -- no duplicate (raw_evidence, extractor_version) processed twice.
I2 one-live-record    -- no subject_key has >1 live supersedable record.
I3 event provenance   -- every memory_event.raw_evidence_id resolves.
I4 record provenance  -- every memory_records.source_event_id resolves.
I5 tri-temporal order -- event_time_lo<=hi and valid_from<=valid_to where set.
I6 parity gate        -- no extraction_job used an unverified extractor_version.
I7 replay closure     -- rebuild --from-raw reproduces the recorded baseline hash.

Each check is defense-in-depth: I1/I2 are also enforced by UNIQUE
constraints/partial indexes at write time, so a violation here means the
schema's own guarantees were bypassed (e.g. a raw executescript, or
PRAGMA foreign_keys off) -- that is itself the finding.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from cognitrace.store.rebuild import rebuild_from_raw, replay_baseline


@dataclass
class Violation:
    code: str
    detail: str


def _i1_idempotency(conn: sqlite3.Connection) -> list[Violation]:
    rows = conn.execute(
        "SELECT raw_evidence_id, extractor_version_id, COUNT(*) c FROM extraction_jobs "
        "GROUP BY raw_evidence_id, extractor_version_id HAVING c > 1"
    ).fetchall()
    return [Violation("I1", f"raw_evidence_id={r['raw_evidence_id']} extractor_version_id="
                             f"{r['extractor_version_id']} processed {r['c']} times") for r in rows]


def _i2_one_live_record(conn: sqlite3.Connection) -> list[Violation]:
    rows = conn.execute(
        "SELECT subject_key, COUNT(*) c FROM memory_records WHERE valid_to IS NULL "
        "AND supersedable = 1 GROUP BY subject_key HAVING c > 1"
    ).fetchall()
    return [Violation("I2", f"subject_key={r['subject_key']!r} has {r['c']} live records") for r in rows]


def _i3_event_provenance(conn: sqlite3.Connection) -> list[Violation]:
    rows = conn.execute(
        "SELECT e.id FROM memory_events e LEFT JOIN raw_evidence r ON r.id = e.raw_evidence_id "
        "WHERE r.id IS NULL"
    ).fetchall()
    return [Violation("I3", f"memory_events.id={r['id']} has no raw_evidence") for r in rows]


def _i4_record_provenance(conn: sqlite3.Connection) -> list[Violation]:
    rows = conn.execute(
        "SELECT m.id FROM memory_records m LEFT JOIN memory_events e ON e.id = m.source_event_id "
        "WHERE e.id IS NULL"
    ).fetchall()
    return [Violation("I4", f"memory_records.id={r['id']} has no source memory_event") for r in rows]


def _i5_tri_temporal_order(conn: sqlite3.Connection) -> list[Violation]:
    out = []
    for r in conn.execute(
        "SELECT id, event_time_lo, event_time_hi FROM memory_events "
        "WHERE event_time_lo IS NOT NULL AND event_time_hi IS NOT NULL AND event_time_lo > event_time_hi"
    ).fetchall():
        out.append(Violation("I5", f"memory_events.id={r['id']} has event_time_lo>hi"))
    for r in conn.execute(
        "SELECT id, valid_from, valid_to FROM memory_records "
        "WHERE valid_to IS NOT NULL AND valid_from > valid_to"
    ).fetchall():
        out.append(Violation("I5", f"memory_records.id={r['id']} has valid_from>valid_to"))
    return out


def _i6_parity_gate(conn: sqlite3.Connection) -> list[Violation]:
    rows = conn.execute(
        "SELECT j.id AS job_id, ev.id AS extractor_version_id FROM extraction_jobs j "
        "JOIN extractor_versions ev ON ev.id = j.extractor_version_id "
        "WHERE ev.batch1_parity_verified_at IS NULL"
    ).fetchall()
    return [Violation("I6", f"extraction_jobs.id={r['job_id']} used unverified "
                             f"extractor_version_id={r['extractor_version_id']}") for r in rows]


def _i7_replay_closure(conn: sqlite3.Connection) -> list[Violation]:
    baseline = replay_baseline(conn)
    if baseline is None:
        return [Violation("I7", "no replay baseline recorded yet (call record_replay_baseline once)")]
    current = rebuild_from_raw(conn)
    if current != baseline:
        return [Violation("I7", f"replay diverged: baseline={baseline} current={current}")]
    return []


_CHECKS = (
    _i1_idempotency, _i2_one_live_record, _i3_event_provenance,
    _i4_record_provenance, _i5_tri_temporal_order, _i6_parity_gate, _i7_replay_closure,
)


def run_doctor(conn: sqlite3.Connection) -> list[Violation]:
    violations: list[Violation] = []
    for check in _CHECKS:
        violations.extend(check(conn))
    return violations
