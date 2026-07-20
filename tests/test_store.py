"""Store spine tests (Sprint 3): DDL invariants, ingest/DLQ, replay closure,
differential oracle, and anomaly resilience."""

from __future__ import annotations

import sqlite3

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cognitrace.store.doctor import run_doctor
from cognitrace.store.ingest import (
    ExtractedEvent,
    UnverifiedExtractorError,
    ingest_turn,
    list_dead_letters,
    register_extractor_version,
    verify_batch1_parity,
)
from cognitrace.store.rebuild import rebuild_from_raw, record_replay_baseline
from cognitrace.store.schema import open_store
from cognitrace.subject.normalizer import build_subject_key


def _fresh_store(tmp_path):
    return open_store(tmp_path / "conv.sqlite3")


def _verified_extractor(conn) -> int:
    ev_id = register_extractor_version(
        conn, model_sha="m1", onnx_sha="o1", onnxruntime_version="1.18.0",
        thresholds={"fact": 0.5}, calibration={"temp": 1.0},
    )
    verify_batch1_parity(conn, ev_id)
    return ev_id


def _extract_one(kind, subject_key, attribute, value):
    def _fn(content):
        return [ExtractedEvent(type=kind, subject_key=subject_key, attribute=attribute, value=value)]
    return _fn


# --- schema / PRAGMAs -------------------------------------------------------

def test_open_store_sets_wal_and_foreign_keys(tmp_path):
    conn = _fresh_store(tmp_path)
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


# --- ingestion: happy path, idempotency, DLQ --------------------------------

def test_ingest_refuses_unverified_extractor(tmp_path):
    conn = _fresh_store(tmp_path)
    ev_id = register_extractor_version(
        conn, model_sha="m1", onnx_sha="o1", onnxruntime_version="1.18.0",
        thresholds={}, calibration={},
    )
    with pytest.raises(UnverifiedExtractorError):
        ingest_turn(
            conn, extractor_version_id=ev_id, turn_id="t1", session_id="s1", role="user",
            content="I adopted a beagle named Biscuit.", mention_time="2023-05-05T13:00:00Z",
            extract_fn=_extract_one("FACT", "user:pet_name", "pet_name", "Biscuit"),
        )


def test_ingest_happy_path_creates_live_record(tmp_path):
    conn = _fresh_store(tmp_path)
    ev_id = _verified_extractor(conn)
    result = ingest_turn(
        conn, extractor_version_id=ev_id, turn_id="t1", session_id="s1", role="user",
        content="I adopted a beagle named Biscuit.", mention_time="2023-05-05T13:00:00Z",
        extract_fn=_extract_one("FACT", "user:pet_name", "pet_name", "Biscuit"),
    )
    assert result.status == "ok"
    assert len(result.event_ids) == 1
    row = conn.execute(
        "SELECT value, valid_to FROM memory_records WHERE subject_key = 'user:pet_name'"
    ).fetchone()
    assert row["value"] == "Biscuit" and row["valid_to"] is None
    violations_without_i7 = [v for v in run_doctor(conn) if v.code != "I7"]
    assert violations_without_i7 == []


def test_reingesting_same_turn_and_version_is_idempotent(tmp_path):
    conn = _fresh_store(tmp_path)
    ev_id = _verified_extractor(conn)
    fn = _extract_one("FACT", "user:pet_name", "pet_name", "Biscuit")
    r1 = ingest_turn(conn, extractor_version_id=ev_id, turn_id="t1", session_id="s1",
                      role="user", content="x", mention_time=None, extract_fn=fn)
    r2 = ingest_turn(conn, extractor_version_id=ev_id, turn_id="t1", session_id="s1",
                      role="user", content="x", mention_time=None, extract_fn=fn)
    assert r1.status == "ok"
    assert r2.status == "duplicate"
    assert r2.raw_evidence_id == r1.raw_evidence_id
    assert r2.event_ids == r1.event_ids
    n_events = conn.execute("SELECT COUNT(*) c FROM memory_events").fetchone()["c"]
    assert n_events == 1  # not doubled


def test_extraction_failure_dead_letters_without_corrupting_store(tmp_path):
    conn = _fresh_store(tmp_path)
    ev_id = _verified_extractor(conn)

    def _boom(content):
        raise ValueError("malformed turn")

    result = ingest_turn(
        conn, extractor_version_id=ev_id, turn_id="t_bad", session_id="s1", role="user",
        content="???", mention_time=None, extract_fn=_boom,
    )
    assert result.status == "dead_letter"
    assert result.event_ids == []
    dead = list_dead_letters(conn)
    assert len(dead) == 1 and dead[0]["turn_id"] == "t_bad"
    non_i7 = [v for v in run_doctor(conn) if v.code != "I7"]
    assert non_i7 == []  # a capture miss is visible, not a corruption


def test_second_fact_on_same_subject_key_supersedes_first(tmp_path):
    conn = _fresh_store(tmp_path)
    ev_id = _verified_extractor(conn)
    ingest_turn(conn, extractor_version_id=ev_id, turn_id="t1", session_id="s1", role="user",
                content="c1", mention_time="2023-05-05T13:00:00Z",
                extract_fn=_extract_one("FACT", "user:pet_name", "pet_name", "Biscuit"))
    ingest_turn(conn, extractor_version_id=ev_id, turn_id="t2", session_id="s1", role="user",
                content="c2", mention_time="2023-06-01T09:00:00Z",
                extract_fn=_extract_one("FACT", "user:pet_name", "pet_name", "Waffles"))
    rows = conn.execute(
        "SELECT value, valid_to, superseded_by FROM memory_records WHERE subject_key='user:pet_name' ORDER BY id"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0]["value"] == "Biscuit" and rows[0]["valid_to"] is not None
    assert rows[1]["value"] == "Waffles" and rows[1]["valid_to"] is None
    live = conn.execute(
        "SELECT COUNT(*) c FROM memory_records WHERE subject_key='user:pet_name' AND valid_to IS NULL"
    ).fetchone()["c"]
    assert live == 1  # I2 holds
    supersessions = conn.execute("SELECT COUNT(*) c FROM supersessions").fetchone()["c"]
    assert supersessions == 1


def test_event_dated_never_supersedes(tmp_path):
    conn = _fresh_store(tmp_path)
    ev_id = _verified_extractor(conn)
    ingest_turn(conn, extractor_version_id=ev_id, turn_id="t1", session_id="s1", role="user",
                content="c1", mention_time="2023-05-05T13:00:00Z",
                extract_fn=_extract_one("EVENT_DATED", "user:trip", "trip", "Paris"))
    ingest_turn(conn, extractor_version_id=ev_id, turn_id="t2", session_id="s1", role="user",
                content="c2", mention_time="2023-06-01T09:00:00Z",
                extract_fn=_extract_one("EVENT_DATED", "user:trip", "trip", "Rome"))
    rows = conn.execute(
        "SELECT value, valid_to FROM memory_records WHERE subject_key='user:trip'"
    ).fetchall()
    assert len(rows) == 2
    assert all(r["valid_to"] is None for r in rows)  # both remain live: episodic, not stateful


# --- Sprint 4.3: exact-key supersession through the real normalizer --------
# Unlike the tests above (which pass literal subject_key strings), these
# route through build_subject_key/normalize_subject -- proving the 4.2
# normalizer composes into a key that Sprint 3's exact-key supersession
# mechanism actually links on, without needing the (not-yet-built) encoder.

def _extract_via_normalizer(reference, attribute, value, speaker, other_speaker):
    def _fn(content):
        key = build_subject_key(reference, attribute, speaker, other_speaker)
        if key is None:
            return []  # conservative: never mint an event on a colliding placeholder
        return [ExtractedEvent(type="FACT", subject_key=key, attribute=attribute, value=value)]
    return _fn


def test_paraphrased_subject_references_collapse_and_supersede(tmp_path):
    conn = _fresh_store(tmp_path)
    ev_id = _verified_extractor(conn)
    ingest_turn(conn, extractor_version_id=ev_id, turn_id="t1", session_id="s1", role="Maya",
                content="I have a dog named Rex.", mention_time="2023-05-05T13:00:00Z",
                extract_fn=_extract_via_normalizer("I", "pet_name", "Rex", "Maya", "Rob"))
    ingest_turn(conn, extractor_version_id=ev_id, turn_id="t2", session_id="s1", role="Maya",
                content="My dog is actually named Max.", mention_time="2023-06-01T09:00:00Z",
                extract_fn=_extract_via_normalizer("my", "pet_name", "Max", "Maya", "Rob"))
    rows = conn.execute(
        "SELECT value, valid_to FROM memory_records WHERE subject_key='maya:pet_name' ORDER BY id"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0]["value"] == "Rex" and rows[0]["valid_to"] is not None
    assert rows[1]["value"] == "Max" and rows[1]["valid_to"] is None
    assert run_doctor(conn) == [] or [v for v in run_doctor(conn) if v.code != "I7"] == []


def test_different_attribute_via_normalizer_does_not_supersede(tmp_path):
    conn = _fresh_store(tmp_path)
    ev_id = _verified_extractor(conn)
    ingest_turn(conn, extractor_version_id=ev_id, turn_id="t1", session_id="s1", role="Maya",
                content="I have a dog named Rex.", mention_time=None,
                extract_fn=_extract_via_normalizer("I", "pet_name", "Rex", "Maya", "Rob"))
    ingest_turn(conn, extractor_version_id=ev_id, turn_id="t2", session_id="s1", role="Maya",
                content="I'm a teacher.", mention_time=None,
                extract_fn=_extract_via_normalizer("I", "job", "teacher", "Maya", "Rob"))
    live = conn.execute(
        "SELECT subject_key, value FROM memory_records WHERE valid_to IS NULL ORDER BY subject_key"
    ).fetchall()
    assert [(r["subject_key"], r["value"]) for r in live] == [
        ("maya:job", "teacher"), ("maya:pet_name", "Rex"),
    ]


def test_unresolved_subject_reference_mints_no_event(tmp_path):
    conn = _fresh_store(tmp_path)
    ev_id = _verified_extractor(conn)
    result = ingest_turn(
        conn, extractor_version_id=ev_id, turn_id="t1", session_id="s1", role="Maya",
        content="Her sister is a teacher.", mention_time=None,
        # "her sister" needs coreference the v0 normalizer doesn't do -- key is None.
        extract_fn=_extract_via_normalizer("her sister", "job", "teacher", "Maya", "Rob"),
    )
    assert result.status == "ok"
    assert result.event_ids == []
    n_records = conn.execute("SELECT COUNT(*) c FROM memory_records").fetchone()["c"]
    assert n_records == 0
    assert [v for v in run_doctor(conn) if v.code != "I7"] == []


# --- structural invariant I2 (partial unique index, not app logic) ----------

def test_one_live_record_is_a_write_time_constraint(tmp_path):
    conn = _fresh_store(tmp_path)
    ev_id = _verified_extractor(conn)
    conn.execute(
        "INSERT INTO raw_evidence (turn_id, session_id, role, content, content_sha256, recorded_at) "
        "VALUES ('t1','s1','user','x','sha','t')"
    )
    conn.execute(
        "INSERT INTO extraction_jobs (raw_evidence_id, extractor_version_id, idempotency_key, "
        "status, created_at) VALUES (1, ?, 'k1', 'ok', 't')", (ev_id,),
    )
    conn.execute(
        "INSERT INTO memory_events (extraction_job_id, raw_evidence_id, type, subject_key, "
        "attribute, value, polarity, asserts_change, raw_confidence, recorded_at) "
        "VALUES (1,1,'FACT','k','a','v1','asserted',0,1.0,'t')"
    )
    conn.execute(
        "INSERT INTO memory_records (source_event_id, type, subject_key, attribute, value, "
        "valid_from, valid_to, supersedable, recorded_at) "
        "VALUES (1,'FACT','k','a','v1','t',NULL,1,'t')"
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO memory_records (source_event_id, type, subject_key, attribute, value, "
            "valid_from, valid_to, supersedable, recorded_at) "
            "VALUES (1,'FACT','k','a','v2','t',NULL,1,'t')"
        )


# --- replay closure (I7) ----------------------------------------------------

def test_replay_from_raw_is_bit_identical(tmp_path):
    conn = _fresh_store(tmp_path)
    ev_id = _verified_extractor(conn)
    for i in range(5):
        ingest_turn(conn, extractor_version_id=ev_id, turn_id=f"t{i}", session_id="s1",
                    role="user", content=f"c{i}", mention_time=f"2023-05-0{i+1}T13:00:00Z",
                    extract_fn=_extract_one("FACT", f"user:attr{i % 2}", "a", f"v{i}"))
    baseline = record_replay_baseline(conn)
    assert rebuild_from_raw(conn) == baseline
    assert rebuild_from_raw(conn) == baseline  # repeated rebuilds are stable
    assert run_doctor(conn) == []


def test_doctor_reports_no_baseline_before_first_record(tmp_path):
    conn = _fresh_store(tmp_path)
    ev_id = _verified_extractor(conn)
    ingest_turn(conn, extractor_version_id=ev_id, turn_id="t1", session_id="s1", role="user",
                content="c", mention_time=None,
                extract_fn=_extract_one("FACT", "user:pet_name", "pet_name", "Biscuit"))
    violations = run_doctor(conn)
    assert any(v.code == "I7" and "no replay baseline" in v.detail for v in violations)


# --- I6: parity gate is enforced even against direct-SQL bypass ------------

def test_doctor_flags_jobs_against_unverified_extractor(tmp_path):
    conn = _fresh_store(tmp_path)
    ev_id = register_extractor_version(
        conn, model_sha="m1", onnx_sha="o1", onnxruntime_version="1.18.0",
        thresholds={}, calibration={},
    )
    conn.execute(
        "INSERT INTO raw_evidence (turn_id, session_id, role, content, content_sha256, recorded_at) "
        "VALUES ('t1','s1','user','x','sha','t')"
    )
    conn.execute(
        "INSERT INTO extraction_jobs (raw_evidence_id, extractor_version_id, idempotency_key, "
        "status, created_at) VALUES (1, ?, 'k1', 'ok', 't')", (ev_id,),
    )
    violations = [v for v in run_doctor(conn) if v.code == "I6"]
    assert len(violations) == 1


# --- anomaly: disk-full via max_page_count ----------------------------------

def test_disk_full_refuses_cleanly_without_corruption(tmp_path):
    conn = _fresh_store(tmp_path)
    ev_id = _verified_extractor(conn)
    conn.execute("PRAGMA max_page_count = 60")  # tiny cap: force SQLITE_FULL quickly
    hit_full = False
    for i in range(2000):
        try:
            ingest_turn(
                conn, extractor_version_id=ev_id, turn_id=f"t{i}", session_id="s1", role="user",
                content="x" * 500, mention_time=None,
                extract_fn=_extract_one("FACT", f"user:attr{i}", "a", "v" * 500),
            )
        except sqlite3.OperationalError as exc:
            assert "full" in str(exc).lower()
            hit_full = True
            break
    assert hit_full
    # doctor is read-only (no re-derive), so it runs fine even with the tiny
    # page cap still in effect -- a health check must never need write headroom.
    non_i7 = [v for v in run_doctor(conn) if v.code != "I7"]
    assert non_i7 == []  # refused past the cap, but nothing corrupted


# --- anomaly: an uncommitted writer crashing leaves nothing behind ----------

def test_uncommitted_transaction_leaves_no_trace_after_reopen(tmp_path):
    path = tmp_path / "conv.sqlite3"
    conn = open_store(path)
    ev_id = _verified_extractor(conn)
    ingest_turn(conn, extractor_version_id=ev_id, turn_id="t1", session_id="s1", role="user",
                content="c", mention_time=None,
                extract_fn=_extract_one("FACT", "user:pet_name", "pet_name", "Biscuit"))
    conn.close()

    crashed = sqlite3.connect(path)
    crashed.execute("BEGIN IMMEDIATE")
    crashed.execute(
        "INSERT INTO raw_evidence (turn_id, session_id, role, content, content_sha256, recorded_at) "
        "VALUES ('t_never_committed','s1','user','x','sha','t')"
    )
    crashed.close()  # simulated kill: no COMMIT ever issued

    reopened = open_store(path)
    row = reopened.execute(
        "SELECT COUNT(*) c FROM raw_evidence WHERE turn_id = 't_never_committed'"
    ).fetchone()
    assert row["c"] == 0
    non_i7 = [v for v in run_doctor(reopened) if v.code != "I7"]
    assert non_i7 == []


# --- differential oracle (3.5): dict-fold latest-wins vs the store --------

@settings(max_examples=50, deadline=None)
@given(
    st.lists(
        st.tuples(
            st.sampled_from(["k1", "k2", "k3"]),
            st.text(alphabet="abcdefg", min_size=1, max_size=5),
        ),
        min_size=1, max_size=30,
    )
)
def test_latest_wins_matches_dict_fold_reference(tmp_path_factory, events):
    tmp_path = tmp_path_factory.mktemp("oracle")
    conn = _fresh_store(tmp_path)
    ev_id = _verified_extractor(conn)
    for i, (key, value) in enumerate(events):
        ingest_turn(
            conn, extractor_version_id=ev_id, turn_id=f"t{i}", session_id="s1", role="user",
            content="x", mention_time=None,
            extract_fn=_extract_one("FACT", key, "a", value),
        )
    reference: dict[str, str] = {}
    for key, value in events:
        reference[key] = value
    live_rows = conn.execute(
        "SELECT subject_key, value FROM memory_records WHERE valid_to IS NULL"
    ).fetchall()
    live = {r["subject_key"]: r["value"] for r in live_rows}
    assert live == reference
    assert [v for v in run_doctor(conn) if v.code != "I7"] == []


# --- differential oracle: naive substring scan vs FTS5 membership ---------

def test_fts5_membership_matches_naive_substring_scan(tmp_path):
    conn = _fresh_store(tmp_path)
    ev_id = _verified_extractor(conn)
    values = ["Biscuit the beagle", "loves hiking", "started cello lessons", "Biscuit chewed shoes"]
    for i, v in enumerate(values):
        ingest_turn(conn, extractor_version_id=ev_id, turn_id=f"t{i}", session_id="s1",
                    role="user", content="x", mention_time=None,
                    extract_fn=_extract_one("FACT", f"k{i}", "a", v))
    term = "biscuit"
    naive_ids = {
        r["id"] for r in conn.execute("SELECT id, value FROM memory_records").fetchall()
        if term in r["value"].lower()
    }
    fts_ids = {
        r["rowid"] for r in conn.execute(
            "SELECT rowid FROM records_fts WHERE records_fts MATCH ?", (term,)
        ).fetchall()
    }
    assert fts_ids == naive_ids
    assert len(fts_ids) == 2
