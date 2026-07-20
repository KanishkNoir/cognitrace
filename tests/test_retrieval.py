"""Tests for the Sprint 4.4 retrieval pipeline: pools -> RRF -> feature
scorer -> budget admission -> context assembly (design_scaffold.md S9)."""

from __future__ import annotations

from cognitrace.store.ingest import (
    ExtractedEvent,
    ingest_turn,
    register_extractor_version,
    verify_batch1_parity,
)
from cognitrace.store.schema import open_store


def _fresh_store(tmp_path):
    return open_store(tmp_path / "conv.sqlite3")


def _verified_extractor(conn):
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


# --- Stage 1: pools ----------------------------------------------------

from cognitrace.retrieval.pools import pool_lex, pool_prot, pool_valid


def test_pool_lex_matches_by_bm25(tmp_path):
    conn = _fresh_store(tmp_path)
    ev_id = _verified_extractor(conn)
    ingest_turn(conn, extractor_version_id=ev_id, turn_id="t1", session_id="s1", role="Maya",
                content="x", mention_time=None,
                extract_fn=_extract_one("FACT", "maya:pet_name", "pet_name", "Rex the beagle"))
    ingest_turn(conn, extractor_version_id=ev_id, turn_id="t2", session_id="s1", role="Maya",
                content="x", mention_time=None,
                extract_fn=_extract_one("FACT", "maya:job", "job", "teacher"))
    hits = pool_lex(conn, "beagle", limit=10)
    assert [h.value for h in hits] == ["Rex the beagle"]
    assert hits[0].turn_id == "t1" and hits[0].session_id == "s1"


def test_pool_lex_returns_empty_for_empty_query(tmp_path):
    conn = _fresh_store(tmp_path)
    assert pool_lex(conn, "", limit=10) == []


def test_pool_lex_includes_superseded_rows_per_A5(tmp_path):
    conn = _fresh_store(tmp_path)
    ev_id = _verified_extractor(conn)
    ingest_turn(conn, extractor_version_id=ev_id, turn_id="t1", session_id="s1", role="Maya",
                content="x", mention_time="2023-01-01T00:00:00Z",
                extract_fn=_extract_one("FACT", "maya:pet_name", "pet_name", "Rex"))
    ingest_turn(conn, extractor_version_id=ev_id, turn_id="t2", session_id="s1", role="Maya",
                content="x", mention_time="2023-02-01T00:00:00Z",
                extract_fn=_extract_one("FACT", "maya:pet_name", "pet_name", "Max"))
    hits = pool_lex(conn, "Rex Max", limit=10)
    assert {h.value for h in hits} == {"Rex", "Max"}


def test_pool_lex_ranks_by_bm25_relevance(tmp_path):
    conn = _fresh_store(tmp_path)
    ev_id = _verified_extractor(conn)
    # High relevance: query term is the dominant content
    ingest_turn(conn, extractor_version_id=ev_id, turn_id="t1", session_id="s1", role="Maya",
                content="x", mention_time=None,
                extract_fn=_extract_one("FACT", "maya:animal", "animal", "elephant"))
    # Low relevance: query term is one word among many others
    ingest_turn(conn, extractor_version_id=ev_id, turn_id="t2", session_id="s1", role="Maya",
                content="x", mention_time=None,
                extract_fn=_extract_one("FACT", "maya:story", "story", "Once upon a time in the jungle there lived an elephant with big ears"))
    hits = pool_lex(conn, "elephant", limit=10)
    assert len(hits) == 2
    # Higher-relevance hit (dominant term) should rank first
    assert hits[0].value == "elephant"
    assert hits[1].value == "Once upon a time in the jungle there lived an elephant with big ears"


def test_pool_valid_returns_only_live_records_most_recent_first(tmp_path):
    conn = _fresh_store(tmp_path)
    ev_id = _verified_extractor(conn)
    ingest_turn(conn, extractor_version_id=ev_id, turn_id="t1", session_id="s1", role="Maya",
                content="x", mention_time="2023-01-01T00:00:00Z",
                extract_fn=_extract_one("FACT", "maya:pet_name", "pet_name", "Rex"))
    ingest_turn(conn, extractor_version_id=ev_id, turn_id="t2", session_id="s1", role="Maya",
                content="x", mention_time="2023-02-01T00:00:00Z",
                extract_fn=_extract_one("FACT", "maya:pet_name", "pet_name", "Max"))
    ingest_turn(conn, extractor_version_id=ev_id, turn_id="t3", session_id="s1", role="Maya",
                content="x", mention_time="2023-01-15T00:00:00Z",
                extract_fn=_extract_one("FACT", "maya:job", "job", "teacher"))
    hits = pool_valid(conn, limit=10)
    assert [h.value for h in hits] == ["Max", "teacher"]
    assert all(h.valid_to is None for h in hits)


def test_pool_prot_is_stubbed_empty(tmp_path):
    conn = _fresh_store(tmp_path)
    assert pool_prot(conn, limit=10) == []


# --- Stage 2: RRF fusion -------------------------------------------------

from cognitrace.retrieval.fusion import reciprocal_rank_fusion
from cognitrace.retrieval.pools import PoolHit


def _hit(rid, **kw):
    base = dict(record_id=rid, subject_key="k", attribute="a", value="v",
                valid_from="t", valid_to=None, event_time_lo=None, event_time_hi=None,
                session_id="s1", turn_id=f"t{rid}")
    base.update(kw)
    return PoolHit(**base)


def test_rrf_combines_ranks_across_pools():
    pools = {
        "lex": [_hit(2), _hit(1), _hit(3)],
        "valid": [_hit(2), _hit(1)],
    }
    fused = reciprocal_rank_fusion(pools, k=60)
    ids = [f.record_id for f in fused]
    assert ids[0] == 2  # rank 1 in both pools -> beats 1 (rank 2 in both) and 3
    assert set(ids) == {1, 2, 3}


def test_rrf_pool_cap_preserves_small_pool_rescue():
    lex_hits = [_hit(i) for i in range(1, 101)]  # a big, generic pool
    valid_hits = [_hit(999)]  # one precise live record, absent from lex
    fused = reciprocal_rank_fusion({"lex": lex_hits, "valid": valid_hits}, pool_cap=24)
    fused_ids = {f.record_id for f in fused}
    assert 999 in fused_ids
    assert 100 not in fused_ids  # rank 100 in lex, beyond pool_cap=24 -> excluded


def test_rrf_tolerates_an_empty_pool():
    fused = reciprocal_rank_fusion({"lex": [_hit(1)], "prot": []})
    assert [f.record_id for f in fused] == [1]


def test_rrf_is_deterministic_on_score_ties():
    fused1 = reciprocal_rank_fusion({"lex": [_hit(2), _hit(1)]})
    fused2 = reciprocal_rank_fusion({"lex": [_hit(2), _hit(1)]})
    assert fused1 == fused2


# --- Stage 3: linear feature scorer --------------------------------------

from cognitrace.retrieval.fusion import FusedHit
from cognitrace.retrieval.scorer import score_candidates


def _score_hit(rid, subject_key="maya:pet_name", valid_to=None, valid_from="2023-01-01T00:00:00Z"):
    return PoolHit(record_id=rid, subject_key=subject_key, attribute="pet_name", value="Rex",
                   valid_from=valid_from, valid_to=valid_to, event_time_lo=None, event_time_hi=None,
                   session_id="s1", turn_id=f"t{rid}")


def test_score_candidates_flags_live_and_subject_match():
    hits_by_id = {1: _score_hit(1, valid_to=None)}
    fused = [FusedHit(record_id=1, rrf_score=0.5, pool_ranks={"lex": 1})]
    scored = score_candidates(fused, hits_by_id, "what is maya's pet's name")
    assert scored[0].is_live is True
    assert scored[0].subject_match is True
    assert scored[0].feature_score == 0.5  # untuned identity this sprint, per S10


def test_score_candidates_recency_favors_more_recent_valid_from():
    hits_by_id = {
        1: _score_hit(1, valid_from="2023-01-01T00:00:00Z"),
        2: _score_hit(2, valid_from="2023-06-01T00:00:00Z"),
    }
    fused = [
        FusedHit(record_id=1, rrf_score=0.1, pool_ranks={}),
        FusedHit(record_id=2, rrf_score=0.1, pool_ranks={}),
    ]
    scored = {s.record_id: s for s in score_candidates(fused, hits_by_id, "")}
    assert scored[2].recency_score > scored[1].recency_score


def test_score_candidates_no_subject_match_when_entity_absent_from_query():
    hits_by_id = {1: _score_hit(1, subject_key="maya:pet_name")}
    fused = [FusedHit(record_id=1, rrf_score=0.1, pool_ranks={})]
    scored = score_candidates(fused, hits_by_id, "what is rob's job")
    assert scored[0].subject_match is False


# --- Stage 4: budget admission --------------------------------------------

from cognitrace.retrieval.budget import admit_budget
from cognitrace.retrieval.scorer import ScoredHit


def _scored(rid, feature_score):
    return ScoredHit(record_id=rid, rrf_score=feature_score, is_live=True,
                      recency_score=1.0, subject_match=False, feature_score=feature_score)


def _hit_with_value(rid, value):
    return PoolHit(record_id=rid, subject_key="k", attribute="a", value=value,
                   valid_from="t", valid_to=None, event_time_lo=None, event_time_hi=None,
                   session_id="s1", turn_id=f"t{rid}")


def test_admit_budget_greedily_fills_by_score_density():
    # id1: 5 tokens, id2: 2 tokens, equal feature_score -> density favors id2.
    hits_by_id = {1: _hit_with_value(1, "one two three four five"),
                  2: _hit_with_value(2, "one two")}
    scored = [_scored(1, 1.0), _scored(2, 1.0)]
    result = admit_budget(scored, hits_by_id, token_budget=6)
    assert {s.record_id for s in result.admitted} == {2}
    reasons = {d.record_id: d.reason for d in result.decisions}
    assert reasons[1] == "budget_exhausted"
    assert reasons[2] == "budget_admitted"


def test_admit_budget_reserved_lane_admitted_before_greedy_fill():
    hits_by_id = {1: _hit_with_value(1, "a b c d e f g h"), 2: _hit_with_value(2, "x")}
    scored = [_scored(1, 100.0), _scored(2, 0.1)]  # id1 scores far higher
    result = admit_budget(scored, hits_by_id, token_budget=1, reserved={"chain_current": [2]})
    assert {s.record_id for s in result.admitted} == {2}
    reasons = {d.record_id: d.reason for d in result.decisions}
    assert reasons[2] == "reserved:chain_current"


def test_admit_budget_reserved_lane_over_budget_is_logged_not_crashed():
    hits_by_id = {1: _hit_with_value(1, "one two three four five six seven eight nine ten")}
    scored = [_scored(1, 1.0)]
    result = admit_budget(scored, hits_by_id, token_budget=2, reserved={"chain_current": [1]})
    assert result.admitted == []
    assert result.decisions[0].reason == "reserved:chain_current:over_budget"


def test_admit_budget_accepts_an_arbitrary_lane_name():
    # Proves the reserved-lane mechanism is generic: "protected"/"multi_hop"
    # aren't populated by the pipeline yet (no detector exists for either),
    # but the mechanism itself doesn't hardcode "chain_current" specially.
    hits_by_id = {1: _hit_with_value(1, "solo")}
    scored = [_scored(1, 1.0)]
    result = admit_budget(scored, hits_by_id, token_budget=5, reserved={"protected": [1]})
    assert result.decisions[0].reason == "reserved:protected"


def test_admit_budget_deduplicates_across_reserved_lanes():
    # When the same record_id appears in multiple lanes (or twice in one),
    # it is admitted and token-debited only once (first lane wins).
    hits_by_id = {
        1: _hit_with_value(1, "duplicate"),  # 1 token
    }
    scored = [_scored(1, 1.0)]
    result = admit_budget(
        scored, hits_by_id, token_budget=3,
        reserved={"chain_current": [1], "protected": [1]},  # id 1 in both lanes
    )
    # Should be admitted exactly once
    assert {s.record_id for s in result.admitted} == {1}
    # Total tokens should be 1, not 2
    assert result.total_tokens == 1
    # Should have exactly 1 decision for id 1, with the first lane's reason
    decisions_for_1 = [d for d in result.decisions if d.record_id == 1]
    assert len(decisions_for_1) == 1
    assert decisions_for_1[0].reason == "reserved:chain_current"
    assert decisions_for_1[0].admitted is True


def test_admit_budget_logs_reserved_id_not_in_scored_by_id():
    # A reserved id that isn't in scored_by_id produces a decision with
    # :not_found reason (not silently skipped).
    hits_by_id = {1: _hit_with_value(1, "found")}
    scored = [_scored(1, 1.0)]  # Only id 1 is scored; id 99 is not
    result = admit_budget(
        scored, hits_by_id, token_budget=10,
        reserved={"chain_current": [99, 1]},  # id 99 not in scored_by_id
    )
    # id 1 should be admitted, id 99 should not
    assert {s.record_id for s in result.admitted} == {1}
    # Both ids should have decisions
    reasons = {d.record_id: d.reason for d in result.decisions}
    assert reasons[99] == "reserved:chain_current:not_found"
    assert reasons[1] == "reserved:chain_current"
    # id 99 decision should have admitted=False and tokens=0
    decision_99 = [d for d in result.decisions if d.record_id == 99][0]
    assert decision_99.admitted is False
    assert decision_99.tokens == 0


# --- Stage 5: pipeline orchestration / evidence-recall@budget -------------

import io
import json

from cognitrace.harness.schema import QAItem
from cognitrace.retrieval.pipeline import index_meta, retrieve


def _qa(qid, question, evidence_turn_ids):
    return QAItem(qid=qid, question=question, answer="", category="single_hop",
                  question_date=None, evidence_session_ids=[], evidence_turn_ids=evidence_turn_ids,
                  is_abstention=False)


def test_retrieve_end_to_end_finds_live_fact_and_computes_recall(tmp_path):
    conn = _fresh_store(tmp_path)
    ev_id = _verified_extractor(conn)
    ingest_turn(conn, extractor_version_id=ev_id, turn_id="t1", session_id="s1", role="Maya",
                content="x", mention_time="2023-01-01T00:00:00Z",
                extract_fn=_extract_one("FACT", "maya:pet_name", "pet_name", "Rex the beagle"))
    result = retrieve(conn, _qa("q1", "what is Maya's dog named", ["t1"]))
    assert "Rex the beagle" in result.context
    assert result.recall_at_retrieved == 1.0
    assert result.recall_at_budget == 1.0
    assert result.index_meta["record_count"] == 1


def test_retrieve_evidence_recall_at_budget_drops_when_budget_too_small(tmp_path):
    conn = _fresh_store(tmp_path)
    ev_id = _verified_extractor(conn)
    ingest_turn(
        conn, extractor_version_id=ev_id, turn_id="t1", session_id="s1", role="Maya",
        content="x", mention_time="2023-01-01T00:00:00Z",
        extract_fn=_extract_one(
            "FACT", "maya:pet_name", "pet_name", "Rex the friendly beagle who loves parks",
        ),
    )
    result = retrieve(conn, _qa("q1", "what is Maya's dog named", ["t1"]), token_budget=1)
    assert result.recall_at_retrieved == 1.0
    assert result.recall_at_budget == 0.0  # the S9 diagnostic: the budget/stage3 narrowing cut the evidence


def test_retrieve_writes_one_jsonl_feature_log_line_per_call(tmp_path):
    conn = _fresh_store(tmp_path)
    ev_id = _verified_extractor(conn)
    ingest_turn(conn, extractor_version_id=ev_id, turn_id="t1", session_id="s1", role="Maya",
                content="x", mention_time=None,
                extract_fn=_extract_one("FACT", "maya:pet_name", "pet_name", "Rex"))
    log = io.StringIO()
    retrieve(conn, _qa("q1", "Rex", ["t1"]), feature_log=log)
    lines = log.getvalue().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["qid"] == "q1"
    assert "index_meta" in parsed and "decisions" in parsed
    # scorer.py computes rrf_score/is_live/recency_score/subject_match to be
    # logged for eventual judgment-list-gated tuning -- confirm they reach
    # the log line (not just computed and discarded).
    admitted_decisions = [d for d in parsed["decisions"] if d["admitted"]]
    assert len(admitted_decisions) == 1
    decision = admitted_decisions[0]
    assert decision["is_live"] is True
    # query "Rex" matches the record's value (subject_key is "maya:pet_name",
    # neither segment of which is a query token) -- subject_match is False.
    assert decision["subject_match"] is False
    assert isinstance(decision["rrf_score"], float)
    assert isinstance(decision["recency_score"], float)


def test_index_meta_reflects_record_count(tmp_path):
    conn = _fresh_store(tmp_path)
    assert index_meta(conn)["record_count"] == 0
    ev_id = _verified_extractor(conn)
    ingest_turn(conn, extractor_version_id=ev_id, turn_id="t1", session_id="s1", role="Maya",
                content="x", mention_time=None,
                extract_fn=_extract_one("FACT", "maya:pet_name", "pet_name", "Rex"))
    assert index_meta(conn)["record_count"] == 1
