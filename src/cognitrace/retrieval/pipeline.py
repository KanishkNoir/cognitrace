"""Orchestrates the Sprint 4.4 phased-ranking pipeline (design_scaffold.md
S9): pools -> RRF fusion -> feature scorer -> budget admission -> context
assembly. Also computes the evidence-recall@budget diagnostic -- the
instrument S9 calls out as "what would have caught the 5/14 delivery
failure": recall over the pre-budget fused candidates vs. recall over the
post-budget admitted set, so evidence loss between retrieval and what a
reader actually sees is visible, not just a lower headline number. Note the
gap this measures includes both the `stage3_top_n` truncation (candidates
past the top-N by fusion rank never reach scoring/admission at all) and the
budget admission step itself, not the token budget in isolation. On the
Sprint 4.5 hard-gate path (`_route_pools`'s "hard_gate" route), a third
contributor joins those two: `retrieved_turn_ids` is drawn only from
`pool_temporal_window`'s results, so `recall_at_retrieved`'s denominator
already reflects the window+lexical filter, not the full corpus -- the gap
`recall_at_budget` measures is narrower on that path, not directly
comparable to the soft path's stage3/budget-only gap.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from typing import TextIO

from cognitrace.harness.protocol import evidence_recall
from cognitrace.harness.schema import QAItem
from cognitrace.retrieval.budget import AdmissionDecision, admit_budget
from cognitrace.retrieval.fusion import reciprocal_rank_fusion
from cognitrace.retrieval.pools import PoolHit, pool_lex, pool_prot, pool_temporal_window, pool_valid
from cognitrace.retrieval.scorer import ScoredHit, score_candidates
from cognitrace.temporal.resolver import parse_anchor_date, resolve_query_anchor

_TOKEN = re.compile(r"[A-Za-z0-9]+")


@dataclass
class RetrievalResult:
    context: str
    cache_boundary_chars: int
    session_ids: list[str]
    turn_ids: list[str]
    retrieved_turn_ids: list[str]
    admitted_turn_ids: list[str]
    recall_at_retrieved: float | None
    recall_at_budget: float | None
    decisions: list[AdmissionDecision]
    index_meta: dict
    temporal_route: str


def index_meta(conn: sqlite3.Connection) -> dict:
    """Cheap lineage fingerprint for the store snapshot a retrieval ran
    against: schema_version + row-count + max id. Not a cryptographic
    guarantee (a full `derive.canonical_dump` hash would be); sufficient
    to detect "the index changed under you" at Phase-1 cardinality
    (hundreds of rows) -- same measured-not-assumed posture as A5;
    escalate to a canonical-dump hash if Sprint 5.6's non-regression
    replay needs stronger proof. O(N) in record count: SQLite keeps no
    cached row count, so COUNT(*) is a full scan of `memory_records` (or
    its smallest index); MAX(id) rides that same scan for free rather
    than being the reason this is cheap."""
    schema_version = conn.execute(
        "SELECT value FROM meta WHERE key = 'schema_version'"
    ).fetchone()
    counts = conn.execute(
        "SELECT COUNT(*) c, COALESCE(MAX(id), 0) m FROM memory_records"
    ).fetchone()
    return {
        "schema_version": schema_version["value"] if schema_version else None,
        "record_count": counts["c"],
        "max_record_id": counts["m"],
    }


def _u_curve_order(items: list[ScoredHit]) -> list[ScoredHit]:
    """Reader-facing context ordering: strongest evidence at both ends of
    the window, weakest in the middle -- the standard mitigation for
    "lost in the middle" (long-context recall degrades away from the
    edges). Deterministic given feature_score + record_id tie-break.
    O(N log N)."""
    ranked = sorted(items, key=lambda s: (-s.feature_score, s.record_id))
    slots: list[ScoredHit | None] = [None] * len(ranked)
    lo, hi = 0, len(ranked) - 1
    for i, item in enumerate(ranked):
        if i % 2 == 0:
            slots[lo] = item
            lo += 1
        else:
            slots[hi] = item
            hi -= 1
    return [s for s in slots if s is not None]


def _format_hit(hit: PoolHit) -> str:
    return f"[session {hit.session_id}] {hit.subject_key}.{hit.attribute} = {hit.value}"


def _assemble_context(admitted: list[ScoredHit], hits_by_id: dict[int, PoolHit]) -> tuple[str, int]:
    """Splits admitted evidence into a "stable" prefix (live/current facts)
    and a "variable" suffix (historical/lexical-only hits). Live records
    tend to be more stable across queries against the same store snapshot
    than lexical hits are (their pool membership doesn't depend on a text
    match), but the admitted subset and its ordering are still
    query-dependent -- both come from query-scored, budget-admitted
    candidates, and `_u_curve_order` sorts by `feature_score` (== rrf_score,
    itself query-dependent). This is NOT a byte-identical prefix a real
    cache_control integration could rely on without further work; it only
    marks where such a boundary *might* go. Each half is independently
    U-curve ordered. No cache_control header is emitted here -- Sprint
    4.7/4.8 wire this to an actual reader call. O(N log N)."""
    stable = _u_curve_order([s for s in admitted if s.is_live])
    variable = _u_curve_order([s for s in admitted if not s.is_live])
    stable_text = "\n".join(_format_hit(hits_by_id[s.record_id]) for s in stable)
    variable_text = "\n".join(_format_hit(hits_by_id[s.record_id]) for s in variable)
    if stable_text and variable_text:
        context = stable_text + "\n" + variable_text
    else:
        context = stable_text or variable_text
    return context, len(stable_text)


def _chain_current_lane(fused_ids: list[int], hits_by_id: dict[int, PoolHit], query_text: str) -> list[int]:
    """The one reserved lane with a deterministic signal to compute today:
    live records whose subject_key's entity segment is literally named in
    the query. `multi_hop`/`protected` lanes are accepted by admit_budget
    but not populated here -- no detector for either exists yet (Sprint
    4.4 finding, not silently invented). O(N)."""
    query_tokens = {t.lower() for t in _TOKEN.findall(query_text)}
    return [
        rid for rid in fused_ids
        if hits_by_id[rid].valid_to is None and hits_by_id[rid].subject_key.split(":")[0] in query_tokens
    ]


def _route_pools(
    conn: sqlite3.Connection, question: QAItem, *, pool_cap: int
) -> tuple[dict[str, list[PoolHit]], str]:
    """Two-tier temporal routing (A2, Sprint 4.5). Tier (a): if the
    question carries a rule-resolvable time anchor (`resolve_query_anchor`
    -- absolute dates always; anchor-relative phrases like "last week"
    only when `question.question_date` gives a real anchor to resolve
    against, per `parse_anchor_date`), the resolved interval becomes a
    first-phase SQL filter (`pool_temporal_window`) and nothing outside
    it is even a candidate -- a genuine gate, not a boost. An anchored
    query whose window matches nothing falls back to the normal fused
    path rather than returning empty (A2's explicit escape hatch) --
    logged as `"empty_window_fallback"` so a misroute (the extractor
    found a phrase, but the store has nothing there) is visible, not
    silently indistinguishable from "no anchor at all". Tier (b): no
    hard-resolvable anchor -- the "soft/inferred" case -- routes through
    the normal fused pools unchanged; A2's "second-phase feature plus a
    dedicated validity-pool arm in the RRF union" for this tier is
    `pool_valid`/`ScoredHit.is_live`, already shipped in Sprint 4.4, not
    new work here. O(1) plus one `pool_temporal_window` query when an
    anchor resolves.

    Trades away `pool_valid`'s lexical-match-free rescue when it fires
    (see `pool_temporal_window`'s docstring): 11.5% of real LoCoMo
    temporal-category questions (37/321) carry an extractable, resolvable
    anchor -- that is measured APPLICABILITY only, not benefit. Whether
    hard-gating those 37 improves or hurts evidence-recall versus the
    soft path is unmeasured until the harness (Sprint 4.7) runs both."""
    anchor = parse_anchor_date(question.question_date)
    interval = resolve_query_anchor(question.question, anchor)
    if interval is not None:
        lo, hi = interval.to_iso()
        hard_hits = pool_temporal_window(conn, question.question, lo, hi, limit=pool_cap)
        if hard_hits:
            return {"temporal": hard_hits}, "hard_gate"
        route = "empty_window_fallback"
    else:
        route = "soft"
    return {
        "lex": pool_lex(conn, question.question, limit=pool_cap),
        "valid": pool_valid(conn, limit=pool_cap),
        "prot": pool_prot(conn, limit=pool_cap),
    }, route


def retrieve(
    conn: sqlite3.Connection, question: QAItem, *,
    token_budget: int = 900, pool_cap: int = 24, stage3_top_n: int = 24,
    feature_log: TextIO | None = None,
) -> RetrievalResult:
    """The Sprint 4.4/4.5 phased-ranking pipeline (S9, A2): two-tier
    temporal routing (`_route_pools`) -> RRF -> feature scorer -> budget
    admission -> context assembly. Lexical-only this sprint (`pool_prot`
    is stubbed; A1 gates the dense arm behind Phase 1b). O(R log R) where
    R is the total row count touched by the routed pool queries --
    dominated by `pool_lex`'s FTS5 MATCH in the worst case (a broad
    OR-of-tokens query can match much of the corpus); `pool_valid`'s full
    O(R) scan of `memory_records` is the same order and can dominate in
    practice for narrow queries; the hard-gate tier adds one more FTS5
    query (`pool_temporal_window`), same order again. At Phase-1
    cardinality (hundreds of rows) the distinction doesn't change the
    bound.
    """
    pools, temporal_route = _route_pools(conn, question, pool_cap=pool_cap)
    hits_by_id: dict[int, PoolHit] = {}
    for hits in pools.values():
        for hit in hits:
            hits_by_id.setdefault(hit.record_id, hit)

    fused = reciprocal_rank_fusion(pools, pool_cap=pool_cap)
    retrieved_turn_ids = [hits_by_id[f.record_id].turn_id for f in fused]

    stage3 = fused[:stage3_top_n]
    scored = score_candidates(stage3, hits_by_id, question.question)

    reserved_ids = _chain_current_lane([f.record_id for f in stage3], hits_by_id, question.question)
    budget_result = admit_budget(
        scored, hits_by_id, token_budget=token_budget,
        reserved={"chain_current": reserved_ids} if reserved_ids else {},
    )

    context, cache_boundary = _assemble_context(budget_result.admitted, hits_by_id)
    admitted_turn_ids = [hits_by_id[s.record_id].turn_id for s in budget_result.admitted]
    session_ids = list(dict.fromkeys(hits_by_id[s.record_id].session_id for s in budget_result.admitted))

    gold = question.evidence_turn_ids
    recall_at_retrieved = evidence_recall(retrieved_turn_ids, gold, k=len(retrieved_turn_ids))
    recall_at_budget = evidence_recall(admitted_turn_ids, gold, k=len(admitted_turn_ids))

    meta = index_meta(conn)
    if feature_log is not None:
        # Carry each decision's scorer features (rrf_score/is_live/
        # recency_score/subject_match) into the log line -- scorer.py
        # computes these to be logged for the eventual judgment-list-gated
        # tuning; without this they'd be computed and silently discarded.
        # A `reserved:*:not_found` decision (Task 4) references a
        # record_id that was never in `scored` at all, so scored_by_id
        # lookup is guarded -- None is the honest value there, not a
        # KeyError.
        scored_by_id = {s.record_id: s for s in scored}
        feature_log.write(json.dumps({
            "qid": question.qid,
            "index_meta": meta,
            "temporal_route": temporal_route,
            "recall_at_retrieved": recall_at_retrieved,
            "recall_at_budget": recall_at_budget,
            "decisions": [
                {
                    "record_id": d.record_id, "admitted": d.admitted, "reason": d.reason, "tokens": d.tokens,
                    "rrf_score": scored_by_id[d.record_id].rrf_score if d.record_id in scored_by_id else None,
                    "is_live": scored_by_id[d.record_id].is_live if d.record_id in scored_by_id else None,
                    "recency_score": scored_by_id[d.record_id].recency_score if d.record_id in scored_by_id else None,
                    "subject_match": scored_by_id[d.record_id].subject_match if d.record_id in scored_by_id else None,
                }
                for d in budget_result.decisions
            ],
        }) + "\n")

    return RetrievalResult(
        context=context, cache_boundary_chars=cache_boundary, session_ids=session_ids,
        turn_ids=admitted_turn_ids, retrieved_turn_ids=retrieved_turn_ids,
        admitted_turn_ids=admitted_turn_ids, recall_at_retrieved=recall_at_retrieved,
        recall_at_budget=recall_at_budget, decisions=budget_result.decisions, index_meta=meta,
        temporal_route=temporal_route,
    )
