"""Stage 4 of the phased-ranking pipeline (design_scaffold.md S9): token-
budget admission. A greedy score-density knapsack, not an LP solver --
~900 tokens doesn't justify the complexity. Every admit/evict decision is
returned with its reason, feeding the JSONL feature log (pipeline.py).
"""

from __future__ import annotations

from dataclasses import dataclass

from cognitrace.retrieval.pools import PoolHit
from cognitrace.retrieval.scorer import ScoredHit


@dataclass
class AdmissionDecision:
    record_id: int
    admitted: bool
    reason: str
    tokens: int


@dataclass
class BudgetResult:
    admitted: list[ScoredHit]
    decisions: list[AdmissionDecision]
    total_tokens: int


def _approx_tokens(text: str) -> int:
    """A word-count proxy, not a real tokenizer -- construction never calls
    an LLM (S8), so there is no API usage-token count to read at this
    layer. Good enough to prove the knapsack mechanism; replace before
    trusting it as a real budget (same posture as A6's latency floor).
    """
    return max(1, len(text.split()))


def admit_budget(
    scored: list[ScoredHit], hits_by_id: dict[int, PoolHit], *,
    token_budget: int = 900, reserved: dict[str, list[int]] | None = None,
) -> BudgetResult:
    """Reserved lanes (e.g. `{"chain_current": [...]}`) are admitted
    first, in caller-given order, before the greedy fill runs on whatever
    budget remains -- a lane the caller doesn't populate is simply absent,
    never invented here (no multi-hop/protected-category detector exists
    yet; see the Sprint 4.4 finding note). O(N log N), N = len(scored).
    """
    reserved = reserved or {}
    reserved_set = {rid for ids in reserved.values() for rid in ids}
    scored_by_id = {s.record_id: s for s in scored}

    decisions: list[AdmissionDecision] = []
    admitted: list[ScoredHit] = []
    remaining = token_budget
    processed_reserved: set[int] = set()

    for lane_name, ids in reserved.items():
        for rid in ids:
            # Skip if already processed in an earlier lane (first lane wins)
            if rid in processed_reserved:
                continue
            processed_reserved.add(rid)

            s = scored_by_id.get(rid)
            if s is None:
                decisions.append(AdmissionDecision(rid, False, f"reserved:{lane_name}:not_found", 0))
                continue
            tokens = _approx_tokens(hits_by_id[rid].value)
            if tokens <= remaining:
                admitted.append(s)
                decisions.append(AdmissionDecision(rid, True, f"reserved:{lane_name}", tokens))
                remaining -= tokens
            else:
                decisions.append(AdmissionDecision(rid, False, f"reserved:{lane_name}:over_budget", tokens))

    candidates = [s for s in scored if s.record_id not in reserved_set]
    ranked = sorted(
        candidates,
        key=lambda s: s.feature_score / _approx_tokens(hits_by_id[s.record_id].value),
        reverse=True,
    )
    for s in ranked:
        tokens = _approx_tokens(hits_by_id[s.record_id].value)
        if tokens <= remaining:
            admitted.append(s)
            decisions.append(AdmissionDecision(s.record_id, True, "budget_admitted", tokens))
            remaining -= tokens
        else:
            decisions.append(AdmissionDecision(s.record_id, False, "budget_exhausted", tokens))

    return BudgetResult(admitted=admitted, decisions=decisions, total_tokens=token_budget - remaining)
