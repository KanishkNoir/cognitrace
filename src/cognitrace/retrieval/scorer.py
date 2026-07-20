"""Stage 3 of the phased-ranking pipeline (design_scaffold.md S9/S10):
extract interpretable per-candidate features and log them for the
eventual judgment-list-gated tuning. `feature_score` this sprint is
untuned identity on `rrf_score` -- S10 forbids inventing boosts/weights
before a versioned judgment list exists, so the other features are
computed and available to the JSONL feature log but do not yet move the
ranking themselves.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from cognitrace.retrieval.fusion import FusedHit
from cognitrace.retrieval.pools import PoolHit

_TOKEN = re.compile(r"[A-Za-z0-9]+")


@dataclass
class ScoredHit:
    record_id: int
    rrf_score: float
    is_live: bool
    recency_score: float
    subject_match: bool
    feature_score: float


def score_candidates(
    fused: list[FusedHit], hits_by_id: dict[int, PoolHit], query_text: str
) -> list[ScoredHit]:
    """O(N log N) for the recency sort, N = len(fused). `subject_match`
    checks every ':'-separated segment of `subject_key` against the query
    (a broad, logged-only signal); the pipeline's `chain_current` reserved
    lane uses a stricter entity-only check -- the two serve different
    purposes (this one is diagnostic, that one gates admission) and are
    deliberately not unified."""
    query_tokens = {t.lower() for t in _TOKEN.findall(query_text)}
    by_recency = sorted(fused, key=lambda f: hits_by_id[f.record_id].valid_from, reverse=True)
    recency_rank = {f.record_id: i for i, f in enumerate(by_recency)}
    n = len(fused) or 1
    out: list[ScoredHit] = []
    for f in fused:
        hit = hits_by_id[f.record_id]
        subject_match = any(part in query_tokens for part in hit.subject_key.split(":"))
        out.append(ScoredHit(
            record_id=f.record_id, rrf_score=f.rrf_score, is_live=hit.valid_to is None,
            recency_score=1.0 - recency_rank[f.record_id] / n, subject_match=subject_match,
            feature_score=f.rrf_score,
        ))
    return out
