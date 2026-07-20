"""Stage 2 of the phased-ranking pipeline (design_scaffold.md S9/S10):
rank-based fusion across Stage-1 pools. Ports the well-known RRF pattern
(k=60) natively against this project's own schema -- there is no
`hybrid.py` file in this repo to copy from (it is CogniKernel's own,
external source); what's ported is the algorithm, not a file.
"""

from __future__ import annotations

from dataclasses import dataclass

from cognitrace.retrieval.pools import PoolHit


@dataclass
class FusedHit:
    record_id: int
    rrf_score: float
    pool_ranks: dict[str, int]


def reciprocal_rank_fusion(
    pools: dict[str, list[PoolHit]], *, k: int = 60, pool_cap: int = 24
) -> list[FusedHit]:
    """Rank-based fusion, not score-based -- pools have incomparable score
    distributions (BM25 vs. a plain SQL recency scan vs., eventually, a
    dense cosine) and RRF sidesteps calibrating them (S10: no per-pool
    multipliers before the judgment list). `pool_cap` bounds how many of
    each pool's own top hits enter the union -- the "type-restricted-pool
    rescue": a small precision pool (P-valid, a handful of live records)
    contributes its full top set regardless of how many hits a larger
    pool (P-lex) has, so it is never drowned out by union size alone.
    O(P*C log(P*C)) where P = pool count, C = pool_cap.
    """
    scores: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}
    seen: set[int] = set()
    for pool_name, hits in pools.items():
        for rank, hit in enumerate(hits[:pool_cap], start=1):
            scores[hit.record_id] = scores.get(hit.record_id, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(hit.record_id, {})[pool_name] = rank
            seen.add(hit.record_id)
    ordered = sorted(seen, key=lambda rid: (-scores[rid], rid))
    return [FusedHit(record_id=rid, rrf_score=scores[rid], pool_ranks=ranks[rid]) for rid in ordered]
