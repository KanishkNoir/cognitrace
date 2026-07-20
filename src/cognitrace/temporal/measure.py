"""Measure how often a real question carries a query-side temporal anchor
(Sprint 4.5) -- gates the two-tier routing decision (A2) the same way
subject/measure.py (Sprint 4.2) gated the subject-key normalizer: a real
number before trusting routing machinery to be worth building.

Two rates, not one: `extracted` (a candidate span was found at all) and
`resolved` (it also turned into a usable interval). They diverge exactly
on anchor-relative phrases ("last week") without a `question_date` to
anchor them against -- real LoCoMo never supplies one, so on that dataset
`resolved` is extraction restricted to absolute dates / before-after
phrasing.
"""

from __future__ import annotations

from dataclasses import dataclass

from cognitrace.harness.schema import QAItem
from cognitrace.temporal.resolver import (
    extract_query_anchor,
    parse_anchor_date,
    resolve_query_anchor,
)


@dataclass
class QueryAnchorReport:
    total: int
    extracted: int
    resolved: int

    @property
    def extraction_rate(self) -> float | None:
        return self.extracted / self.total if self.total else None

    @property
    def resolution_rate(self) -> float | None:
        return self.resolved / self.total if self.total else None


def measure_questions(questions: list[QAItem]) -> QueryAnchorReport:
    total = extracted = resolved = 0
    for q in questions:
        total += 1
        if extract_query_anchor(q.question) is not None:
            extracted += 1
            anchor = parse_anchor_date(q.question_date)
            if resolve_query_anchor(q.question, anchor) is not None:
                resolved += 1
    return QueryAnchorReport(total=total, extracted=extracted, resolved=resolved)
