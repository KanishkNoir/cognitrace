"""Deterministic temporal resolution (Phase 1a spine, S12).

The encoder never does date math (S12): it only classifies whether a
statement is dated/durative/past/planned. This module is the rule-based
resolver that turns a temporal expression + an anchor timestamp into a
concrete (event_time_lo, event_time_hi, grain) interval, entirely offline
and hand-rolled -- deliberately not `dateutil`/`arrow`, whose fuzzy-parsing
behavior isn't guaranteed stable release to release. The replay-
determinism contract (S6) needs the same (expression, anchor) pair to
resolve to the same interval forever; a third-party library upgrade must
never silently change a stored fact's meaning.

Conservative by design (mirrors reader.deterministic_match's own
discipline): an ambiguous expression resolves to None rather than a
guessed-wrong interval. Unresolved expressions fall through to the
second-phase soft-temporal feature (S12/4.5), never a fabricated hard
filter.
"""
