"""Subject-key normalization (Phase 1a spine, Sprint 4.2).

Deterministic, closed-vocab v0: resolves first/second-person pronouns to
the speaker/other-speaker in a conversation, and gives relationship-
possessive phrases ("my sister") a stable compound key. It deliberately
does NOT attempt real coreference resolution (bare third-person pronouns,
disambiguating which of two same-relation people is meant) -- that's a
genuinely hard problem the design defers to a learned linkage step only
after a pair-annotated eval exists (S1/4.3's "precision-first" rule).

`measure.py` turns that limitation into a number instead of a hope: real
LoCoMo dialogue often self-reveals identity ("my sister Nina") in the
same breath, which gives dataset-native ground truth for measuring how
often the naive key collides (two different people, one key) or splits
(one person, two keys) -- without needing human-annotated coreference
data, which doesn't exist yet.
"""
