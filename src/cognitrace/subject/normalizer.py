"""Deterministic subject-key normalizer v0 (Sprint 4.2).

`normalize_subject` resolves a single reference (a pronoun, name, or
relationship phrase as it appears in text) to a stable key, given who is
speaking and who the other party is. Two-person conversations only (the
datasets in scope are dyadic) -- a third-party mention that isn't the
speaker or the other party is either a proper name (used as-is) or a
relationship phrase (kept as a speaker-anchored compound key).

Every rule is a generic English pronoun/kinship pattern; none reference
any dataset's specific names or formatting.
"""

from __future__ import annotations

import re

_FIRST_PERSON = {"i", "me", "my", "myself", "mine"}
_SECOND_PERSON = {"you", "your", "yourself", "yours", "u", "ur"}
# Bare third-person pronouns are NOT resolved in v0 -- disambiguating which
# previously-mentioned person "she"/"they" refers to is real coreference
# resolution, deferred behind a pair-annotated eval (S1/4.3).
_THIRD_PERSON_AMBIGUOUS = {
    "he", "him", "his", "she", "her", "hers", "they", "them", "their", "theirs",
}

# Generic English kinship/relationship terms -- deliberately common nouns,
# not anything dataset-specific.
_RELATION_WORDS = (
    "mother", "mom", "father", "dad", "sister", "brother", "wife", "husband",
    "son", "daughter", "cousin", "aunt", "uncle", "grandmother", "grandfather",
    "grandma", "grandpa", "niece", "nephew", "friend", "boyfriend", "girlfriend",
    "partner", "fiancee", "fiance", "roommate", "neighbor", "neighbour",
    "boss", "colleague", "coworker", "co-worker", "therapist", "doctor",
)
_RELATION_PATTERN = "|".join(sorted(_RELATION_WORDS, key=len, reverse=True))

_POSSESSIVE_RELATION = re.compile(
    rf"\b(my|your|his|her|their|our)\s+({_RELATION_PATTERN})\b", re.IGNORECASE
)
# "my sister Nina" / "her friend Sarah" -- a relation phrase immediately
# followed by a capitalized name is a self-revealing co-mention: free,
# dataset-native ground truth that the phrase and the name are one entity.
_RELATION_WITH_NAME = re.compile(
    # Scoped (?i:...) keeps the possessive+relation case-insensitive while
    # leaving [A-Z][a-z]+ case-SENSITIVE -- a blanket IGNORECASE flag would
    # make [A-Z] match lowercase too, breaking the "must be capitalized"
    # proper-name signal this pattern depends on.
    rf"\b(?i:(my|your|his|her|their|our)\s+({_RELATION_PATTERN}))\s+([A-Z][a-z]+)\b"
)


def _canon(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip()).lower()


def normalize_subject(reference: str, speaker: str, other_speaker: str) -> str | None:
    """Resolve `reference` (as it literally appears in text) to a stable
    key. Returns None for bare third-person pronouns v0 doesn't resolve --
    callers must treat None as "unresolved", never as "no subject"."""
    key = reference.strip().lower()
    if not key:
        return None
    if key in _FIRST_PERSON:
        return _canon(speaker)
    if key in _SECOND_PERSON:
        return _canon(other_speaker)
    if key in _THIRD_PERSON_AMBIGUOUS:
        return None

    m = _POSSESSIVE_RELATION.fullmatch(key)
    if m:
        possessive, relation = m.groups()
        possessor = _resolve_possessive(possessive, speaker, other_speaker)
        if possessor is None:
            return None
        return f"{possessor}:{relation.lower()}"

    # A bare proper name / anything else: used as-is, case-normalized.
    return _canon(key)


def _resolve_possessive(possessive: str, speaker: str, other_speaker: str) -> str | None:
    p = possessive.lower()
    if p == "my":
        return _canon(speaker)
    if p == "your":
        return _canon(other_speaker)
    if p in ("his", "her", "their", "our"):
        return None  # v0: whose relation this is needs coreference, not resolved
    return None


def build_subject_key(reference: str, attribute: str, speaker: str, other_speaker: str) -> str | None:
    """Compose the compound key exact-key supersession (Sprint 4.3) runs
    on: the entity `normalize_subject` resolves, plus `attribute`. Two
    facts collapse onto the same live record iff they share both -- an
    unresolved reference (None) must never be papered over with a shared
    placeholder, since that would silently collide unrelated subjects."""
    subject = normalize_subject(reference, speaker, other_speaker)
    if subject is None:
        return None
    return f"{subject}:{attribute}"


def find_relation_name_comentions(text: str) -> list[tuple[str, str, str]]:
    """Find `(possessive, relation, name)` triples where a relationship
    phrase and a proper name co-occur in the same breath ("my sister
    Nina") -- the self-revealing signal `measure.py` uses as free ground
    truth, since no human-annotated coreference data exists yet."""
    return [
        (m.group(1).lower(), m.group(2).lower(), m.group(3))
        for m in _RELATION_WITH_NAME.finditer(text)
    ]


def find_relation_phrases(text: str) -> list[tuple[str, str]]:
    """Find `(possessive, relation)` pairs anywhere in text, including
    ones NOT accompanied by a name -- used to detect the split case (the
    same relation mentioned elsewhere without the name that was revealed
    for it elsewhere in the conversation)."""
    return [(m.group(1).lower(), m.group(2).lower()) for m in _POSSESSIVE_RELATION.finditer(text)]
