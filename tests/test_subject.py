"""Tests for the subject-key normalizer v0 and its collision/split
measurement (Sprint 4.2)."""

from __future__ import annotations

from cognitrace.harness.schema import Session, Turn
from cognitrace.subject.measure import measure_conversation
from cognitrace.subject.normalizer import (
    find_relation_name_comentions,
    find_relation_phrases,
    normalize_subject,
)

# --- normalize_subject -------------------------------------------------

def test_first_person_resolves_to_speaker():
    assert normalize_subject("I", "Maya", "Rob") == "maya"
    assert normalize_subject("my", "Maya", "Rob") == "maya"
    assert normalize_subject("myself", "Maya", "Rob") == "maya"


def test_second_person_resolves_to_other_speaker():
    assert normalize_subject("you", "Maya", "Rob") == "rob"
    assert normalize_subject("your", "Maya", "Rob") == "rob"


def test_bare_third_person_pronoun_is_unresolved():
    assert normalize_subject("she", "Maya", "Rob") is None
    assert normalize_subject("they", "Maya", "Rob") is None
    assert normalize_subject("his", "Maya", "Rob") is None


def test_proper_name_used_as_is_case_normalized():
    assert normalize_subject("Nina", "Maya", "Rob") == "nina"
    assert normalize_subject("  Nina  ", "Maya", "Rob") == "nina"


def test_my_relation_resolves_to_speaker_anchored_compound_key():
    assert normalize_subject("my sister", "Maya", "Rob") == "maya:sister"


def test_your_relation_resolves_to_other_speaker_anchored_key():
    assert normalize_subject("your mother", "Maya", "Rob") == "rob:mother"


def test_third_person_possessive_relation_is_unresolved():
    # "her sister" -- whose relation this is needs coreference (unresolved v0).
    assert normalize_subject("her sister", "Maya", "Rob") is None
    assert normalize_subject("their friend", "Maya", "Rob") is None


def test_empty_reference_is_unresolved():
    assert normalize_subject("", "Maya", "Rob") is None
    assert normalize_subject("   ", "Maya", "Rob") is None


# --- self-revealing co-mention extraction ------------------------------

def test_finds_relation_name_comention():
    assert find_relation_name_comentions("My sister Nina is visiting.") == [("my", "sister", "Nina")]


def test_finds_bare_relation_phrase():
    assert ("my", "sister") in find_relation_phrases("I talked to my sister about it.")


def test_no_false_positive_on_unrelated_text():
    assert find_relation_name_comentions("I went for a run yesterday.") == []
    assert find_relation_phrases("I went for a run yesterday.") == []


# --- collision/split measurement (synthetic conversations) ------------

def _session(turns):
    return Session(session_id="s1", date="1 May 2023", turns=turns)


def test_measure_detects_a_split():
    # Maya's sister is named once ("Nina"), and mentioned bare elsewhere --
    # the naive scheme would give "maya:sister" and "nina" two different
    # keys for the same real person.
    sessions = [_session([
        Turn(role="Maya", content="My sister Nina just got a new job.", turn_id="t0"),
        Turn(role="Rob", content="That's great!", turn_id="t1"),
        Turn(role="Maya", content="I called my sister again today.", turn_id="t2"),
    ])]
    report = measure_conversation(sessions)
    assert report.pairs_with_reveal == 1
    assert report.splits == 1
    assert report.collisions == 0
    assert report.split_examples == [("maya", "sister", "Nina")]


def test_measure_detects_a_collision():
    # Two different names revealed for the same (speaker, relation) key.
    sessions = [_session([
        Turn(role="Maya", content="My sister Nina visited last week.", turn_id="t0"),
        Turn(role="Maya", content="My sister Clara called me today.", turn_id="t1"),
    ])]
    report = measure_conversation(sessions)
    assert report.collisions == 1
    assert report.splits == 0
    assert sorted(report.collision_examples[0][2]) == ["Clara", "Nina"]


def test_measure_clean_case_has_no_collision_or_split():
    # Named once, never mentioned bare elsewhere -- no split detected
    # (the method can't see a problem that never manifests as bare reuse).
    sessions = [_session([
        Turn(role="Maya", content="My sister Nina just got a new job.", turn_id="t0"),
    ])]
    report = measure_conversation(sessions)
    assert report.pairs_with_reveal == 1
    assert report.collisions == 0
    assert report.splits == 0


def test_measure_empty_conversation_reports_zero_not_error():
    report = measure_conversation([_session([Turn(role="Maya", content="Hi.", turn_id="t0")])])
    assert report.pairs_with_reveal == 0
    assert report.collision_rate is None
    assert report.split_rate is None
