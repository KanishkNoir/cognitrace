"""Tests for the query-side anchor hit-rate measurement (Sprint 4.5).

Method mirrors subject/measure.py (Sprint 4.2): a real, tested measurement
against synthetic fixtures here; the number reported in SPRINT.md is a
separate one-off run against real LoCoMo, not asserted by these tests
(the real dataset isn't a fixture -- see the 4.5 finding for that number)."""

from __future__ import annotations

from cognitrace.harness.schema import QAItem
from cognitrace.temporal.measure import measure_questions


def _qa(question: str, question_date: str | None = None) -> QAItem:
    return QAItem(qid="q", question=question, answer="", category="temporal",
                  question_date=question_date, evidence_session_ids=[], evidence_turn_ids=[],
                  is_abstention=False)


def test_measure_counts_extracted_and_resolved_absolute_dates():
    questions = [
        _qa("What did John attend in March 2023?"),
        _qa("When did Caroline go to the support group?"),  # no date at all
    ]
    report = measure_questions(questions)
    assert report.total == 2
    assert report.extracted == 1
    assert report.resolved == 1


def test_measure_anchor_relative_extracted_but_unresolved_without_question_date():
    # "last week" is found by extraction but LoCoMo-style questions carry
    # no question_date -- extracted, not resolved.
    questions = [_qa("What did I do last week?")]
    report = measure_questions(questions)
    assert report.extracted == 1
    assert report.resolved == 0


def test_measure_anchor_relative_resolves_when_question_date_is_present():
    questions = [_qa("What did I do last week?", question_date="2023-05-10")]
    report = measure_questions(questions)
    assert report.extracted == 1
    assert report.resolved == 1


def test_measure_empty_list_reports_zero_not_error():
    report = measure_questions([])
    assert report.total == 0
    assert report.extraction_rate is None
    assert report.resolution_rate is None


def test_measure_rates_are_fractions_of_total():
    questions = [
        _qa("What did John attend in March 2023?"),
        _qa("When did Caroline go to the support group?"),
        _qa("How is Melanie feeling?"),
        _qa("What did I do last week?"),
    ]
    report = measure_questions(questions)
    assert report.total == 4
    assert report.extracted == 2  # March 2023, last week
    assert report.resolved == 1  # only March 2023 (no question_date given)
    assert report.extraction_rate == 0.5
    assert report.resolution_rate == 0.25
