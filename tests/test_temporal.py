"""Tests for the deterministic temporal resolver (Sprint 4.1, S12)."""

from __future__ import annotations

from datetime import date

from cognitrace.temporal.resolver import (
    extract_query_anchor,
    parse_anchor_date,
    resolve,
    resolve_query_anchor,
)

_ANCHOR = date(2023, 5, 10)  # a Wednesday


def _iso(interval):
    return interval.to_iso() if interval else None


# --- absolute dates (real LoCoMo temporal-category phrasing) --------------

def test_absolute_day_month_year():
    r = resolve("7 May 2023", _ANCHOR)
    assert r.grain == "day"
    assert _iso(r) == ("2023-05-07", "2023-05-07")


def test_absolute_day_month_year_with_ordinal():
    r = resolve("9th June 2023", _ANCHOR)
    assert _iso(r) == ("2023-06-09", "2023-06-09")


def test_absolute_month_day_year():
    r = resolve("June 9, 2023", _ANCHOR)
    assert _iso(r) == ("2023-06-09", "2023-06-09")


def test_absolute_month_year_no_day():
    r = resolve("June 2023", _ANCHOR)
    assert r.grain == "month"
    assert _iso(r) == ("2023-06-01", "2023-06-30")


def test_absolute_year_only():
    r = resolve("2022", _ANCHOR)
    assert r.grain == "year"
    assert _iso(r) == ("2022-01-01", "2022-12-31")


def test_year_with_leading_in():
    r = resolve("In 2013", _ANCHOR)
    assert _iso(r) == ("2013-01-01", "2013-12-31")


# --- weekday/unit relative to an absolute date ----------------------------

def test_the_sunday_before_a_date():
    # 25 May 2023 is a Thursday; the Sunday before it is 21 May 2023.
    r = resolve("The Sunday before 25 May 2023", _ANCHOR)
    assert r.grain == "day"
    assert _iso(r) == ("2023-05-21", "2023-05-21")


def test_the_week_before_a_date():
    # 9 June 2023 is a Friday, in the week of 5-11 June; the week before
    # runs 29 May - 4 June.
    r = resolve("the week before 9 June 2023", _ANCHOR)
    assert r.grain == "week"
    assert _iso(r) == ("2023-05-29", "2023-06-04")


def test_the_day_after_a_date():
    r = resolve("the day after 7 May 2023", _ANCHOR)
    assert _iso(r) == ("2023-05-08", "2023-05-08")


def test_the_weekend_before_a_date():
    # 17 July 2023 is a Monday (week of 17-23 July); the weekend before
    # that week is Sat 15 July - Sun 16 July.
    r = resolve("The weekend before 17 July 2023", _ANCHOR)
    assert r.grain == "week"
    assert _iso(r) == ("2023-07-15", "2023-07-16")


def test_word_number_count_before_a_date():
    # Two weekends before the weekend containing 17 July 2023 (Mon):
    # one weekend back is 15-16 July, two weekends back is 8-9 July.
    r = resolve("two weekends before 17 July 2023", _ANCHOR)
    assert _iso(r) == ("2023-07-08", "2023-07-09")


def test_unrecognized_unit_before_date_is_unresolved_not_absolute():
    # "fortnight" isn't a known unit -- must return None, never silently
    # match just the embedded "22 October 2023" and ignore the qualifier.
    assert resolve("two fortnights before 22 October 2023", _ANCHOR) is None


# --- anchor-relative expressions -----------------------------------------

def test_yesterday_today_tomorrow():
    assert _iso(resolve("yesterday", _ANCHOR)) == ("2023-05-09", "2023-05-09")
    assert _iso(resolve("today", _ANCHOR)) == ("2023-05-10", "2023-05-10")
    assert _iso(resolve("tomorrow", _ANCHOR)) == ("2023-05-11", "2023-05-11")


def test_last_week_and_next_week():
    # anchor 2023-05-10 (Wed) is in the week of 8-14 May; last week is 1-7 May.
    r = resolve("last week", _ANCHOR)
    assert r.grain == "week"
    assert _iso(r) == ("2023-05-01", "2023-05-07")
    r2 = resolve("next week", _ANCHOR)
    assert _iso(r2) == ("2023-05-15", "2023-05-21")


def test_this_month_last_month_next_month():
    assert _iso(resolve("this month", _ANCHOR)) == ("2023-05-01", "2023-05-31")
    assert _iso(resolve("last month", _ANCHOR)) == ("2023-04-01", "2023-04-30")
    assert _iso(resolve("next month", _ANCHOR)) == ("2023-06-01", "2023-06-30")


def test_last_year_next_year():
    assert _iso(resolve("last year", _ANCHOR)) == ("2022-01-01", "2022-12-31")
    assert _iso(resolve("next year", _ANCHOR)) == ("2024-01-01", "2024-12-31")


def test_n_units_ago_and_in_n_units():
    assert _iso(resolve("3 days ago", _ANCHOR)) == ("2023-05-07", "2023-05-07")
    assert _iso(resolve("in 2 weeks", _ANCHOR)) == ("2023-05-22", "2023-05-28")
    assert _iso(resolve("6 months ago", _ANCHOR)) == ("2022-11-01", "2022-11-30")
    assert _iso(resolve("in 1 year", _ANCHOR)) == ("2024-01-01", "2024-12-31")


def test_month_arithmetic_across_year_boundary():
    r = resolve("3 months ago", date(2023, 1, 15))
    assert _iso(r) == ("2022-10-01", "2022-10-31")


# --- conservatism: ambiguous/unrecognized text resolves to None ----------

def test_unresolvable_expression_returns_none():
    assert resolve("sometime last spring", _ANCHOR) is None
    assert resolve("a while back", _ANCHOR) is None
    assert resolve("", _ANCHOR) is None
    assert resolve("   ", _ANCHOR) is None


def test_bare_weekday_alone_is_not_resolved():
    # "Sunday" alone is ambiguous (which Sunday?) -- only "the Sunday
    # before/after <date>" is well-defined. Never guess.
    assert resolve("Sunday", _ANCHOR) is None


def test_invalid_calendar_date_returns_none():
    assert resolve("31 February 2023", _ANCHOR) is None


def test_month_year_with_comma():
    r = resolve("January, 2023", _ANCHOR)
    assert r is not None
    assert _iso(r) == ("2023-01-01", "2023-01-31")


# --- query-side anchor extraction (Sprint 4.5, A2) -------------------------
# Real LoCoMo question phrasing (measured: 37/321 = 11.5% of category-2
# questions carry an extractable, resolvable absolute-date anchor -- all
# absolute dates, zero anchor-relative, since LoCoMo never supplies a
# question_date to anchor "last week"/"yesterday" against).

def test_extract_query_anchor_finds_absolute_month_year():
    q = "What did John attend with his colleagues in March 2023?"
    assert extract_query_anchor(q) == "March 2023"


def test_extract_query_anchor_finds_absolute_day_month_year():
    q = "What did Audrey eat for dinner on October 24, 2023?"
    assert extract_query_anchor(q) == "October 24, 2023"


def test_extract_query_anchor_strips_trailing_question_mark():
    q = "Where was Tim in the week before 16 November 2023?"
    assert extract_query_anchor(q) == "the week before 16 November 2023"


def test_extract_query_anchor_finds_anchor_relative_phrase():
    assert extract_query_anchor("What did I do last week?") == "last week"


def test_extract_query_anchor_returns_none_for_a_pure_asking_for_date_question():
    # Real LoCoMo phrasing: no date is present, the question is asking for one.
    assert extract_query_anchor("When did Caroline go to the LGBTQ support group?") is None


def test_extract_query_anchor_does_not_match_a_bare_number_in_prose():
    # _ABS_YEAR stays fullmatch-only in resolve() itself (a stray 4-digit
    # number in prose is a false-positive risk) -- extraction must not
    # loosen that guard just because it now searches instead of fullmatches.
    assert extract_query_anchor("I have 2023 dollars saved") is None


def test_extract_query_anchor_prefers_the_before_after_qualifier_over_the_bare_date_within_it():
    # If a smaller absolute-date pattern matched first inside "the week
    # before 16 November 2023", it would grab just "16 November 2023" and
    # silently drop the qualifier that changes its meaning.
    q = "Where was Tim in the week before 16 November 2023?"
    span = extract_query_anchor(q)
    assert span is not None and "before" in span


# --- query-side resolution (extract + resolve, anchor-optional) ------------

def test_resolve_query_anchor_absolute_date_needs_no_anchor():
    r = resolve_query_anchor("What did John attend with his colleagues in March 2023?", None)
    assert r is not None
    assert _iso(r) == ("2023-03-01", "2023-03-31")


def test_resolve_query_anchor_anchor_relative_is_unresolved_without_a_real_anchor():
    # Conservative: an anchor-relative phrase with no question_date to
    # anchor against must never guess -- this matches real LoCoMo, which
    # never supplies question_date at all.
    assert resolve_query_anchor("What did I do last week?", None) is None


def test_resolve_query_anchor_anchor_relative_resolves_when_a_real_anchor_is_given():
    r = resolve_query_anchor("What did I do last week?", _ANCHOR)
    assert r is not None
    assert _iso(r) == _iso(resolve("last week", _ANCHOR))


def test_resolve_query_anchor_returns_none_when_nothing_extracted():
    assert resolve_query_anchor("How are you feeling about the trip?", _ANCHOR) is None


# --- question_date -> anchor date (lenient, fail-safe) ---------------------

def test_parse_anchor_date_accepts_iso_format():
    assert parse_anchor_date("2023-05-10") == date(2023, 5, 10)


def test_parse_anchor_date_accepts_an_iso_datetime_prefix():
    assert parse_anchor_date("2023-05-10T14:00:00Z") == date(2023, 5, 10)


def test_parse_anchor_date_accepts_the_resolver_absolute_date_formats():
    assert parse_anchor_date("10 May 2023") == date(2023, 5, 10)
    assert parse_anchor_date("May 10, 2023") == date(2023, 5, 10)


def test_parse_anchor_date_returns_none_for_unparseable_text():
    assert parse_anchor_date("sometime in the spring") is None


def test_parse_anchor_date_returns_none_for_none_or_empty():
    assert parse_anchor_date(None) is None
    assert parse_anchor_date("") is None
