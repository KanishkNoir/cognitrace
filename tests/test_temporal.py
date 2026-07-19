"""Tests for the deterministic temporal resolver (Sprint 4.1, S12)."""

from __future__ import annotations

from datetime import date

from cognitrace.temporal.resolver import resolve

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
