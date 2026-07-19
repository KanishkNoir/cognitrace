"""Rule-based temporal expression resolver (S12).

`resolve(expression, anchor)` maps a natural-language temporal expression
plus an anchor date (the mention's own session timestamp) to a concrete
interval + grain. Every rule here is a generic English time-expression
pattern (absolute dates, weekday-relative-to-a-date, anchor-relative
phrases like "last week") -- none reference any specific dataset's
formatting, which is the Maximem-scandal firewall applied to dates: a
rule that wouldn't make sense on an unseen dataset doesn't belong here.

Unresolved/ambiguous expressions return None rather than a guessed
interval -- the same conservatism as `reader.deterministic_match`: a
rule that can't be confident must never fabricate a wrong answer.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta

_MONTHS = {
    name.lower(): i
    for i, name in enumerate(
        ["", "January", "February", "March", "April", "May", "June", "July",
         "August", "September", "October", "November", "December"]
    )
    if name
}
_MONTH_ABBR = {
    name.lower(): i
    for i, name in enumerate(
        ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )
    if name
}
_ALL_MONTHS = {**_MONTHS, **_MONTH_ABBR}
_MONTH_PATTERN = "|".join(sorted(_ALL_MONTHS, key=len, reverse=True))

_WEEKDAYS = {
    name.lower(): i
    for i, name in enumerate(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )
}

_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def _parse_count(text: str | None) -> int:
    if not text:
        return 1
    key = text.lower()
    return _WORD_NUMBERS[key] if key in _WORD_NUMBERS else int(key)


@dataclass
class TemporalInterval:
    lo: date
    hi: date
    grain: str  # "day" | "week" | "month" | "year"

    def to_iso(self) -> tuple[str, str]:
        return self.lo.isoformat(), self.hi.isoformat()


def _month_end(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def _day_interval(d: date) -> TemporalInterval:
    return TemporalInterval(d, d, "day")


def _week_interval_containing(d: date) -> TemporalInterval:
    monday = d - timedelta(days=d.weekday())
    return TemporalInterval(monday, monday + timedelta(days=6), "week")


def _month_interval(year: int, month: int) -> TemporalInterval:
    return TemporalInterval(date(year, month, 1), date(year, month, _month_end(year, month)), "month")


def _year_interval(year: int) -> TemporalInterval:
    return TemporalInterval(date(year, 1, 1), date(year, 12, 31), "year")


# --- absolute date parsing (no external library: stable-forever parsing) ---

_ABS_DAY_MONTH_YEAR = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s*({_MONTH_PATTERN})\.?,?\s*(\d{{4}})\b", re.IGNORECASE
)
_ABS_MONTH_DAY_YEAR = re.compile(
    rf"\b({_MONTH_PATTERN})\.?\s*(\d{{1,2}})(?:st|nd|rd|th)?,?\s*(\d{{4}})\b", re.IGNORECASE
)
_ABS_MONTH_YEAR = re.compile(rf"\b({_MONTH_PATTERN})\.?,?\s*(\d{{4}})\b", re.IGNORECASE)
_ABS_YEAR = re.compile(r"^(?:in\s+)?(\d{4})$", re.IGNORECASE)


def _parse_absolute_date(text: str) -> date | None:
    """The most specific match wins: day+month+year, then month+year alone
    (handled by the caller as a month-grain interval, not here)."""
    m = _ABS_DAY_MONTH_YEAR.search(text)
    if m:
        day, month_name, year = m.groups()
        month = _ALL_MONTHS[month_name.lower()]
        try:
            return date(int(year), month, int(day))
        except ValueError:
            return None
    m = _ABS_MONTH_DAY_YEAR.search(text)
    if m:
        month_name, day, year = m.groups()
        month = _ALL_MONTHS[month_name.lower()]
        try:
            return date(int(year), month, int(day))
        except ValueError:
            return None
    return None


def _parse_absolute(text: str) -> TemporalInterval | None:
    # A structural day-month-year match that turns out invalid (e.g. "31
    # February") must NOT silently downgrade to a month-year guess -- that
    # would fabricate a different, unrequested interval.
    m = _ABS_DAY_MONTH_YEAR.search(text)
    if m:
        day, month_name, year = m.groups()
        try:
            return _day_interval(date(int(year), _ALL_MONTHS[month_name.lower()], int(day)))
        except ValueError:
            return None
    m = _ABS_MONTH_DAY_YEAR.search(text)
    if m:
        month_name, day, year = m.groups()
        try:
            return _day_interval(date(int(year), _ALL_MONTHS[month_name.lower()], int(day)))
        except ValueError:
            return None
    m = _ABS_MONTH_YEAR.search(text)
    if m:
        month_name, year = m.groups()
        return _month_interval(int(year), _ALL_MONTHS[month_name.lower()])
    m = _ABS_YEAR.fullmatch(text.strip())
    if m:
        return _year_interval(int(m.group(1)))
    return None


# --- weekday / day / week / weekend relative to an absolute date ---------
#
# One unified pattern for every "[the] [N] <unit> before/after <date>" shape
# -- weekday names, day, week, and weekend are all ordinary, dataset-
# independent English units, including a word-number count ("two weekends
# before X"). Deliberately requires resolving to a full interval or nothing:
# `resolve()` never falls through to plain absolute-date parsing once a
# before/after qualifier is present, so an unrecognized unit ("two
# fortnights before X") returns None rather than silently matching just
# the embedded date and ignoring the qualifier that changes its meaning.

_WEEKDAY_PATTERN = "|".join(_WEEKDAYS)
_COUNT_PATTERN = r"\d+|" + "|".join(_WORD_NUMBERS)
_REL_UNIT_PATTERN = f"{_WEEKDAY_PATTERN}|day|week|weekend"

_REL_TO_DATE = re.compile(
    rf"\b(?:the\s+)?(?:({_COUNT_PATTERN})\s+)?({_REL_UNIT_PATTERN})s?\s+(before|after)\s+(.+)$",
    re.IGNORECASE,
)


def _nearest_weekday(anchor: date, weekday: int, direction: str) -> date:
    if direction == "before":
        delta = (anchor.weekday() - weekday) % 7
        delta = 7 if delta == 0 else delta
        return anchor - timedelta(days=delta)
    delta = (weekday - anchor.weekday()) % 7
    delta = 7 if delta == 0 else delta
    return anchor + timedelta(days=delta)


def _parse_relative_to_date(text: str) -> TemporalInterval | None:
    m = _REL_TO_DATE.search(text)
    if not m:
        return None
    count_s, unit, direction, rest = m.groups()
    base = _parse_absolute_date(rest)
    if base is None:
        return None
    count = _parse_count(count_s)
    sign = -1 if direction.lower() == "before" else 1
    unit_l = unit.lower()

    if unit_l in _WEEKDAYS:
        if count != 1:
            return None  # "two Sundays before X" has no single well-defined day
        d = _nearest_weekday(base, _WEEKDAYS[unit_l], direction.lower())
        return _day_interval(d)
    if unit_l == "day":
        return _day_interval(base + timedelta(days=sign * count))
    if unit_l == "week":
        target = base + timedelta(days=7 * sign * count)
        return _week_interval_containing(target)
    if unit_l == "weekend":
        base_monday = base - timedelta(days=base.weekday())
        target_monday = base_monday + timedelta(days=7 * sign * count)
        return TemporalInterval(target_monday + timedelta(days=5), target_monday + timedelta(days=6), "week")
    return None


# --- anchor-relative expressions ("yesterday", "last week", "3 days ago") --

_SIMPLE_ANCHOR = {
    "today": (0, "day"), "yesterday": (-1, "day"), "tomorrow": (1, "day"),
}
_REL_WORD_UNIT = re.compile(r"\b(this|last|next)\s+(day|week|month|year)\b", re.IGNORECASE)
_N_UNITS_AGO = re.compile(r"\b(\d+)\s+(day|week|month|year)s?\s+ago\b", re.IGNORECASE)
_IN_N_UNITS = re.compile(r"\bin\s+(\d+)\s+(day|week|month|year)s?\b", re.IGNORECASE)


def _add_months(d: date, n: int) -> date:
    month0 = d.month - 1 + n
    year = d.year + month0 // 12
    month = month0 % 12 + 1
    day = min(d.day, _month_end(year, month))
    return date(year, month, day)


def _parse_anchor_relative(text: str, anchor: date) -> TemporalInterval | None:
    key = text.strip().lower()
    if key in _SIMPLE_ANCHOR:
        offset, grain = _SIMPLE_ANCHOR[key]
        return _day_interval(anchor + timedelta(days=offset))

    m = _REL_WORD_UNIT.fullmatch(key)
    if m:
        which, unit = m.groups()
        sign = {"last": -1, "this": 0, "next": 1}[which.lower()]
        if unit.lower() == "day":
            return _day_interval(anchor + timedelta(days=sign))
        if unit.lower() == "week":
            target = anchor + timedelta(days=7 * sign)
            return _week_interval_containing(target)
        if unit.lower() == "month":
            target = _add_months(anchor, sign)
            return _month_interval(target.year, target.month)
        if unit.lower() == "year":
            return _year_interval(anchor.year + sign)

    m = _N_UNITS_AGO.fullmatch(key) or _IN_N_UNITS.fullmatch(key)
    if m:
        n, unit = m.groups()
        n = int(n)
        sign = -1 if "ago" in key else 1
        if unit.lower() == "day":
            return _day_interval(anchor + timedelta(days=sign * n))
        if unit.lower() == "week":
            target = anchor + timedelta(days=7 * sign * n)
            return _week_interval_containing(target)
        if unit.lower() == "month":
            target = _add_months(anchor, sign * n)
            return _month_interval(target.year, target.month)
        if unit.lower() == "year":
            return _year_interval(anchor.year + sign * n)
    return None


_BEFORE_AFTER = re.compile(r"\b(?:before|after)\b", re.IGNORECASE)


def resolve(expression: str, anchor: date) -> TemporalInterval | None:
    """Resolve one temporal expression against an anchor date. Tries, in
    order: a date-relative phrase ("the Sunday before X"); an absolute
    date/month/year; an anchor-relative phrase ("last week", "3 days
    ago"). Returns None if nothing matches confidently -- callers must
    treat that as "unresolved", not as "no time reference exists"."""
    text = expression.strip()
    if not text:
        return None
    # Once a before/after qualifier is present, resolve fully via that path
    # or not at all -- never fall through to plain absolute-date parsing,
    # which uses substring search and would otherwise match an embedded
    # date and silently ignore a qualifier ("the weekend before X") it
    # doesn't know how to handle.
    if _BEFORE_AFTER.search(text):
        return _parse_relative_to_date(text)
    interval = _parse_absolute(text)
    if interval is not None:
        return interval
    return _parse_anchor_relative(text, anchor)
