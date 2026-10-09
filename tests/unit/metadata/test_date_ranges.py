"""Tests for metadata.clock and metadata.date_ranges, with a frozen "today"."""

from datetime import date

import pytest

from metadata.clock import FixedClock, SystemClock
from metadata.date_ranges import DateRange, DateRangeResolver, DateSpecError

# A Sunday: the edge case for "this week" / "next week".
TODAY = date(2026, 10, 4)


@pytest.fixture
def resolve():
    return DateRangeResolver(FixedClock(TODAY)).resolve


def test_last_october_means_october_of_last_year(resolve):
    """ "tavaly októberben": the question that started this design."""
    r = resolve({"kind": "calendar", "year_offset": -1, "month": 10})

    assert r == DateRange(date(2025, 10, 1), date(2025, 10, 31))
    assert r.days == 31


def test_a_month_of_a_named_year_and_a_whole_year_and_a_day(resolve):
    assert resolve({"kind": "calendar", "year": 2022, "month": 3}) == DateRange(
        date(2022, 3, 1), date(2022, 3, 31)
    )
    assert resolve({"kind": "calendar", "year": 2022}) == DateRange(
        date(2022, 1, 1), date(2022, 12, 31)
    )
    assert resolve(
        {"kind": "calendar", "year": 2022, "month": 3, "day": 5}
    ) == DateRange(date(2022, 3, 5), date(2022, 3, 5))


def test_february_has_the_right_length_in_leap_and_ordinary_years(resolve):
    assert resolve({"kind": "calendar", "year": 2024, "month": 2}).days == 29
    assert resolve({"kind": "calendar", "year": 2023, "month": 2}).days == 28


def test_a_quarter(resolve):
    assert resolve({"kind": "calendar", "year": 2022, "quarter": 2}) == DateRange(
        date(2022, 4, 1), date(2022, 6, 30)
    )
    assert resolve({"kind": "calendar", "year_offset": 0, "quarter": 4}) == DateRange(
        date(2026, 10, 1), date(2026, 12, 31)
    )


def test_next_week_runs_monday_to_sunday_even_when_today_is_a_sunday(resolve):
    """Today is Sunday 2026-10-04: this week is Sep 28 - Oct 4, so next week starts tomorrow."""
    assert resolve({"kind": "relative", "unit": "week", "offset": 0}) == DateRange(
        date(2026, 9, 28), date(2026, 10, 4)
    )
    assert resolve({"kind": "relative", "unit": "week", "offset": 1}) == DateRange(
        date(2026, 10, 5), date(2026, 10, 11)
    )
    assert resolve({"kind": "relative", "unit": "week", "offset": -1}) == DateRange(
        date(2026, 9, 21), date(2026, 9, 27)
    )


def test_a_week_can_start_on_sunday_instead():
    resolve = DateRangeResolver(FixedClock(TODAY), week_starts_on=6).resolve

    assert resolve({"kind": "relative", "unit": "week", "offset": 0}) == DateRange(
        date(2026, 10, 4), date(2026, 10, 10)
    )


@pytest.mark.parametrize(
    ("unit", "offset", "expected"),
    [
        ("day", 0, (date(2026, 10, 4), date(2026, 10, 4))),
        ("day", -1, (date(2026, 10, 3), date(2026, 10, 3))),
        ("month", 0, (date(2026, 10, 1), date(2026, 10, 31))),
        ("month", -1, (date(2026, 9, 1), date(2026, 9, 30))),
        ("month", 3, (date(2027, 1, 1), date(2027, 1, 31))),  # crosses the year
        ("quarter", 0, (date(2026, 10, 1), date(2026, 12, 31))),
        ("quarter", -1, (date(2026, 7, 1), date(2026, 9, 30))),
        ("year", -1, (date(2025, 1, 1), date(2025, 12, 31))),
    ],
)
def test_relative_units(resolve, unit, offset, expected):
    assert resolve({"kind": "relative", "unit": unit, "offset": offset}) == DateRange(
        *expected
    )


def test_a_month_offset_does_not_overflow_on_a_short_month():
    resolve = DateRangeResolver(FixedClock(date(2026, 3, 31))).resolve

    assert resolve({"kind": "relative", "unit": "month", "offset": -1}) == DateRange(
        date(2026, 2, 1), date(2026, 2, 28)
    )


def test_a_rolling_window_includes_today(resolve):
    assert resolve({"kind": "rolling", "unit": "day", "count": 30}) == DateRange(
        date(2026, 9, 5), date(2026, 10, 4)
    )
    assert resolve(
        {"kind": "rolling", "unit": "day", "count": 7, "direction": "future"}
    ) == DateRange(date(2026, 10, 4), date(2026, 10, 10))
    assert resolve({"kind": "rolling", "unit": "week", "count": 2}).days == 14
    assert resolve({"kind": "rolling", "unit": "month", "count": 3}) == DateRange(
        date(2026, 7, 5), date(2026, 10, 4)
    )


def test_between_runs_from_the_start_of_one_to_the_end_of_the_other(resolve):
    """ "a 2020 és 2024 közötti időszakban"."""
    r = resolve(
        {
            "kind": "between",
            "from": {"kind": "calendar", "year": 2020},
            "to": {"kind": "calendar", "year": 2024},
        }
    )

    assert r == DateRange(date(2020, 1, 1), date(2024, 12, 31))


def test_between_two_months(resolve):
    r = resolve(
        {
            "kind": "between",
            "from": {"kind": "calendar", "year": 2022, "month": 11},
            "to": {"kind": "calendar", "year": 2023, "month": 2},
        }
    )

    assert r == DateRange(date(2022, 11, 1), date(2023, 2, 28))


def test_an_absolute_range(resolve):
    assert resolve(
        {"kind": "absolute", "start": "2025-10-01", "end": "2025-10-31"}
    ) == DateRange(date(2025, 10, 1), date(2025, 10, 31))


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("last october", "must be an object"),
        ({}, "unknown date spec kind"),
        ({"kind": "yearly"}, "unknown date spec kind"),
        ({"kind": "calendar", "month": 3}, "exactly one of"),
        ({"kind": "calendar", "year": 2022, "year_offset": -1}, "exactly one of"),
        ({"kind": "calendar", "year": 2022, "month": 13}, "month must be"),
        ({"kind": "calendar", "year": 2022, "month": 3, "quarter": 1}, "not both"),
        ({"kind": "calendar", "year": 2022, "quarter": 5}, "quarter must be"),
        ({"kind": "calendar", "year": 2022, "day": 4}, "needs a month"),
        ({"kind": "calendar", "year": 2023, "month": 2, "day": 30}, "impossible date"),
        ({"kind": "relative", "unit": "week"}, "integer 'offset'"),
        ({"kind": "relative", "unit": "fortnight", "offset": 1}, "unknown unit"),
        ({"kind": "rolling", "unit": "day", "count": 0}, "positive integer"),
        (
            {"kind": "rolling", "unit": "day", "count": 3, "direction": "sideways"},
            "past' or 'future",
        ),
        (
            {"kind": "between", "from": {"kind": "calendar", "year": 2022}},
            "'from' and 'to'",
        ),
        (
            {
                "kind": "between",
                "from": {"kind": "calendar", "year": 2024},
                "to": {"kind": "calendar", "year": 2020},
            },
            "ends before it starts",
        ),
        (
            {"kind": "absolute", "start": "2025-10-31", "end": "2025-10-01"},
            "ends before it starts",
        ),
        (
            {"kind": "absolute", "start": "31/10/2025", "end": "2025-11-01"},
            "impossible date",
        ),
        (
            {"kind": "absolute", "start": "1000-01-01", "end": "2025-01-01"},
            "more than 100 years",
        ),
    ],
)
def test_a_malformed_or_impossible_spec_is_rejected_with_a_clear_message(
    resolve, spec, message
):
    with pytest.raises(DateSpecError, match=message):
        resolve(spec)


def test_the_week_start_must_be_a_weekday_number():
    with pytest.raises(ValueError, match="week_starts_on"):
        DateRangeResolver(FixedClock(TODAY), week_starts_on=7)


def test_the_clock_is_what_decides_today():
    r = DateRangeResolver(FixedClock(date(2027, 1, 15))).resolve(
        {"kind": "relative", "unit": "year", "offset": -1}
    )

    assert r == DateRange(date(2026, 1, 1), date(2026, 12, 31))


def test_the_system_clock_returns_a_date_in_its_zone():
    assert isinstance(SystemClock("Europe/Budapest").today(), date)
    assert isinstance(SystemClock("UTC").today(), date)
