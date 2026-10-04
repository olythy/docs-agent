"""Turning a symbolic date description into a concrete, inclusive date range.

A question says "last October", "next week", "between 2020 and 2024". An LLM
planner may *describe* that in a small closed grammar, but the arithmetic is done
here, deterministically, from an injected :class:`metadata.clock.Clock`: models
are unreliable at calendar arithmetic (week boundaries, month lengths, "last
year"), and a wrong range returns a clean, confident, wrong count.

The grammar (each spec is a dict, so it travels as JSON):

    {"kind": "absolute", "start": "2025-10-01", "end": "2025-10-31"}
    {"kind": "calendar", "year": 2022, "month": 3}            # 2022 March
    {"kind": "calendar", "year_offset": -1, "month": 10}      # October of last year
    {"kind": "calendar", "year": 2022, "quarter": 2}
    {"kind": "calendar", "year": 2022, "month": 3, "day": 5}
    {"kind": "relative", "unit": "week", "offset": 1}         # next week
    {"kind": "rolling", "unit": "day", "count": 30}           # the last 30 days
    {"kind": "between", "from": <spec>, "to": <spec>}         # from the start of one to the end of the other

Key exports:
    DateRange         -- An inclusive range of days.
    DateRangeResolver -- Resolves a spec to a DateRange.
    DateSpecError     -- A spec that is malformed or impossible.
"""

import calendar
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from metadata.clock import Clock

#: A guard against a typo turning into a century-wide range.
_MAX_YEARS = 100


class DateSpecError(ValueError):
    """A date spec that is malformed, impossible, or out of bounds."""


@dataclass(frozen=True)
class DateRange:
    """An inclusive range of calendar days.

    Attributes:
        start: First day.
        end: Last day (never before ``start``).
    """

    start: date
    end: date

    @property
    def days(self) -> int:
        """Number of days in the range, both ends included."""
        return (self.end - self.start).days + 1


def _month_start(year: int, month: int) -> date:
    return date(year, month, 1)


def _month_end(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def _add_months(day: date, months: int) -> date:
    """Shift a date by whole months, clamping the day to the target month's length."""
    index = day.year * 12 + (day.month - 1) + months
    year, month = divmod(index, 12)
    return date(year, month + 1, min(day.day, calendar.monthrange(year, month + 1)[1]))


class DateRangeResolver:
    """Resolves date specs to concrete ranges relative to a clock's today.

    Args:
        clock: The source of today's date.
        week_starts_on: First day of the week, 0 = Monday (the Hungarian and ISO
            convention), 6 = Sunday.
    """

    def __init__(self, clock: Clock, week_starts_on: int = 0) -> None:
        if not 0 <= week_starts_on <= 6:
            raise ValueError("week_starts_on must be 0 (Monday) to 6 (Sunday)")
        self._clock = clock
        self._week_starts_on = week_starts_on

    def resolve(self, spec: dict[str, Any]) -> DateRange:
        """Resolve ``spec`` to an inclusive range.

        Args:
            spec: A dict in the grammar described in the module docstring.

        Returns:
            The concrete range.

        Raises:
            DateSpecError: If the spec is malformed, names an impossible date,
                ends before it starts, or spans an implausible length.
        """
        if not isinstance(spec, dict):
            raise DateSpecError(f"a date spec must be an object, got {spec!r}")
        kind = spec.get("kind")
        handlers = {
            "absolute": self._absolute,
            "calendar": self._calendar,
            "relative": self._relative,
            "rolling": self._rolling,
            "between": self._between,
        }
        handler = handlers.get(kind)  # type: ignore[arg-type]
        if handler is None:
            raise DateSpecError(
                f"unknown date spec kind {kind!r} in {spec!r}; every spec needs a "
                f"'kind', one of {sorted(handlers)}"
            )
        try:
            result = handler(spec)
        except (ValueError, OverflowError) as exc:
            if isinstance(exc, DateSpecError):
                raise
            raise DateSpecError(f"impossible date in {spec!r}: {exc}") from exc
        return self._checked(result, spec)

    # ------------------------------------------------------------------ kinds

    def _absolute(self, spec: dict[str, Any]) -> DateRange:
        return DateRange(
            date.fromisoformat(spec["start"]), date.fromisoformat(spec["end"])
        )

    def _calendar(self, spec: dict[str, Any]) -> DateRange:
        year = self._year(spec)
        month, quarter, day = spec.get("month"), spec.get("quarter"), spec.get("day")
        if month is not None and quarter is not None:
            raise DateSpecError("give a month or a quarter, not both")
        if day is not None and month is None:
            raise DateSpecError("a day needs a month")
        if quarter is not None:
            if quarter not in (1, 2, 3, 4):
                raise DateSpecError(f"quarter must be 1-4, got {quarter!r}")
            first = 3 * (quarter - 1) + 1
            return DateRange(_month_start(year, first), _month_end(year, first + 2))
        if month is None:
            return DateRange(date(year, 1, 1), date(year, 12, 31))
        if not 1 <= month <= 12:
            raise DateSpecError(f"month must be 1-12, got {month!r}")
        if day is None:
            return DateRange(_month_start(year, month), _month_end(year, month))
        only = date(year, month, day)
        return DateRange(only, only)

    def _year(self, spec: dict[str, Any]) -> int:
        has_year, has_offset = (
            spec.get("year") is not None,
            spec.get("year_offset") is not None,
        )
        if has_year == has_offset:
            raise DateSpecError(
                "a calendar spec needs exactly one of 'year' and 'year_offset'"
            )
        return (
            spec["year"] if has_year else self._clock.today().year + spec["year_offset"]
        )

    def _relative(self, spec: dict[str, Any]) -> DateRange:
        today, offset, unit = self._clock.today(), spec.get("offset"), spec.get("unit")
        if not isinstance(offset, int):
            raise DateSpecError("a relative spec needs an integer 'offset'")
        if unit == "day":
            day = today + timedelta(days=offset)
            return DateRange(day, day)
        if unit == "week":
            back = (today.weekday() - self._week_starts_on) % 7
            start = today - timedelta(days=back) + timedelta(weeks=offset)
            return DateRange(start, start + timedelta(days=6))
        if unit == "month":
            shifted = _add_months(today, offset)
            return DateRange(
                _month_start(shifted.year, shifted.month),
                _month_end(shifted.year, shifted.month),
            )
        if unit == "quarter":
            shifted = _add_months(today, 3 * offset)
            first = 3 * ((shifted.month - 1) // 3) + 1
            return DateRange(
                _month_start(shifted.year, first), _month_end(shifted.year, first + 2)
            )
        if unit == "year":
            year = today.year + offset
            return DateRange(date(year, 1, 1), date(year, 12, 31))
        raise DateSpecError(
            f"unknown unit {unit!r}; use day, week, month, quarter or year"
        )

    def _rolling(self, spec: dict[str, Any]) -> DateRange:
        """A window of ``count`` units ending today (past, the default) or starting today (future).

        The window includes today, so "the last 30 days" is exactly 30 days.
        """
        today, count, unit = self._clock.today(), spec.get("count"), spec.get("unit")
        direction = spec.get("direction", "past")
        if not isinstance(count, int) or count < 1:
            raise DateSpecError("a rolling spec needs a positive integer 'count'")
        if direction not in ("past", "future"):
            raise DateSpecError("direction must be 'past' or 'future'")
        sign = -1 if direction == "past" else 1
        if unit == "day":
            edge = today + timedelta(days=sign * (count - 1))
        elif unit == "week":
            edge = today + timedelta(days=sign * (7 * count - 1))
        elif unit in ("month", "year"):
            months = count * (12 if unit == "year" else 1)
            edge = _add_months(today, sign * months) - timedelta(days=sign)
        else:
            raise DateSpecError(f"unknown unit {unit!r}; use day, week, month or year")
        return DateRange(min(today, edge), max(today, edge))

    def _between(self, spec: dict[str, Any]) -> DateRange:
        if "from" not in spec or "to" not in spec:
            raise DateSpecError("a between spec needs 'from' and 'to'")
        return DateRange(self.resolve(spec["from"]).start, self.resolve(spec["to"]).end)

    @staticmethod
    def _checked(result: DateRange, spec: dict[str, Any]) -> DateRange:
        if result.end < result.start:
            raise DateSpecError(f"the range ends before it starts in {spec!r}")
        if result.days > _MAX_YEARS * 366:
            raise DateSpecError(
                f"the range spans more than {_MAX_YEARS} years in {spec!r}"
            )
        return result
