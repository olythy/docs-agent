"""The source of "today", injected so relative dates are testable.

A question like "last October" or "next week" means something different every
day, so whatever resolves it must not call ``date.today()`` itself. It takes a
:class:`Clock`; production uses :class:`SystemClock`, tests use :class:`FixedClock`
and so get the same answer on every day they run.

Key exports:
    Clock       -- The contract: what day is it?
    SystemClock -- The real calendar.
    FixedClock  -- A frozen day, for tests and reproducible evals.
"""

from abc import ABC, abstractmethod
from datetime import date, datetime
from zoneinfo import ZoneInfo


class Clock(ABC):
    """Tells what today's date is."""

    @abstractmethod
    def today(self) -> date:
        """Return today's date."""


class SystemClock(Clock):
    """The real calendar day, in a stated time zone.

    "Today" depends on where the user is: just after midnight in Budapest it is
    still yesterday in UTC, and "next week" would be a day off. So the zone is
    explicit rather than whatever the machine happens to be set to.

    Args:
        timezone: IANA zone name; the default suits the Hungarian corpus.
    """

    def __init__(self, timezone: str = "Europe/Budapest") -> None:
        self._zone = ZoneInfo(timezone)

    def today(self) -> date:
        return datetime.now(self._zone).date()


class FixedClock(Clock):
    """A frozen day.

    Args:
        day: The date every call returns.
    """

    def __init__(self, day: date) -> None:
        self._day = day

    def today(self) -> date:
        return self._day
