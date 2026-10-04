"""Language-specific readers that find dates written in a text.

The verifier needs to know whether an extracted date is *derivable* from the
quote that supposedly shows it, and how dates are written depends on the
language. So the reader is a small Strategy: the verifier takes any
:class:`DateParser`, and a corpus in another language supplies its own.

Key exports:
    DateParser          -- The contract.
    HungarianDateParser -- ISO, "2024. 12. 16." and "2024. december 16." forms,
                           including the older "2024. év december hó 16. napján".
"""

import re
from abc import ABC, abstractmethod
from datetime import date

_HUNGARIAN_MONTHS = {
    name: number
    for number, name in enumerate(
        [
            "január",
            "február",
            "március",
            "április",
            "május",
            "június",
            "július",
            "augusztus",
            "szeptember",
            "október",
            "november",
            "december",
        ],
        start=1,
    )
}


def _valid(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


class DateParser(ABC):
    """Finds the calendar dates a piece of text states."""

    @abstractmethod
    def dates_in(self, text: str) -> set[date]:
        """Return every valid date written in ``text``."""


class HungarianDateParser(DateParser):
    """Reads ISO dates and the Hungarian written forms of a date.

    Confirmed on this corpus's decisions: the forms in the signature line are
    "2024. december 16.", "2024. 12. 16." and, in older documents,
    "2024. év december hó 16. napján".
    """

    _ISO = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
    _NUMERIC = re.compile(r"(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.?")
    _NAMED = re.compile(
        r"(\d{4})\.\s*(?:év\s+)?([a-záéíóöőúüű]+)\s+(?:hó\s+)?(\d{1,2})\b",
        re.IGNORECASE,
    )

    def dates_in(self, text: str) -> set[date]:
        candidates: list[date | None] = []
        for y, m, d in self._ISO.findall(text):
            candidates.append(_valid(int(y), int(m), int(d)))
        for y, m, d in self._NUMERIC.findall(text):
            candidates.append(_valid(int(y), int(m), int(d)))
        for y, name, d in self._NAMED.findall(text):
            month = _HUNGARIAN_MONTHS.get(name.lower())
            if month:
                candidates.append(_valid(int(y), month, int(d)))
        return {c for c in candidates if c is not None}
