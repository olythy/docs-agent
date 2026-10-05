"""Checks that an extracted value really exists in the document it came from.

An LLM extractor can invent a value. Every extracted row therefore carries a
verbatim quote, and this module verifies two things deterministically: the
quote is present in the text the extractor was given, and the value is derivable
from the quote (a date the quote writes, a number the quote contains, a token the
catalog allows). That guarantees the value *exists* in the document. It does not
prove the extractor picked the *right* one when a document states several, which
is why ambiguous keys store every candidate with a role instead.

Key exports:
    EvidenceVerifier   -- Runs the checks.
    VerificationResult -- What was and was not confirmed.
"""

import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation

from metadata.date_parsers import DateParser, HungarianDateParser
from models import MetaKey, ValueType

# Digits grouped by '.', space or NBSP ("57.709.272"), optional decimal comma.
_NUMBER = re.compile(r"\d{1,3}(?:[.  ]\d{3})+(?:,\d+)?|\d+(?:,\d+)?")


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _alnum(text: str) -> str:
    """Lower-case letters and digits only, accents stripped: a deliberately loose key."""
    folded = unicodedata.normalize("NFKD", text.lower())
    return re.sub(
        r"[^a-z0-9]", "", "".join(c for c in folded if not unicodedata.combining(c))
    )


def numbers_in(text: str) -> set[Decimal]:
    """Return every number written in ``text``, reading '.'/space as thousands and ',' as decimal."""
    found: set[Decimal] = set()
    for match in _NUMBER.findall(text):
        cleaned = re.sub(r"[.  ]", "", match).replace(",", ".")
        try:
            found.add(Decimal(cleaned))
        except InvalidOperation:
            continue
    return found


@dataclass(frozen=True)
class VerificationResult:
    """The outcome of verifying one extracted value.

    Attributes:
        quote_exact: The quote is in the text, ignoring case and whitespace.
        quote_loose: The quote is in the text ignoring everything but letters
            and digits (and accents); true whenever ``quote_exact`` is.
        value_derivable: The value follows from the quote (see the module doc).
    """

    quote_exact: bool
    quote_loose: bool
    value_derivable: bool

    @property
    def verified(self) -> bool:
        """Whether the value may be used: the quote exists and the value follows from it."""
        return self.quote_loose and self.value_derivable


class EvidenceVerifier:
    """Verifies extracted values against the text the extractor was shown.

    Args:
        text: The text the extractor saw (the selected chunks' bodies).
        date_parser: How dates are written in this corpus's language.
    """

    def __init__(self, text: str, date_parser: DateParser | None = None) -> None:
        self._squashed = _squash(text)
        self._alnum = _alnum(text)
        self._dates = date_parser or HungarianDateParser()

    def contains(self, evidence: str | None) -> bool:
        """Whether a quote appears in the text (whitespace/case-insensitive, then loose).

        For a decision that has no typed value to derive (such as a document's
        type): the quote is the only evidence.
        """
        if not evidence or not evidence.strip():
            return False
        return _squash(evidence) in self._squashed or _alnum(evidence) in self._alnum

    def verify(
        self, key: MetaKey, value: str | None, evidence: str | None
    ) -> VerificationResult:
        """Verify one value of ``key`` against its quote.

        Args:
            key: The catalog key (its type and allowed values drive the check).
            value: The extracted value as text; a date as ISO ``YYYY-MM-DD``.
            evidence: The verbatim quote the extractor gave.

        Returns:
            What was confirmed. A missing value or quote confirms nothing.
        """
        if not value or not evidence:
            return VerificationResult(False, False, False)
        exact = _squash(evidence) in self._squashed
        loose = exact or _alnum(evidence) in self._alnum
        return VerificationResult(exact, loose, self._derivable(key, value, evidence))

    def _derivable(self, key: MetaKey, value: str, evidence: str) -> bool:
        if key.allowed_values is not None and value not in key.allowed_values:
            return False
        try:
            if key.value_type is ValueType.DATE:
                return date.fromisoformat(value) in self._dates.dates_in(evidence)
            if key.value_type is ValueType.NUMBER:
                return Decimal(value) in numbers_in(evidence)
        except (ValueError, InvalidOperation):
            return False
        return True
