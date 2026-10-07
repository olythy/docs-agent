"""Comparing identifiers (case numbers, invoice numbers, ...) the way they are written.

An identifier is written in several forms: with spaces ("10. P. 20.277/2019/77."), with
a trailing full stop, in capitals or not, and often with a suffix that names the copy
or the instance ("...-ítélet", "/II", "-III"). Comparing them as plain text misses
all of these, so a key of type ``identifier`` is compared in a **normalised** form, by
one rule that this module defines and that the compiled SQL repeats (a database test
holds the two equal).

The rule: normalise both sides; a stored identifier matches a wanted one when they are
equal, or when the stored one *continues* the wanted one with something that is not a
digit (a suffix). It must not be a digit, or ``...2023/4`` would also match
``...2023/40``.

Key exports:
    normalize_identifier -- The normalised form.
    identifier_matches   -- The matching rule.
    IDENTIFIER_TRIM      -- The punctuation trimmed from both ends.
"""

import re
import unicodedata

#: Characters trimmed from both ends of a normalised identifier. The SQL uses the same set.
IDENTIFIER_TRIM = "./-,;:"

_WHITESPACE = re.compile(r"\s+")


def normalize_identifier(text: str) -> str:
    """The form in which identifiers are compared.

    Compatibility-normalised (so a no-break space or a full-width digit is the plain
    one), lower-cased, without any whitespace, and without punctuation at either end.

    Args:
        text: An identifier as written.
    """
    folded = unicodedata.normalize("NFKC", text).lower()
    return _WHITESPACE.sub("", folded).strip(IDENTIFIER_TRIM)


def identifier_matches(stored: str, wanted: str) -> bool:
    """Whether a stored identifier is the wanted one, or a written variant of it.

    Args:
        stored: The value in the metadata, as written (not yet normalised).
        wanted: The identifier looked for, as written.

    Returns:
        ``True`` when, once normalised, the two are equal or ``stored`` continues
        ``wanted`` with a non-digit. An empty ``wanted`` matches nothing.
    """
    s, w = normalize_identifier(stored), normalize_identifier(wanted)
    if not w:
        return False
    return s == w or (s.startswith(w) and not s[len(w)].isdigit())
