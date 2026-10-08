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
``...2023/40``. A *short* wanted identifier (``4.P``) is only matched when equal: as a
prefix it would match every number that starts that way.

Key exports:
    normalize_identifier -- The normalised form.
    identifier_matches   -- The matching rule (equal, or continued by a non-digit).
    identifier_contains  -- The partial rule (the wanted one is a part of the stored one).
    is_specific_enough   -- Whether an identifier may be looked for as a part.
    partial_pattern      -- The partial rule as a SQL regex.
    compact_identifier, identifier_compact_contains, compact_pattern, compact_sql
                         -- The separator-insensitive rule (letters and digits only).
    normalized_sql       -- The same normalisation as a SQL expression.
    IDENTIFIER_TRIM      -- The punctuation trimmed from both ends.
    MIN_CONTINUATION_LENGTH -- How long a wanted identifier must be to match a continuation.
"""

import re
import unicodedata

#: Characters trimmed from both ends of a normalised identifier. The SQL uses the same set.
IDENTIFIER_TRIM = "./-,;:"

#: How long (normalised) a wanted identifier must be to match a stored one that merely
#: *continues* it. Shorter ones ("4.P", "HU001") match only when equal.
MIN_CONTINUATION_LENGTH = 8

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
        ``True`` when, once normalised, the two are equal, or ``stored`` continues
        ``wanted`` with a non-digit and ``wanted`` is at least
        :data:`MIN_CONTINUATION_LENGTH` long. An empty ``wanted`` matches nothing.
    """
    s, w = normalize_identifier(stored), normalize_identifier(wanted)
    if not w:
        return False
    return s == w or (
        len(w) >= MIN_CONTINUATION_LENGTH
        and s.startswith(w)
        and not s[len(w)].isdigit()
    )


def normalized_sql(column: str) -> str:
    """The SQL expression that normalises ``column`` the way :func:`normalize_identifier` does.

    The compiled filters and the resolver's lookup both compare identifiers in SQL, and
    both use this one expression, so there is a single SQL spelling of the rule (a
    database test holds it equal to the Python one).

    Args:
        column: A text column or expression, e.g. ``m.value_text``.
    """
    return (
        f"btrim(regexp_replace(lower(normalize({column}, NFKC)), '\\s+', '', 'g'), "
        f"'{IDENTIFIER_TRIM}')"
    )


# --------------------------------------------------------------- partial identifiers

#: How much an identifier must say before it may match a *part* of a stored one. A short or
#: bare token ("4.P", "2023-01", "123456") would match far too many documents.
MIN_PARTIAL_ALPHANUMERICS = 6
_SEPARATORS = "./-"

#: A date written with digits and separators, which is not an identifier part.
_DATE_LIKE = re.compile(
    r"\d{4}[-./]\d{1,2}([-./]\d{1,2})?|\d{1,2}[-./]\d{1,2}[-./]\d{4}"
)


def is_specific_enough(wanted: str) -> bool:
    """Whether an identifier says enough to be looked for as a *part* of another.

    At least :data:`MIN_PARTIAL_ALPHANUMERICS` letters and digits, at least one digit, not a
    date, and some structure: a letter, at least two separators, or eight or more letters
    and digits. So ``20.277/2019/77`` and ``2024/00123`` qualify; ``4.P`` (too short),
    ``2023-01`` and ``2023-01-15`` (dates) and ``123456`` (a short bare number) do not.

    Args:
        wanted: An identifier as written.
    """
    w = normalize_identifier(wanted)
    alphanumerics = sum(c.isalnum() for c in w)
    letters = sum(c.isalpha() for c in w)
    separators = sum(c in _SEPARATORS for c in w)
    return (
        alphanumerics >= MIN_PARTIAL_ALPHANUMERICS
        and any(c in "0123456789" for c in w)
        and not _DATE_LIKE.fullmatch(w)
        and (letters >= 1 or separators >= 2 or alphanumerics >= 8)
    )


def identifier_contains(stored: str, wanted: str) -> bool:
    """Whether the wanted identifier is a *part* of the stored one, set off by separators.

    People often give only the end of a composite identifier ("P.20.277/2019/77" or
    "20.277/2019/77" for a stored "10.P.20.277/2019/77"): the leading part (a court, an
    office, a series) is left out. The wanted identifier then stands inside the stored one,
    with nothing alphanumeric directly before it and no digit directly after it (so
    ``.../4`` is not found in ``.../40``, nor ``0.277/2019/77`` in ``10.P.20.277/2019/77``).
    A wanted identifier that is not :func:`is_specific_enough` matches nothing.

    Args:
        stored: The value in the metadata, as written.
        wanted: The identifier looked for, as written.
    """
    w = normalize_identifier(wanted)
    if not is_specific_enough(w):
        return False
    s = normalize_identifier(stored)
    start = s.find(w)
    while start != -1:
        end = start + len(w)
        before_ok = start == 0 or not s[start - 1].isalnum()
        after_ok = end == len(s) or s[end] not in "0123456789"
        if before_ok and after_ok:
            return True
        start = s.find(w, start + 1)
    return False


def partial_pattern(wanted: str) -> str:
    """The regex (PostgreSQL) for :func:`identifier_contains`, on a normalised stored value.

    The compiled lookup runs this against the normalised stored value; a database test holds
    it equal to the Python rule.

    Args:
        wanted: An identifier as written.

    Raises:
        ValueError: If ``wanted`` is not :func:`is_specific_enough`: a pattern for such a
            wish would match far too much, so none is made.
    """
    if not is_specific_enough(wanted):
        raise ValueError(f"{wanted!r} is too weak to be looked for as a part")
    return f"(^|[^[:alnum:]]){re.escape(normalize_identifier(wanted))}([^0-9]|$)"


# ------------------------------------------------------- separator-insensitive identifiers


def compact_identifier(text: str) -> str:
    """The identifier reduced to lower-case ASCII letters and digits.

    Everything else (separators, spaces, accented letters, a trailing word) is dropped, so
    "P.20103.2022.19" and the stored "4.P.20.103/2022/19-ítélet" compare on
    ``p20103202219`` and ``4p20103202219tlet``.

    Args:
        text: An identifier as written.
    """
    return "".join(c for c in normalize_identifier(text) if c in _COMPACT_CHARACTERS)


_COMPACT_CHARACTERS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789")


def identifier_compact_contains(stored: str, wanted: str) -> bool:
    """Whether the wanted identifier is in the stored one once separators are ignored.

    The last resort for an identifier written with other separators than the stored one
    ("P.20103.2022.19" for "4.P.20.103/2022/19"). Both are reduced to letters and digits
    (:func:`compact_identifier`); the wanted one must stand inside the stored one with no
    digit directly after it, and, when it starts with a digit, none directly before it (a
    letter-initial wanted one may follow the digits of a leading series). A wanted identifier
    that is not :func:`is_specific_enough` matches nothing.

    Args:
        stored: The value in the metadata, as written.
        wanted: The identifier looked for, as written.
    """
    if not is_specific_enough(wanted):
        return False
    w, s = compact_identifier(wanted), compact_identifier(stored)
    if not w:
        return False
    start = s.find(w)
    while start != -1:
        end = start + len(w)
        before_ok = not w[0].isdigit() or start == 0 or not s[start - 1].isdigit()
        after_ok = end == len(s) or not s[end].isdigit()
        if before_ok and after_ok:
            return True
        start = s.find(w, start + 1)
    return False


def compact_pattern(wanted: str) -> str:
    """The regex (PostgreSQL) for :func:`identifier_compact_contains`, on a compact stored value.

    Raises:
        ValueError: If ``wanted`` is not :func:`is_specific_enough`.
    """
    if not is_specific_enough(wanted):
        raise ValueError(f"{wanted!r} is too weak to be looked for without separators")
    w = compact_identifier(wanted)
    lead = "(^|[^0-9])" if w[0].isdigit() else ""
    return f"{lead}{w}([^0-9]|$)"


def compact_sql(normalized_column: str) -> str:
    """The SQL expression that makes a *normalised* column compact (letters and digits only).

    Args:
        normalized_column: An expression that is already normalised (:func:`normalized_sql`).
    """
    return f"regexp_replace({normalized_column}, '[^a-z0-9]', '', 'g')"
