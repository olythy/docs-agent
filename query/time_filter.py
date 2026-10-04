"""Extract the years a question is about, so retrieval can look at them explicitly.

Embeddings capture topical similarity but are weak at telling years apart,
so a question like "... 2020 és 2022 között" can come back dominated by
documents from the wrong years (confirmed live, see docs/decisions.md).
Documents already carry their own ``document_date`` (see
``models.ChunkMetadata``); this module only reads the *question's* side.

Key exports:
    extract_years -- the distinct years a question refers to, as a sorted list.

Deliberately a pure, deterministic function (no LLM call) so it adds no
latency or cost, and returns ``[]`` whenever it is unsure -- the caller then
behaves exactly as it did before.
"""

import re

from store import extract_identifier_tokens

_YEAR = r"((?:19[89]\d|20[0-3]\d))"
# "2020-2022", "2020–2022"
_DASH_RANGE = re.compile(_YEAR + r"\s*[-–—]\s*" + _YEAR)
# Hungarian "2020 és 2022 között/közötti", English "between 2020 and 2022"
_BETWEEN_RANGE = re.compile(
    rf"(?:{_YEAR}\s+(?:és|és a)\s+{_YEAR}\s+közöt)|(?:between\s+{_YEAR}\s+and\s+{_YEAR})",
    re.IGNORECASE,
)
_SINGLE_YEAR = re.compile(rf"(?<![\d/.]){_YEAR}(?![\d/])")
#: Guards against a typo or a stray number expanding into a huge range.
_MAX_RANGE_YEARS = 30


def extract_years(question: str) -> list[int]:
    """Return the distinct years ``question`` refers to, ascending.

    A range ("2020 és 2022 között", "2020-2022", "between 2020 and 2022")
    expands to every year in it; separate mentions ("a 2021-es és a
    2023-as években") stay separate -- 2022 is *not* implied. Years that are
    part of an identifier (a case number like ``P.20.457/2018/11``) are
    ignored: they are not a time period the question asks about.

    Args:
        question: The user's natural-language question.

    Returns:
        Sorted distinct years, or ``[]`` if none were found.
    """
    text = question
    for token in extract_identifier_tokens(question):
        # A bare year range ("2020-2022") also looks identifier-like, but it
        # is exactly what we want to read here -- only strip real identifiers.
        if not _DASH_RANGE.fullmatch(token):
            text = text.replace(token, " ")

    years: set[int] = set()
    for pattern in (_DASH_RANGE, _BETWEEN_RANGE):
        for match in pattern.finditer(text):
            bounds = [int(g) for g in match.groups() if g]
            low, high = min(bounds), max(bounds)
            if high - low < _MAX_RANGE_YEARS:
                years.update(range(low, high + 1))
        text = pattern.sub(" ", text)

    years.update(int(m.group(1)) for m in _SINGLE_YEAR.finditer(text))
    return sorted(years)
