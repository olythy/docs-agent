"""Shared console-output formatting helpers for scripts/'s diagnostic tools.

Purpose:
    A single place for printing conventions several scripts converge on
    independently otherwise (``evaluate_retrieval.py``, ``extract_text.py``,
    ``inspect_chunks.py``):
        - ``truncate()`` — shortening a piece of free text to a fixed
          width for a table column or preview, without cutting a word in
          half.
        - ``wrap()`` — wrapping a long explanatory paragraph across
          multiple lines instead of one unreadable, terminal-width-busting
          line — for prose meant to be read in full, where truncating
          would lose the point rather than just shorten it.

    Both scripts printed a single, several-hundred-character explanatory
    paragraph as one unwrapped line before this existed (confirmed
    empirically: 745 and 817 characters respectively) — this is the fix.
"""

import textwrap

#: Conservative, universally-safe terminal width — every helper here
#: targets staying under it, with a little margin.
DEFAULT_WIDTH = 78


def truncate(text: str, max_len: int = DEFAULT_WIDTH) -> str:
    """Shorten ``text`` to at most ``max_len`` characters, on a word boundary.

    Uses ``textwrap.shorten`` (stdlib) rather than a plain ``text[:max_len]``
    slice — that would cut a word in half whenever the limit happens to
    fall inside one, which for arbitrary free text (a question, a chunk of
    extracted text) is most of the time. Also collapses internal
    whitespace/newlines to single spaces, which ``textwrap.shorten`` does
    as part of the same pass — desirable here too, since the result is
    meant to fit on one line/row.

    Args:
        text: The text to shorten.
        max_len: Maximum length of the result, including the placeholder.

    Returns:
        ``text`` unchanged if it already fits within ``max_len``,
        otherwise shortened with a trailing ``"..."``, never mid-word.
    """
    return textwrap.shorten(text, width=max_len, placeholder="...")


def wrap(text: str, width: int = DEFAULT_WIDTH) -> str:
    """Wrap ``text`` across multiple lines of at most ``width`` characters.

    Uses ``textwrap.fill`` (stdlib) — unlike :func:`truncate`, nothing is
    cut: every word survives, just spread across lines. For long
    explanatory prose that needs to be read in full, not previewed.

    Args:
        text: The paragraph to wrap.
        width: Maximum width per line.

    Returns:
        ``text`` reflowed into a multi-line string, ready to ``print()``.
    """
    return textwrap.fill(text, width=width)
