"""Tests for scripts.format_utils.truncate and .wrap."""

from scripts.format_utils import truncate, wrap


def test_truncate_returns_text_unchanged_when_it_already_fits():
    assert truncate("short text", max_len=50) == "short text"


def test_truncate_shortens_on_a_word_boundary():
    text = "one two three four five six seven eight nine ten"
    result = truncate(text, max_len=20)

    assert len(result) <= 20
    assert result.endswith("...")
    # Every word in the result must be a real word from the original —
    # never a fragment produced by cutting mid-word.
    words = result[: -len("...")].split()
    assert all(w in text.split() for w in words)


def test_truncate_collapses_internal_whitespace():
    assert truncate("a   b\nc", max_len=50) == "a b c"


def test_wrap_returns_single_line_unchanged_when_it_already_fits():
    assert wrap("short text", width=50) == "short text"


def test_wrap_splits_a_long_paragraph_across_multiple_lines_at_word_boundaries():
    text = "one two three four five six seven eight nine ten"
    result = wrap(text, width=20)

    lines = result.split("\n")
    assert len(lines) > 1
    assert all(len(line) <= 20 for line in lines)
    # No word may be cut in half — every line's words are real words.
    assert all(w in text.split() for line in lines for w in line.split())


def test_wrap_loses_no_words():
    text = "one two three four five six seven eight nine ten"
    result = wrap(text, width=20)
    assert result.replace("\n", " ").split() == text.split()
