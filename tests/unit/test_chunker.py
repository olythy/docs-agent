"""Tests for ingestion.chunker: word-based chunking with overlap."""

import warnings

import pytest

import ingestion.chunker as chunker_module
from ingestion.chunker import (
    _split_words_into_chunks,
    chunk_pages,
    validate_chunk_size_against_model,
)


def test_split_empty_words_returns_empty_list():
    assert _split_words_into_chunks([], chunk_size=5, chunk_overlap=1) == []


def test_split_no_overlap_splits_into_equal_groups():
    words = ["a", "b", "c", "d", "e", "f"]
    result = _split_words_into_chunks(words, chunk_size=2, chunk_overlap=0)
    assert result == [["a", "b"], ["c", "d"], ["e", "f"]]


def test_split_last_group_may_be_shorter():
    words = ["a", "b", "c", "d", "e"]
    result = _split_words_into_chunks(words, chunk_size=2, chunk_overlap=0)
    assert result == [["a", "b"], ["c", "d"], ["e"]]


def test_split_overlap_repeats_words_between_adjacent_groups():
    words = ["a", "b", "c", "d"]
    result = _split_words_into_chunks(words, chunk_size=3, chunk_overlap=1)
    # step = chunk_size - chunk_overlap = 2: [0:3], [2:4] — "c" appears in both.
    assert result == [["a", "b", "c"], ["c", "d"]]


@pytest.mark.parametrize("chunk_size,chunk_overlap", [(5, 5), (5, 6)])
def test_split_raises_when_overlap_not_smaller_than_size(chunk_size, chunk_overlap):
    """The exact bug from the code review: overlap >= size used to hang forever."""
    with pytest.raises(ValueError, match="chunk_overlap"):
        _split_words_into_chunks(["a"], chunk_size, chunk_overlap)


def test_split_raises_on_non_positive_chunk_size():
    with pytest.raises(ValueError, match="chunk_size"):
        _split_words_into_chunks(["a"], chunk_size=0, chunk_overlap=0)


def test_split_raises_on_negative_overlap():
    with pytest.raises(ValueError, match="chunk_overlap"):
        _split_words_into_chunks(["a"], chunk_size=5, chunk_overlap=-1)


def test_chunk_pages_builds_content_and_metadata():
    pages = [{"page_number": 1, "text": "one two three four"}]
    chunks = chunk_pages(pages, source_file="doc.pdf", chunk_size=2, chunk_overlap=0)

    assert [c["content"] for c in chunks] == ["one two", "three four"]
    assert chunks[0]["metadata"] == {
        "source_file": "doc.pdf",
        "page_number": 1,
        "chunk_index": 0,
    }
    assert chunks[1]["metadata"]["chunk_index"] == 1


def test_chunk_pages_skips_blank_pages():
    pages = [
        {"page_number": 1, "text": "   "},
        {"page_number": 2, "text": "hello world"},
    ]
    chunks = chunk_pages(pages, source_file="doc.pdf", chunk_size=5, chunk_overlap=0)
    assert len(chunks) == 1
    assert chunks[0]["metadata"]["page_number"] == 2


def test_chunk_pages_chunk_index_is_global_across_pages():
    pages = [
        {"page_number": 1, "text": "a b"},
        {"page_number": 2, "text": "c d"},
    ]
    chunks = chunk_pages(pages, source_file="doc.pdf", chunk_size=2, chunk_overlap=0)
    assert [c["metadata"]["chunk_index"] for c in chunks] == [0, 1]


def test_chunk_pages_uses_settings_defaults_when_not_overridden(
    monkeypatch, settings_override
):
    monkeypatch.setattr(
        chunker_module,
        "settings",
        settings_override(CHUNK_SIZE=3, CHUNK_OVERLAP=0),
    )
    pages = [{"page_number": 1, "text": "one two three four five six"}]

    chunks = chunk_pages(pages, source_file="doc.pdf")

    assert [c["content"] for c in chunks] == ["one two three", "four five six"]


def test_chunk_pages_propagates_invalid_overlap_error():
    pages = [{"page_number": 1, "text": "one two three"}]
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_pages(pages, source_file="doc.pdf", chunk_size=3, chunk_overlap=3)


def test_validate_chunk_size_no_warning_when_max_seq_length_unknown():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_chunk_size_against_model(chunk_size=99999, max_seq_length=None)


def test_validate_chunk_size_no_warning_when_within_limit():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_chunk_size_against_model(chunk_size=50, max_seq_length=128)


def test_validate_chunk_size_warns_when_exceeding_limit():
    with pytest.warns(UserWarning, match="CHUNK_SIZE"):
        validate_chunk_size_against_model(chunk_size=100, max_seq_length=128)


def test_validate_chunk_size_warns_for_the_actual_default_config():
    """Regression test for the real bug this guard exists for.

    The default CHUNK_SIZE=500 vs. the default local model's real
    max_seq_length=128 — confirmed empirically:
    SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2').max_seq_length
    is 128, not the 512 the old chunker.py docstring assumed.
    """
    with pytest.warns(UserWarning, match="CHUNK_SIZE=500"):
        validate_chunk_size_against_model(chunk_size=500, max_seq_length=128)
