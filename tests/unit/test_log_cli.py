"""Unit tests for scripts.log_cli (pure logic, event formatting, stats, clear)."""

import json
from pathlib import Path

from scripts.log_cli import (
    cmd_clear,
    cmd_stats,
    cmd_tail,
    format_event,
    format_timestamp,
    read_recent_lines,
    resolve_log_path,
)


def test_format_timestamp():
    assert format_timestamp("2026-09-28T09:24:04.426134+00:00") == "09:24:04"
    assert format_timestamp("invalid") == "invalid"
    assert format_timestamp("") == "??:??:??"


def test_format_event_fts_query_filtered():
    entry = {
        "timestamp": "2026-09-28T10:00:00+00:00",
        "action": "fts_query_filtered",
        "data": {
            "original_query": "What is Python?",
            "kept_terms": ["Python"],
            "dropped_terms": ["What", "is"],
        },
    }
    formatted = format_event(entry, color=False)
    assert "[FTS FILTER]" in formatted
    assert "'What is Python?'" in formatted
    assert "['Python']" in formatted
    assert "['What', 'is']" in formatted


def test_format_event_rerank_applied():
    entry = {
        "timestamp": "2026-09-28T10:00:00+00:00",
        "action": "rerank_applied",
        "data": {
            "question": "Sample question?",
            "reranker_model": "cross-encoder/model",
            "threshold": 0.5,
            "candidates_count": 4,
            "accepted_count": 2,
            "top_score": 1.25,
        },
    }
    formatted = format_event(entry, color=False)
    assert "[RERANK]" in formatted
    assert "'Sample question?'" in formatted
    assert "2/4 (50%)" in formatted
    assert "Top Score: 1.25" in formatted
    assert "cross-encoder/model" in formatted


def test_format_event_unknown_action():
    entry = {
        "timestamp": "2026-09-28T10:00:00+00:00",
        "action": "custom_action",
        "data": {"key": "val"},
    }
    formatted = format_event(entry, color=False)
    assert "[CUSTOM_ACTION]" in formatted
    assert '"key": "val"' in formatted


def test_read_recent_lines(tmp_path: Path):
    sample_file = tmp_path / "test.txt"
    sample_file.write_text("line1\nline2\nline3\nline4\nline5\n")

    assert read_recent_lines(sample_file, 2) == ["line4\n", "line5\n"]
    assert read_recent_lines(sample_file, 0) == []
    assert read_recent_lines(tmp_path / "nonexistent.txt", 5) == []


def test_resolve_log_path():
    p_dev = resolve_log_path(is_test=False)
    assert p_dev.name == "log.jsonl"

    p_test = resolve_log_path(is_test=True)
    assert p_test.name == "log-test.jsonl"

    p_custom = resolve_log_path(explicit_path="/tmp/custom.jsonl")
    assert p_custom == Path("/tmp/custom.jsonl")


def test_cmd_stats(tmp_path: Path, capsys):
    log_file = tmp_path / "test.jsonl"
    entries = [
        {
            "action": "fts_query_filtered",
            "data": {
                "original_query": "q1",
                "kept_terms": ["a"],
                "dropped_terms": ["the", "is"],
            },
        },
        {
            "action": "fts_query_filtered",
            "data": {
                "original_query": "q2",
                "kept_terms": ["b"],
                "dropped_terms": ["the"],
            },
        },
        {
            "action": "rerank_applied",
            "data": {
                "candidates_count": 2,
                "accepted_count": 1,
            },
        },
    ]
    with log_file.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    code = cmd_stats(["--path", str(log_file)])
    assert code == 0

    captured = capsys.readouterr().out
    assert "Total structured entries: 3" in captured
    assert "fts_query_filtered" in captured
    assert "rerank_applied" in captured
    assert "'the'" in captured


def test_cmd_clear(tmp_path: Path):
    log_file = tmp_path / "test.jsonl"
    log_file.write_text("some content\n")

    code = cmd_clear(["--path", str(log_file)])
    assert code == 0
    assert log_file.stat().st_size == 0


def test_cmd_tail(tmp_path: Path, capsys):
    log_file = tmp_path / "test.jsonl"
    log_file.write_text(
        json.dumps(
            {
                "timestamp": "2026-09-28T10:00:00+00:00",
                "action": "fts_query_filtered",
                "data": {"original_query": "test query"},
            }
        )
        + "\n"
    )

    code = cmd_tail(["--path", str(log_file), "--no-color"])
    assert code == 0
    assert "Query: 'test query'" in capsys.readouterr().out
