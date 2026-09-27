"""Tests for logger: EventLogger and LogAction."""

import json
from pathlib import Path

from logger import EventLogger, LogAction, get_logger


def test_event_logger_writes_valid_jsonl(tmp_path: Path):
    log_file = tmp_path / "test_log.jsonl"
    logger = EventLogger(log_path=log_file)

    entry = logger.log(LogAction.FTS_QUERY_FILTERED, {"test": "data"})
    assert entry["action"] == "fts_query_filtered"
    assert "timestamp" in entry
    assert entry["data"] == {"test": "data"}

    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    parsed = json.loads(lines[0])
    assert parsed["action"] == "fts_query_filtered"
    assert parsed["data"] == {"test": "data"}


def test_event_logger_log_fts_query_filtered_helper(tmp_path: Path):
    log_file = tmp_path / "test_fts.jsonl"
    logger = EventLogger(log_path=log_file)

    entry = logger.log_fts_query_filtered(
        original_query="What is AI?",
        kept_terms=["AI"],
        dropped_terms=["What", "is"],
    )

    assert entry["action"] == LogAction.FTS_QUERY_FILTERED
    assert entry["data"]["original_query"] == "What is AI?"
    assert entry["data"]["kept_terms"] == ["AI"]
    assert entry["data"]["dropped_terms"] == ["What", "is"]


def test_get_logger_singleton():
    logger1 = get_logger()
    logger2 = get_logger()
    assert logger1 is logger2


def test_event_logger_default_path():
    from config import settings

    logger = EventLogger()
    assert logger.log_path == Path(settings.LOG_FILE)
