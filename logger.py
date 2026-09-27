"""Structured JSONL telemetry and event logging module.

Provides structured audit logging for pipeline operations (such as keyword filtering,
chunking, reranking, and query execution) into a line-delimited JSON (.jsonl) file.
This allows inspecting why specific keywords were dropped or how relevance scores were
assigned without cluttering standard application output.

Key exports:
    LogAction   -- StrEnum of recognizable pipeline actions/events.
    EventLogger -- Core class responsible for writing structured event records.
    get_logger  -- Factory returning the singleton EventLogger instance.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

from config import settings


class LogAction(StrEnum):
    """Recognizable action types recorded in the structured event log."""

    FTS_QUERY_FILTERED = "fts_query_filtered"
    DOCUMENT_INGESTED = "document_ingested"
    RELEVANCE_GATE_CHECKED = "relevance_gate_checked"
    RERANK_APPLIED = "rerank_applied"
    ANSWER_GENERATED = "answer_generated"


class EventLogger:
    """Writes structured JSON lines to a designated event log file.

    Attributes:
        log_path: Path to the target JSONL file. Defaults to ``settings.LOG_FILE_PATH``.
    """

    def __init__(self, log_path: str | Path | None = None) -> None:
        """Initialise the logger with a designated destination path.

        Args:
            log_path: Path to the log file, or ``None`` to use ``settings.LOG_FILE_PATH``.
        """
        self._log_path = Path(log_path or settings.LOG_FILE_PATH)

    @property
    def log_path(self) -> Path:
        """Return the active log file path."""
        return self._log_path

    def log(
        self, action: LogAction | str, data: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Append a single structured JSON line to the log file.

        Args:
            action: The action identifier (from :class:`LogAction` or string).
            data: Arbitrary metadata dictionary associated with the action.

        Returns:
            The complete entry dict that was written.
        """
        entry: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "action": str(action),
            "data": data or {},
        }

        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            # Observability logging should never crash the main application pipeline
            pass

        return entry

    def log_fts_query_filtered(
        self,
        original_query: str,
        kept_terms: list[str],
        dropped_terms: list[str],
    ) -> dict[str, Any]:
        """Record keyword filtering decisions made during full-text search preparation.

        Args:
            original_query: The raw input query string.
            kept_terms: Tokens retained for the full-text search query.
            dropped_terms: Tokens filtered out (stop words or short words).

        Returns:
            The logged entry dict.
        """
        return self.log(
            action=LogAction.FTS_QUERY_FILTERED,
            data={
                "original_query": original_query,
                "kept_terms": kept_terms,
                "dropped_terms": dropped_terms,
            },
        )


@lru_cache(maxsize=1)
def get_logger() -> EventLogger:
    """Return the active singleton EventLogger instance."""
    return EventLogger()
