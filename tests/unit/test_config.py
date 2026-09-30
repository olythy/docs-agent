"""Tests for config.Settings defaults and immutability."""

import dataclasses

import pytest

from config import settings


def test_settings_has_sensible_defaults():
    assert settings.EMBEDDING_DRIVER in {"local", "openai", "openrouter", "gemini"}
    assert settings.LLM_DRIVER in {"openrouter", "openai", "gemini"}
    assert settings.CHUNK_SIZE > settings.CHUNK_OVERLAP >= 0
    assert 0 <= settings.RETRIEVAL_MIN_SCORE <= 1
    assert settings.WORDS_PER_TOKEN > 0


def test_settings_is_frozen():
    # setattr(), not `settings.CHUNK_SIZE = 999` directly: the direct form is
    # a static type error pyright correctly flags (Settings is frozen) — but
    # that's exactly what this test verifies at runtime, not a bug to fix.
    # setattr() is a dynamic call pyright doesn't type-check, so the same
    # runtime assertion holds with nothing to suppress there.
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(settings, "CHUNK_SIZE", 999)  # noqa: B010 -- see comment above


def test_settings_parsed_ingest_extensions():
    assert {".pdf", ".md", ".markdown"}.issubset(settings.parsed_ingest_extensions)

    custom = dataclasses.replace(settings, INGEST_EXTENSIONS="md, .PDF,  TXT  ")
    assert custom.parsed_ingest_extensions == frozenset({".md", ".pdf", ".txt"})


def test_settings_log_file_defaults():
    assert settings.LOG_FILE.endswith(".jsonl")
