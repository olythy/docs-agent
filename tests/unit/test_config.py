"""Tests for config.Settings defaults and immutability."""

import dataclasses

import pytest

from config import settings


def test_settings_has_sensible_defaults():
    assert settings.EMBEDDING_DRIVER in {"local", "openai"}
    assert settings.LLM_DRIVER in {"openrouter", "openai"}
    assert settings.CHUNK_SIZE > settings.CHUNK_OVERLAP >= 0
    assert 0 <= settings.RETRIEVAL_MIN_SCORE <= 1


def test_settings_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.CHUNK_SIZE = 999
