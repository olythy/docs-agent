"""Shared fixtures for the docs-agent test suite."""

import dataclasses

import pytest

from config import Settings
from config import settings as real_settings


@pytest.fixture
def settings_override():
    """Factory fixture: build a modified copy of the real ``Settings`` singleton.

    ``Settings`` is a frozen dataclass singleton, and every consuming module
    binds its own ``from config import settings`` reference, so neither
    ``setattr``-ing the singleton nor patching ``config.settings`` affects
    modules that already imported their own name. Build a replacement with
    this fixture, then patch the specific module under test:

        modified = settings_override(CHUNK_SIZE=100, CHUNK_OVERLAP=100)
        monkeypatch.setattr(chunker, "settings", modified)
    """

    def _make(**overrides) -> Settings:
        return dataclasses.replace(real_settings, **overrides)

    return _make


@pytest.fixture(autouse=True)
def clear_driver_caches():
    """Clear lru_cache on driver factories before and after each test.

    Ensures tests that override driver settings via monkeypatch receive a fresh
    driver instance and don't leak state across test boundaries.
    """
    from drivers.embedding import get_embedding_driver
    from drivers.llm import get_answer_driver
    from drivers.reranker import get_reranker_driver

    get_embedding_driver.cache_clear()
    get_reranker_driver.cache_clear()
    get_answer_driver.cache_clear()
    try:
        yield
    finally:
        get_embedding_driver.cache_clear()
        get_reranker_driver.cache_clear()
        get_answer_driver.cache_clear()
