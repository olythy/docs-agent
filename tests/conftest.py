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
    driver instance and don't leak state across test boundaries. Also resets
    drivers.gcloud_auth's module-level token cache -- shared across every
    Vertex AI driver instance (not per-instance), so without this reset, a
    test that warms it would make a later test's "subprocess.run" mock never
    actually get called, silently breaking that test's own assertions.
    """
    from drivers import gcloud_auth
    from drivers.embedding import get_embedding_driver
    from drivers.llm import get_answer_driver
    from drivers.reranker import get_reranker_driver

    def _reset_gcloud_auth_cache() -> None:
        gcloud_auth._cached_token = None
        gcloud_auth._token_fetched_at = 0.0

    get_embedding_driver.cache_clear()
    get_reranker_driver.cache_clear()
    get_answer_driver.cache_clear()
    _reset_gcloud_auth_cache()
    try:
        yield
    finally:
        get_embedding_driver.cache_clear()
        get_reranker_driver.cache_clear()
        get_answer_driver.cache_clear()
        _reset_gcloud_auth_cache()
