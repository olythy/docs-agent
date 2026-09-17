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
