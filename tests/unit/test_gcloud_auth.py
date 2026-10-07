"""Tests for drivers.gcloud_auth: the shared Vertex AI access-token cache."""

import subprocess
from unittest.mock import MagicMock

import pytest

from drivers import gcloud_auth
from drivers.gcloud_auth import get_access_token
from retry_policy import TransientAPIError


def _fake_gcloud_token(token="fake-access-token"):
    result = MagicMock()
    result.returncode = 0
    result.stdout = f"{token}\n"
    return result


def test_get_access_token_returns_stripped_stdout(monkeypatch):
    monkeypatch.setattr("subprocess.run", MagicMock(return_value=_fake_gcloud_token()))
    assert get_access_token() == "fake-access-token"


def test_get_access_token_caches_across_calls(monkeypatch):
    fake_run = MagicMock(return_value=_fake_gcloud_token())
    monkeypatch.setattr("subprocess.run", fake_run)

    get_access_token()
    get_access_token()

    fake_run.assert_called_once()


def test_get_access_token_refetches_after_cache_reset(monkeypatch):
    """Regression guard for the shared-cache test-isolation fix in
    tests/conftest.py's clear_driver_caches: resetting the module-level
    cache state must force a fresh fetch."""
    fake_run = MagicMock(return_value=_fake_gcloud_token())
    monkeypatch.setattr("subprocess.run", fake_run)

    get_access_token()
    gcloud_auth._cached_token = None
    gcloud_auth._token_fetched_at = 0.0
    get_access_token()

    assert fake_run.call_count == 2


def test_get_access_token_raises_if_gcloud_fails(monkeypatch):
    result = MagicMock(returncode=1, stderr="not logged in")
    monkeypatch.setattr("subprocess.run", MagicMock(return_value=result))

    with pytest.raises(RuntimeError, match="gcloud auth print-access-token failed"):
        get_access_token()


def test_get_access_token_passes_a_timeout_to_subprocess_run(monkeypatch):
    fake_run = MagicMock(return_value=_fake_gcloud_token())
    monkeypatch.setattr("subprocess.run", fake_run)

    get_access_token()

    assert (
        fake_run.call_args.kwargs.get("timeout") == gcloud_auth._GCLOUD_TIMEOUT_SECONDS
    )


def test_get_access_token_raises_transient_error_on_timeout(monkeypatch):
    """Regression for a real multi-hour hang (see docs/decisions.md): a
    hung gcloud subprocess used to block forever with no timeout at all.
    Must now surface as a retryable TransientAPIError, not hang or crash
    with an uncaught subprocess.TimeoutExpired."""
    monkeypatch.setattr(
        "subprocess.run",
        MagicMock(side_effect=subprocess.TimeoutExpired(cmd="gcloud", timeout=15)),
    )

    with pytest.raises(TransientAPIError, match="timed out"):
        get_access_token()


def test_credentials_expired_tracks_the_token_cache(monkeypatch):
    """Regression: google-genai checks ``credentials.expired`` (not ``valid``)
    to decide whether to call refresh(). With no ``expiry`` set it was always
    False, so a stale or invalidated token was never replaced."""
    monkeypatch.setattr("subprocess.run", MagicMock(return_value=_fake_gcloud_token()))
    creds = gcloud_auth.get_credentials()
    assert creds.expired is False

    gcloud_auth.invalidate()
    assert creds.expired is True


def test_credentials_refresh_fetches_a_new_token_after_invalidate(monkeypatch):
    fake_run = MagicMock(
        side_effect=[_fake_gcloud_token("old"), _fake_gcloud_token("new")]
    )
    monkeypatch.setattr("subprocess.run", fake_run)
    creds = gcloud_auth.get_credentials()
    assert creds.token == "old"

    gcloud_auth.invalidate()
    creds.refresh(request=None)

    assert creds.token == "new"
    assert creds.expired is False


def test_threads_that_find_the_token_stale_start_one_gcloud_not_one_each(monkeypatch):
    """With several workers the first calls arrive together: only one may refresh."""
    import threading
    import time

    calls = []

    def slow_gcloud(*args, **kwargs):
        calls.append(threading.current_thread().name)
        time.sleep(0.2)  # long enough for the other threads to arrive meanwhile
        return _fake_gcloud_token()

    monkeypatch.setattr("subprocess.run", slow_gcloud)
    tokens: list[str] = []
    threads = [
        threading.Thread(target=lambda: tokens.append(get_access_token()))
        for _ in range(8)
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert len(calls) == 1
    assert len(tokens) == 8 and len(set(tokens)) == 1
