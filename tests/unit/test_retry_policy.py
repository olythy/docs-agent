"""Tests for retry_policy.py: the shared retry-with-backoff decorator."""

from unittest.mock import MagicMock

import pytest

from retry_policy import TransientAPIError, retry_on_transient_error


def test_transient_api_error_is_an_os_error():
    """Callers that already catch OSError around outbound calls (e.g.
    ingestion.ingest.add_directory's per-file error handling) must catch
    this without needing to know about it specifically."""
    assert isinstance(TransientAPIError("boom"), OSError)


def test_succeeds_on_first_attempt_without_sleeping(monkeypatch):
    fake_sleep = MagicMock()
    monkeypatch.setattr("time.sleep", fake_sleep)
    calls = []

    @retry_on_transient_error(max_attempts=3, initial_wait=2.0)
    def flaky():
        calls.append(1)
        return "ok"

    assert flaky() == "ok"
    assert len(calls) == 1
    fake_sleep.assert_not_called()


def test_retries_transient_error_then_succeeds(monkeypatch):
    fake_sleep = MagicMock()
    monkeypatch.setattr("time.sleep", fake_sleep)
    attempts = {"count": 0}

    @retry_on_transient_error(max_attempts=3, initial_wait=2.0)
    def flaky():
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise TransientAPIError("transient")
        return "ok"

    assert flaky() == "ok"
    assert attempts["count"] == 2
    fake_sleep.assert_called_once_with(2.0)


def test_raises_after_exhausting_all_attempts(monkeypatch):
    fake_sleep = MagicMock()
    monkeypatch.setattr("time.sleep", fake_sleep)
    attempts = {"count": 0}

    @retry_on_transient_error(max_attempts=3, initial_wait=2.0)
    def always_fails():
        attempts["count"] += 1
        raise TransientAPIError("still broken")

    with pytest.raises(TransientAPIError, match="still broken"):
        always_fails()

    assert attempts["count"] == 3
    assert fake_sleep.call_args_list == [((2.0,),), ((4.0,),)]


def test_non_transient_error_propagates_immediately(monkeypatch):
    fake_sleep = MagicMock()
    monkeypatch.setattr("time.sleep", fake_sleep)
    attempts = {"count": 0}

    @retry_on_transient_error(max_attempts=3, initial_wait=2.0)
    def raises_value_error():
        attempts["count"] += 1
        raise ValueError("not retryable")

    with pytest.raises(ValueError, match="not retryable"):
        raises_value_error()

    assert attempts["count"] == 1
    fake_sleep.assert_not_called()
