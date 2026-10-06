"""Shared retry-with-backoff policy for outbound calls to rate-limited external APIs.

Exports:
    - ``is_retryable_status()``: which API error statuses are retried (429, 499, 5xx).
    - ``TransientAPIError``: the exception a call site raises to mark a
      failure as retryable (network error, 429, 5xx).
    - ``retry_on_transient_error()``: a ``tenacity`` decorator factory with
      this project's standard policy (exponential backoff doubling from a
      configurable initial wait). Used by every embedding driver
      (``drivers/embedding.py``) and the court-decision downloader
      (``corpus/download_court_decisions.py``), so the retry/backoff policy
      lives in one place instead of being hand-rolled per call site.
"""

import logging

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


#: HTTP statuses retried like a server error: 429 (rate limit) and 499 (``CANCELLED``:
#: confirmed live once, in the middle of a run, and gone on the very next try).
RETRYABLE_STATUSES = frozenset({429, 499})


def is_retryable_status(status: int | None) -> bool:
    """Whether an API error status is worth retrying: 429, 499 or any 5xx.

    Args:
        status: The HTTP status code of the error, if it has one.
    """
    return status is not None and (status in RETRYABLE_STATUSES or status >= 500)


class TransientAPIError(OSError):
    """Marks a failure as retryable (network error, HTTP 429, or 5xx).

    Subclasses ``OSError`` so it's still caught by callers that already
    handle ``OSError`` around outbound calls (e.g.
    ``ingestion.ingest.add_directory``'s per-file error handling) without
    needing to know about this type specifically — matching
    ``requests.RequestException``, which is also an ``OSError`` subclass and
    was what callers previously saw here.
    """


def retry_on_transient_error(max_attempts: int = 3, initial_wait: float = 2.0):
    """Build a ``tenacity`` retry decorator for ``TransientAPIError``.

    Retries only ``TransientAPIError`` (raised explicitly by the decorated
    function for a retryable failure) with exponential backoff — any other
    exception is a real, non-retryable error and propagates immediately.

    Args:
        max_attempts: Total attempts including the first (default 3).
        initial_wait: Seconds before the first retry, doubling after each
            subsequent one (default 2.0 -> 2s, 4s, 8s, ...).

    Returns:
        A decorator to apply to a function that raises ``TransientAPIError``
        on a retryable failure and lets any other exception propagate.
    """
    return retry(
        retry=retry_if_exception_type(TransientAPIError),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=initial_wait, min=initial_wait),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
