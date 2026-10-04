"""Cached Google Cloud OAuth access token, shared by every Vertex AI driver.

Vertex AI authenticates via OAuth access tokens, not a static API key, so
there's no ``VERTEX_API_KEY`` setting anywhere in this project. Every
Vertex-backed driver (:class:`drivers.embedding.VertexEmbeddingDriver`,
:class:`drivers.reranker.VertexRankerDriver`) shells out to the already-
authenticated ``gcloud`` CLI session (``gcloud auth print-access-token``)
instead of requiring a separate Application Default Credentials setup,
which would need its own interactive browser login.

Key exports:
    get_access_token -- Return a cached token, refreshed shortly before
        its ~1-hour expiry. Cached at module level (not per-driver-instance)
        so multiple Vertex drivers in the same process share one token and
        one ``gcloud`` subprocess call, not one each.
    get_credentials -- A ``google.auth.credentials.Credentials`` wrapper
        around ``get_access_token``, for SDKs (e.g. ``google-genai``'s
        ``vertexai=True`` mode) that want a credentials object rather than
        a raw bearer string -- confirmed live that this works without a
        separate ``gcloud auth application-default login`` setup, which
        would need its own interactive browser login.
    invalidate -- Force the next ``get_access_token()`` call to actually
        refetch, instead of trusting the ~1-hour assumption. Confirmed
        live, during a real multi-hour bulk ingest: a token can stop
        working before that assumption expects, and every driver's retry
        handling calls this on a 401 before retrying -- retrying with the
        *same* still-cached token would just fail identically.
"""

import subprocess
import time

from retry_policy import TransientAPIError

_REFRESH_MARGIN_SECONDS = 300  # refresh 5 min before the ~1h expiry
_TOKEN_LIFETIME_SECONDS = 3600
#: Confirmed live during a real multi-hour bulk ingest: subprocess.run()
#: here had no timeout at all, so a single hung `gcloud` invocation (the
#: process went to 0% CPU and made zero progress for over an hour) froze
#: the entire ingest with no way to recover automatically. Bounding it
#: turns that into a TransientAPIError the caller's own
#: @retry_on_transient_error already handles -- same backoff-and-retry
#: path as a 401, not a new recovery mechanism.
_GCLOUD_TIMEOUT_SECONDS = 15

_cached_token: str | None = None
_token_fetched_at: float = 0.0


def get_access_token() -> str:
    """Return a cached OAuth access token, refreshing it if stale.

    Returns:
        A bearer token string, from the already-authenticated ``gcloud``
        CLI session.

    Raises:
        RuntimeError: If ``gcloud auth print-access-token`` fails (e.g.
            not logged in) -- not retryable, since retrying a bad login
            state would just fail identically.
        TransientAPIError: If the ``gcloud`` subprocess doesn't finish
            within ``_GCLOUD_TIMEOUT_SECONDS`` -- the caller's own
            ``@retry_on_transient_error`` handles backoff and retry, same
            as any other transient failure.
    """
    global _cached_token, _token_fetched_at

    if is_stale():
        try:
            result = subprocess.run(
                ["gcloud", "auth", "print-access-token"],
                capture_output=True,
                text=True,
                check=False,
                timeout=_GCLOUD_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise TransientAPIError(
                f"gcloud auth print-access-token timed out after "
                f"{_GCLOUD_TIMEOUT_SECONDS}s"
            ) from exc
        if result.returncode != 0:
            raise RuntimeError(f"gcloud auth print-access-token failed: {result.stderr}")
        _cached_token = result.stdout.strip()
        _token_fetched_at = time.monotonic()
    assert _cached_token is not None
    return _cached_token


def is_stale() -> bool:
    """Whether the cached token is missing, invalidated, or near its expiry."""
    age = time.monotonic() - _token_fetched_at
    return _cached_token is None or age > (
        _TOKEN_LIFETIME_SECONDS - _REFRESH_MARGIN_SECONDS
    )


def invalidate() -> None:
    """Force the next :func:`get_access_token` call to refetch a token."""
    global _cached_token, _token_fetched_at
    _cached_token = None
    _token_fetched_at = 0.0


def get_credentials():
    """Return a ``google.auth.credentials.Credentials`` backed by :func:`get_access_token`.

    ``expired`` (and ``valid``) are driven by :func:`is_stale` instead of
    the base class's ``expiry`` timestamp, which is never set here. Confirmed
    live (a 401 that kept recurring despite an earlier ``valid=False``
    override) that ``google-genai`` decides whether to call ``refresh()``
    from ``credentials.expired``, not ``valid`` -- and the base class's
    ``expired`` is ``False`` forever when ``expiry`` is ``None``, so the
    token was never refreshed after the first one, and
    :func:`invalidate` on a 401 had no effect either. ``refresh()`` just
    delegates to :func:`get_access_token`, the single source of truth for
    whether a real ``gcloud`` subprocess call is actually needed.

    Returns:
        A ``Credentials`` instance usable as the ``credentials=`` argument
        to ``genai.Client(vertexai=True, ...)``.
    """
    import google.auth.credentials

    class _GcloudCliCredentials(google.auth.credentials.Credentials):
        @property
        def expired(self) -> bool:
            return is_stale()

        @property
        def valid(self) -> bool:
            return not is_stale()

        def refresh(self, request) -> None:
            self.token = get_access_token()

    creds = _GcloudCliCredentials()
    creds.token = get_access_token()
    return creds
