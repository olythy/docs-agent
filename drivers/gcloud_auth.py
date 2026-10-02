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
"""

import subprocess
import time

_REFRESH_MARGIN_SECONDS = 300  # refresh 5 min before the ~1h expiry
_TOKEN_LIFETIME_SECONDS = 3600

_cached_token: str | None = None
_token_fetched_at: float = 0.0


def get_access_token() -> str:
    """Return a cached OAuth access token, refreshing it if stale.

    Returns:
        A bearer token string, from the already-authenticated ``gcloud``
        CLI session.

    Raises:
        RuntimeError: If ``gcloud auth print-access-token`` fails (e.g.
            not logged in).
    """
    global _cached_token, _token_fetched_at

    age = time.monotonic() - _token_fetched_at
    if _cached_token is None or age > (_TOKEN_LIFETIME_SECONDS - _REFRESH_MARGIN_SECONDS):
        result = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"gcloud auth print-access-token failed: {result.stderr}")
        _cached_token = result.stdout.strip()
        _token_fetched_at = time.monotonic()
    return _cached_token


def get_credentials():
    """Return a ``google.auth.credentials.Credentials`` backed by :func:`get_access_token`.

    ``valid`` always reports ``False`` so any caller that checks it before
    using the token (e.g. ``google-genai``'s request-signing logic) always
    calls ``refresh()`` first -- which just delegates to
    :func:`get_access_token`'s own freshness check, the single source of
    truth for whether a real ``gcloud`` subprocess call is actually needed.
    Without this override, the base class's default ``valid`` becomes
    ``True`` forever once a token is set (since ``expiry`` is never set
    here), and nothing would ever pick up a refreshed token again.

    Returns:
        A ``Credentials`` instance usable as the ``credentials=`` argument
        to ``genai.Client(vertexai=True, ...)``.
    """
    import google.auth.credentials

    class _GcloudCliCredentials(google.auth.credentials.Credentials):
        @property
        def valid(self) -> bool:
            return False

        def refresh(self, request) -> None:
            self.token = get_access_token()

    creds = _GcloudCliCredentials()
    creds.token = get_access_token()
    return creds
