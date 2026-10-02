"""Cached Google Cloud OAuth access token, shared by every Vertex AI driver.

Vertex AI authenticates via OAuth access tokens, not a static API key, so
there's no ``VERTEX_API_KEY`` setting anywhere in this project. Every
Vertex-backed driver (:class:`drivers.embedding.VertexEmbeddingDriver`,
:class:`drivers.reranker.VertexRankerDriver`) shells out to the already-
authenticated ``gcloud`` CLI session (``gcloud auth print-access-token``)
instead of requiring a separate Application Default Credentials setup,
which would need its own interactive browser login.

Key export:
    get_access_token -- Return a cached token, refreshed shortly before
        its ~1-hour expiry. Cached at module level (not per-driver-instance)
        so multiple Vertex drivers in the same process share one token and
        one ``gcloud`` subprocess call, not one each.
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
