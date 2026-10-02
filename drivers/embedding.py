"""Embedding driver abstractions.

Defines the common interface (``EmbeddingDriver``) that all embedding backends
must implement, following the Strategy / Driver pattern described in AGENTS.md.

The active driver is selected at runtime via ``settings.EMBEDDING_DRIVER``:
    - ``"local"``      → :class:`LocalSentenceTransformerDriver` (free, offline)
    - ``"openai"``     → :class:`OpenAIEmbeddingDriver` (paid API, direct OpenAI)
    - ``"openrouter"`` → :class:`OpenRouterEmbeddingDriver` (paid API, routed
                         through OpenRouter — e.g. Google's
                         ``google/gemini-embedding-001``, same account/key
                         shape as ``LLM_DRIVER=openrouter``)
    - ``"gemini"``     → :class:`GeminiEmbeddingDriver` (Google's native AI
                         Studio API directly, not via OpenRouter — free-tier
                         friendly, rate-limited via
                         ``EMBEDDING_REQUEST_DELAY_SECONDS``)
    - ``"jina"``       → :class:`JinaEmbeddingDriver` (Jina AI's Embeddings
                         API — a hosted alternative to running the local
                         model, with no GPU/infra to manage)

Usage::

    from drivers.embedding import get_embedding_driver
    driver = get_embedding_driver()
    vector = driver.embed_text("Mennyi az SZJA tartozásom?")
"""

import logging
from abc import ABC, abstractmethod
from functools import lru_cache

from config import settings
from retry_policy import TransientAPIError, retry_on_transient_error

logger = logging.getLogger(__name__)

_TOKEN_LENGTH_WARNING_SUPPRESSED = False


class _SuppressTokenLengthWarning(logging.Filter):
    """Drops transformers' "Token indices sequence length is longer than..." log line."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "Token indices sequence length is longer" not in record.getMessage()


def _suppress_token_length_warning() -> None:
    """Silence one specific, known-benign transformers warning, once.

    ``LocalSentenceTransformerDriver.count_tokens()`` deliberately tokenizes
    without truncation to detect real overflow (see its docstring) — the
    warning's "will result in indexing errors" caveat doesn't apply here,
    since the over-length result is only ever used for counting, never fed
    through the model. Left unsuppressed, it clutters any script that calls
    ``count_tokens()`` a lot (e.g. ``scripts/inspect_chunks.py``): it's
    emitted via ``logging`` straight to stderr rather than raised as a
    catchable ``warnings.warn()``, so it interleaves with normal stdout
    output instead of being collectible in one place.
    """
    global _TOKEN_LENGTH_WARNING_SUPPRESSED
    if _TOKEN_LENGTH_WARNING_SUPPRESSED:
        return
    logging.getLogger("transformers.tokenization_utils_base").addFilter(
        _SuppressTokenLengthWarning()
    )
    _TOKEN_LENGTH_WARNING_SUPPRESSED = True


class EmbeddingDriver(ABC):
    """Abstract base class for all embedding backends.

    Subclasses must implement :meth:`embed_text` and :meth:`embed_batch`.
    The :attr:`dimension` property must match the vector size expected by the
    ``document_chunks.embedding`` column in Postgres (``vector(N)``).
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Output vector dimension (must match the pgvector column size)."""

    def embed_text(self, text: str) -> list[float]:
        """Embed a single string into a dense float vector.

        Default implementation delegates to :meth:`embed_batch` — every
        driver's real work happens there, so subclasses only need to
        override this if they can do meaningfully better for a single
        string (none currently do).

        Args:
            text: The input text to embed.

        Returns:
            A list of floats with length equal to :attr:`dimension`.
        """
        return self.embed_batch([text])[0]

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings in one batched call.

        Batching is significantly more efficient than calling :meth:`embed_text`
        in a loop, especially for the local sentence-transformer model.

        Args:
            texts: A list of input strings.

        Returns:
            A list of float vectors, one per input string.
        """

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query string into a dense float vector.

        Default implementation delegates to :meth:`embed_text`. Asymmetric models
        (such as E5 or BGE) override this to prepend instruction prefixes (e.g. 'query: ').

        Args:
            text: The search query text.

        Returns:
            A list of floats with length equal to :attr:`dimension`.
        """
        return self.embed_text(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document chunk strings in one batched call.

        Default implementation delegates to :meth:`embed_batch`. Asymmetric models
        (such as E5) override this to prepend passage prefixes (e.g. 'passage: ').

        Args:
            texts: List of document chunk strings.

        Returns:
            A list of float vectors, one per input string.
        """
        return self.embed_batch(texts)

    def max_sequence_length(self) -> int | None:
        """Return this driver's maximum input length in tokens, if known.

        Text beyond this length is silently truncated during embedding —
        the truncated tail is invisible to similarity search, which can
        badly hurt retrieval quality without ever raising an error.

        Returns:
            The model's max sequence length in tokens, or ``None`` if this
            driver has no practical limit worth checking against (the
            default; e.g. remote APIs with very generous limits).
        """
        return None

    def count_tokens(self, text: str) -> int | None:
        """Return the real (untruncated) token count for ``text``, if this driver can.

        Used by ``CHUNK_OVERFLOW_STRATEGY=split`` to decide whether a chunk
        actually overflows :meth:`max_sequence_length`, instead of the
        ``warn`` strategy's word-count estimate. This is ground truth, not
        an approximation — so it only makes sense for drivers that expose
        their own tokenizer.

        Returns:
            The exact token count, or ``None`` if this driver has no local
            tokenizer to count with (the default; e.g. remote APIs).
        """
        return None

    def supports_token_counting(self) -> bool:
        """Return True if :meth:`count_tokens` gives a real, non-None result.

        A capability check, not a side-effecting probe — callers (e.g.
        :class:`ingestion.chunker.SplitOverflowStrategy`) used to detect this
        by calling ``count_tokens()`` on a sample chunk and checking for
        ``None``, which meant no answer at all for an empty chunk list, and
        an unnecessary tokenizer call just to check a capability.

        Returns:
            False by default; overridden by drivers whose :meth:`count_tokens`
            actually works.
        """
        return False


class LocalSentenceTransformerDriver(EmbeddingDriver):
    """Embedding driver using a locally downloaded sentence-transformer model.

    Runs entirely offline after the model is downloaded on first use (~470 MB).
    Recommended for development and for Hungarian-language documents.

    The model is loaded lazily on the first call to avoid import-time cost.
    """

    def __init__(self, model_name: str | None = None) -> None:
        """Initialise the driver without loading the model yet.

        Args:
            model_name: HuggingFace model identifier. Defaults to
                ``settings.EMBEDDING_MODEL``.
        """
        self._model_name = model_name or settings.EMBEDDING_MODEL
        self._model = None  # Loaded lazily on first embed call
        is_e5 = "e5" in self._model_name.lower()
        self._query_prefix = "query: " if is_e5 else ""
        self._passage_prefix = "passage: " if is_e5 else ""

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query, prepending 'query: ' if using an E5 model.

        Args:
            text: The search query text.

        Returns:
            Dense float vector representing the query.
        """
        if self._query_prefix and not text.startswith(self._query_prefix):
            text = f"{self._query_prefix}{text}"
        return self.embed_text(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document chunk strings, prepending 'passage: ' if using an E5 model.

        Args:
            texts: List of document chunk strings.

        Returns:
            List of float vectors, one per document chunk.
        """
        if self._passage_prefix:
            texts = [
                f"{self._passage_prefix}{t}"
                if not t.startswith(self._passage_prefix)
                else t
                for t in texts
            ]
        return self.embed_batch(texts)

    def _get_model(self):
        """Load and cache the sentence-transformer model.

        Returns:
            The loaded ``SentenceTransformer`` instance.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    @property
    def dimension(self) -> int:
        """Return the embedding dimension from settings."""
        return settings.EMBEDDING_DIMENSION

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings in one batched inference call.

        Args:
            texts: A list of input strings.

        Returns:
            A list of float vectors, one per input string.
        """
        model = self._get_model()
        # convert_to_python=True returns plain Python lists instead of tensors
        embeddings = model.encode(texts, convert_to_numpy=True)
        return [e.tolist() for e in embeddings]

    def max_sequence_length(self) -> int | None:
        """Return the loaded model's max sequence length in tokens.

        Loads the model if it isn't already loaded — but this is only ever
        called right before :meth:`embed_batch` in the ingestion pipeline,
        which was going to load it anyway, so this never triggers an extra
        model load on its own.
        """
        return self._get_model().max_seq_length

    def count_tokens(self, text: str) -> int | None:
        """Return the real token count from the model's own tokenizer.

        Calls the underlying HuggingFace tokenizer directly, with no
        ``truncation``/``max_length`` argument — confirmed empirically that
        ``model.tokenize()`` (the method ``encode()`` uses internally)
        already truncates to ``max_seq_length``, which would make overflow
        undetectable. Calling the raw tokenizer instead reports the true,
        untruncated length.
        """
        _suppress_token_length_warning()
        model = self._get_model()
        return len(model.tokenizer(text)["input_ids"])

    def supports_token_counting(self) -> bool:
        """This driver's count_tokens() always works — see its docstring."""
        return True


class OpenAIEmbeddingDriver(EmbeddingDriver):
    """Embedding driver using the OpenAI Embeddings API.

    Requires a valid ``EMBEDDING_API_KEY`` in settings.
    Suitable for production use or when English content quality matters most.
    """

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        """Initialise the OpenAI driver.

        Args:
            model: The OpenAI embedding model to use.
        """
        self._model = model

    @property
    def dimension(self) -> int:
        """Return the embedding dimension from settings."""
        return settings.EMBEDDING_DIMENSION

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings in one OpenAI API call.

        Passes ``dimensions=self.dimension`` explicitly — without it, the
        ``text-embedding-3-*`` models return their native size (1536 for
        ``text-embedding-3-small``) regardless of ``EMBEDDING_DIMENSION``,
        silently breaking the contract :attr:`dimension` claims to honor:
        ``document_chunks.embedding`` is a fixed-width ``vector(N)`` column
        (``N`` = ``EMBEDDING_DIMENSION`` at migration time), so a native-size
        vector would fail to insert with a dimension mismatch the moment
        ``EMBEDDING_DIMENSION`` differs from the model's native size (which
        it does at this project's default of 384).

        Args:
            texts: A list of input strings.

        Returns:
            A list of float vectors, one per input string, each of length
            :attr:`dimension`.
        """
        from openai import OpenAI

        client = OpenAI(api_key=settings.EMBEDDING_API_KEY)
        response = client.embeddings.create(
            input=texts, model=self._model, dimensions=self.dimension
        )
        return [item.embedding for item in response.data]


class OpenRouterEmbeddingDriver(EmbeddingDriver):
    """Embedding driver using OpenRouter's ``/api/v1/embeddings`` endpoint.

    Routes to any embedding model OpenRouter offers — e.g. Google's
    ``google/gemini-embedding-001`` — through the same OpenRouter account
    ``LLM_DRIVER=openrouter`` already uses for answer generation, so no
    separate provider account/API key is needed (just set
    ``EMBEDDING_API_KEY`` to the same OpenRouter key as ``LLM_API_KEY``).

    Confirmed empirically against the real endpoint: passing OpenRouter's
    ``dimensions`` request parameter truncates a model's native output
    (3072 for ``gemini-embedding-001``) down to whatever
    ``EMBEDDING_DIMENSION`` is configured — so switching to this driver at
    the project's default (384) needs no ``document_chunks`` schema
    migration, unlike a driver whose native dimension doesn't match.

    No token counting is exposed (:meth:`count_tokens`/:meth:`max_sequence_length`
    stay the ABC's ``None`` defaults, same as :class:`OpenAIEmbeddingDriver`)
    — set ``CHUNKING_STRATEGY=word`` rather than ``langchain`` when using
    this driver, confirmed empirically: without a real tokenizer,
    ``langchain`` silently measures ``CHUNK_SIZE`` in raw *characters*
    instead of words, producing far smaller/more numerous chunks than
    intended (one real document went from the expected ~15 chunks to 407).
    """

    #: Google's embedding API (reached via OpenRouter, backed by Vertex AI)
    #: hard-rejects a batch outside 1-250 items — confirmed empirically
    #: against the real endpoint (HTTP 400, "batchSize value of 300 but
    #: the supported range is from 1 (inclusive) to 251 (exclusive)").
    _MAX_BATCH_SIZE = 250

    def __init__(self, model: str | None = None) -> None:
        """Initialise the driver.

        Args:
            model: OpenRouter embedding model id. Defaults to
                ``settings.EMBEDDING_MODEL``.
        """
        self._model = model or settings.EMBEDDING_MODEL

    @property
    def dimension(self) -> int:
        """Return the embedding dimension from settings."""
        return settings.EMBEDDING_DIMENSION

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings, splitting into sub-batches of at most 250.

        Args:
            texts: A list of input strings.

        Returns:
            A list of float vectors, one per input string (same order),
            each truncated to :attr:`dimension`.
        """
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self._MAX_BATCH_SIZE):
            embeddings.extend(
                self._embed_one_batch(texts[start : start + self._MAX_BATCH_SIZE])
            )
        return embeddings

    @retry_on_transient_error(max_attempts=3)
    def _embed_one_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed at most ``_MAX_BATCH_SIZE`` strings in one OpenRouter API call.

        Decorated with :func:`retry.retry_on_transient_error`: a network
        error, 429, or 5xx raises :class:`retry.TransientAPIError`, which
        triggers a retry with exponential backoff (up to 3 attempts) — a
        4xx other than 429 (e.g. the batch-size-limit 400
        :attr:`_MAX_BATCH_SIZE` exists to avoid) is a real request error,
        not a transient one, and propagates immediately without retrying.

        Raises:
            TransientAPIError: If every retry is exhausted.
            requests.HTTPError: On a non-retryable HTTP error status.
        """
        import requests

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {settings.EMBEDDING_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "input": texts,
                    "dimensions": self.dimension,
                },
                timeout=90,
            )
        except requests.RequestException as exc:
            raise TransientAPIError(str(exc)) from exc

        if response.status_code == 429 or response.status_code >= 500:
            raise TransientAPIError(
                f"OpenRouter embeddings request status {response.status_code}"
            )

        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]


class GeminiEmbeddingDriver(EmbeddingDriver):
    """Embedding driver using Google's native Gemini API (AI Studio) directly, not via OpenRouter.

    For a free-tier AI Studio API key, which carries its own
    requests-per-minute limit distinct from any paid quota. Throttled via
    a real sleep (``EMBEDDING_REQUEST_DELAY_SECONDS``) before each
    request, since the actual current free-tier limit is account/tier-
    specific and changes over time — check your own AI Studio quota page
    rather than trusting a number hardcoded here.

    Same 250-item batch limit as :class:`OpenRouterEmbeddingDriver`
    (the same underlying Google backend serves both paths) and the same
    ``EMBEDDING_DIMENSION`` truncation behavior, via this API's own
    ``output_dimensionality`` config instead of OpenRouter's ``dimensions``
    request parameter.
    """

    _MAX_BATCH_SIZE = 250

    def __init__(self, model: str | None = None) -> None:
        """Initialise the driver without creating the client yet.

        Args:
            model: Gemini embedding model id. Defaults to
                ``settings.EMBEDDING_MODEL``.
        """
        self._model = model or settings.EMBEDDING_MODEL
        self._client = None  # Created lazily on first embed call

    def _get_client(self):
        """Create and cache the ``google-genai`` client.

        Returns:
            The ``genai.Client`` instance.
        """
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=settings.EMBEDDING_API_KEY)
        return self._client

    @property
    def dimension(self) -> int:
        """Return the embedding dimension from settings."""
        return settings.EMBEDDING_DIMENSION

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings, splitting into sub-batches of at most 250.

        Args:
            texts: A list of input strings.

        Returns:
            A list of float vectors, one per input string (same order),
            each truncated to :attr:`dimension`.
        """
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self._MAX_BATCH_SIZE):
            embeddings.extend(
                self._embed_one_batch(texts[start : start + self._MAX_BATCH_SIZE])
            )
        return embeddings

    @retry_on_transient_error(max_attempts=3)
    def _embed_one_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed at most ``_MAX_BATCH_SIZE`` strings in one Gemini API call.

        Sleeps ``EMBEDDING_REQUEST_DELAY_SECONDS`` before every attempt
        (including retries, since the whole function re-runs from the top
        on each retry) to respect a free-tier rate limit. Decorated with
        :func:`retry.retry_on_transient_error`: a 429 (rate-limited) or 5xx
        response raises :class:`retry.TransientAPIError`, triggering a
        retry with exponential backoff (up to 3 attempts), same policy as
        :meth:`OpenRouterEmbeddingDriver._embed_one_batch`; any other error
        propagates immediately.

        Also retries ``httpx.TransportError`` (connection failures, DNS
        errors, timeouts — confirmed live with a real "No route to host"
        mid-ingestion, a transient WiFi drop) the same way — the
        ``google-genai`` SDK uses ``httpx`` internally, and this exception
        is a completely different hierarchy from ``APIError`` (Gemini's own
        API-level error type), so it was previously not retried at all and
        crashed the whole ``add_directory()`` run instead of just this one
        batch, the same class of bug already fixed for
        :meth:`OpenRouterEmbeddingDriver._embed_one_batch` via
        ``requests.RequestException``.

        Raises:
            TransientAPIError: If every retry is exhausted.
            Exception: Whatever the ``google-genai`` client raises, for a
                non-retryable error.
        """
        import time
        from typing import cast

        import httpx
        from google.genai import types
        from google.genai.errors import APIError

        if settings.EMBEDDING_REQUEST_DELAY_SECONDS > 0:
            time.sleep(settings.EMBEDDING_REQUEST_DELAY_SECONDS)

        client = self._get_client()
        try:
            response = client.models.embed_content(
                model=self._model,
                # google-genai's ContentListUnion stub doesn't include a
                # plain list[str] (list invariance: list[str] isn't a
                # list[str | Image | File | Part]), but a plain string list
                # is confirmed working live against the real API — this
                # cast documents that the stub is narrower than reality,
                # not a guess.
                contents=cast("types.ContentListUnion", texts),
                config=types.EmbedContentConfig(output_dimensionality=self.dimension),
            )
        except httpx.TransportError as exc:
            raise TransientAPIError(f"Gemini embeddings network error: {exc}") from exc
        except APIError as exc:
            status = getattr(exc, "code", None)
            if status == 429 or (status is not None and status >= 500):
                raise TransientAPIError(
                    f"Gemini embeddings request status {status}"
                ) from exc
            raise

        assert response.embeddings is not None, (
            "a successful embed_content() response always has embeddings"
        )
        embeddings = []
        for item in response.embeddings:
            assert item.values is not None, (
                "a successful embed_content() response always has values "
                "per embedding item"
            )
            embeddings.append(item.values)
        return embeddings


class JinaEmbeddingDriver(EmbeddingDriver):
    """Embedding driver using Jina AI's hosted Embeddings API.

    A fully-managed alternative to :class:`LocalSentenceTransformerDriver`:
    no model to load, no CPU/GPU to provision — chosen specifically to
    offload the CPU-bound embedding cost confirmed live during ingestion
    profiling (~0.28s/chunk on the local model, ~90%+ of per-document
    ingest time; see docs/decisions.md). Defaults to
    ``settings.EMBEDDING_MODEL``, which should be a Jina model id (e.g.
    ``jina-embeddings-v3``) when this driver is active -- the default
    value of that setting is the local model's name and only applies when
    ``EMBEDDING_DRIVER=local``.

    Uses the ``task`` request parameter to distinguish query vs. document
    embedding (``retrieval.query``/``retrieval.passage``), the API-level
    equivalent of :class:`LocalSentenceTransformerDriver`'s
    ``"query: "``/``"passage: "`` text prefixes for E5-family models --
    same asymmetric-embedding concept, just expressed as a parameter
    instead of a string prefix since Jina's models expect it that way.

    Throttled via a real sleep (``EMBEDDING_REQUEST_DELAY_SECONDS``)
    before each request, same mechanism as :class:`GeminiEmbeddingDriver`
    -- confirmed live during a real bulk `add_directory()` run that a
    plain per-document ingest loop (no batching across documents) can
    exceed a Jina account's *token*-per-minute limit (not just a
    requests-per-minute one) well before the 3-attempt retry's ~14s of
    total backoff lets the per-minute window clear, even though each
    individual request is well within the per-call batch size limit --
    the cap is cumulative across many small calls in the same minute, not
    about any single call being too large.
    """

    _ENDPOINT = "https://api.jina.ai/v1/embeddings"
    _MAX_BATCH_SIZE = 2048

    def __init__(self, model: str | None = None) -> None:
        """Initialise the driver without creating an HTTP client yet.

        Args:
            model: Jina embedding model id. Defaults to ``settings.EMBEDDING_MODEL``.
        """
        self._model = model or settings.EMBEDDING_MODEL

    @property
    def dimension(self) -> int:
        """Return the embedding dimension from settings."""
        return settings.EMBEDDING_DIMENSION

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings with ``task="retrieval.passage"``.

        Most callers go through :meth:`embed_documents`/:meth:`embed_query`
        instead, which set the correct ``task`` for their use -- this
        default matches :class:`EmbeddingDriver`'s base ``embed_text``
        delegating here, where there's no query/document distinction to
        make.

        Args:
            texts: A list of input strings.

        Returns:
            A list of float vectors, one per input string (same order).
        """
        return self._embed(texts, task="retrieval.passage")

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query with ``task="retrieval.query"``."""
        return self._embed([text], task="retrieval.query")[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document chunks with ``task="retrieval.passage"``, batched."""
        return self._embed(texts, task="retrieval.passage")

    def _embed(self, texts: list[str], task: str) -> list[list[float]]:
        """Embed ``texts`` in sub-batches of at most ``_MAX_BATCH_SIZE``.

        Args:
            texts: A list of input strings.
            task: Jina's ``task`` parameter -- ``retrieval.query`` or
                ``retrieval.passage``.

        Returns:
            A list of float vectors, one per input string (same order).
        """
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self._MAX_BATCH_SIZE):
            embeddings.extend(
                self._embed_one_batch(texts[start : start + self._MAX_BATCH_SIZE], task)
            )
        return embeddings

    @retry_on_transient_error(max_attempts=3)
    def _embed_one_batch(self, texts: list[str], task: str) -> list[list[float]]:
        """Embed at most ``_MAX_BATCH_SIZE`` strings in one Jina API call.

        Sleeps ``EMBEDDING_REQUEST_DELAY_SECONDS`` before every attempt
        (including retries, since the whole function re-runs from the top
        on each retry) to stay under the account's token-per-minute limit.

        Raises:
            TransientAPIError: On a 429/5xx response or a network-level
                failure, triggering a retry with exponential backoff (up
                to 3 attempts) -- same policy as
                :meth:`GeminiEmbeddingDriver._embed_one_batch`.
        """
        import time

        import httpx

        if settings.EMBEDDING_REQUEST_DELAY_SECONDS > 0:
            time.sleep(settings.EMBEDDING_REQUEST_DELAY_SECONDS)

        try:
            response = httpx.post(
                self._ENDPOINT,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {settings.EMBEDDING_API_KEY}",
                    "Accept": "application/json",
                },
                json={
                    "model": self._model,
                    "task": task,
                    "dimensions": self.dimension,
                    "input": texts,
                },
                timeout=60.0,
            )
        except httpx.TransportError as exc:
            raise TransientAPIError(f"Jina embeddings network error: {exc}") from exc

        if response.status_code == 429 or response.status_code >= 500:
            raise TransientAPIError(
                f"Jina embeddings request status {response.status_code}: {response.text}"
            )
        response.raise_for_status()

        data = response.json()["data"]
        ordered = sorted(data, key=lambda item: item["index"])
        return [item["embedding"] for item in ordered]


class VertexEmbeddingDriver(EmbeddingDriver):
    """Embedding driver using Google Cloud's Vertex AI text embedding API.

    Same no-infra motivation as :class:`JinaEmbeddingDriver`, added right
    after it: confirmed live that a real bulk ingest kept hitting Jina's
    free-tier tokens-per-minute cap even with ``EMBEDDING_REQUEST_DELAY_SECONDS``
    throttling, while 20 rapid, undelayed Vertex AI requests in a row all
    succeeded on this project's default quota -- billed against GCP credit
    instead of needing its own separate rate-limit workaround.

    Authenticates via the already-authenticated ``gcloud`` CLI session
    (``gcloud auth print-access-token``) rather than a static API key or a
    separate Application Default Credentials setup -- there's no
    ``VERTEX_API_KEY`` setting because Vertex AI doesn't authenticate that
    way; it's OAuth-token-based. The token is cached in memory and
    refreshed shortly before its ~1-hour expiry, so most calls don't pay
    the cost of spawning a ``gcloud`` subprocess.

    Uses the per-instance ``task_type`` field (``RETRIEVAL_QUERY``/
    ``RETRIEVAL_DOCUMENT``) to distinguish query vs. document embedding --
    the same asymmetric-embedding concept as :class:`JinaEmbeddingDriver`'s
    ``task`` parameter, just shaped per-instance instead of per-request
    since that's how this API expects it.
    """

    _MAX_BATCH_SIZE = 250
    # Confirmed live on real Hungarian legal text: ~5 tokens/word for this
    # model's tokenizer (22,276 tokens / 4,484 words) -- a completely
    # different ratio from the project's general WORDS_PER_TOKEN=0.75
    # (~1.33 tokens/word), which is calibrated for a different tokenizer
    # and is the wrong number for this API, not just an under-margined one.
    _ESTIMATED_TOKENS_PER_WORD = 5.5
    # The API's real cap is 20,000; kept under it with margin for
    # estimation error on text that tokenizes even worse than our sample.
    _MAX_TOKENS_PER_BATCH = 15000
    _TOKEN_REFRESH_MARGIN_SECONDS = 300  # refresh 5 min before the ~1h expiry

    def __init__(self, model: str | None = None) -> None:
        """Initialise the driver without fetching an access token yet.

        Args:
            model: Vertex AI text embedding model id. Defaults to
                ``settings.EMBEDDING_MODEL``.
        """
        self._model = model or settings.EMBEDDING_MODEL
        self._cached_token: str | None = None
        self._token_fetched_at: float = 0.0

    @property
    def dimension(self) -> int:
        """Return the embedding dimension from settings."""
        return settings.EMBEDDING_DIMENSION

    @property
    def _endpoint(self) -> str:
        """Build the regional predict endpoint from settings."""
        return (
            f"https://{settings.VERTEX_LOCATION}-aiplatform.googleapis.com/v1/"
            f"projects/{settings.VERTEX_PROJECT_ID}/locations/{settings.VERTEX_LOCATION}/"
            f"publishers/google/models/{self._model}:predict"
        )

    def _get_access_token(self) -> str:
        """Return a cached OAuth access token, refreshing it if stale.

        Returns:
            A bearer token string, from the already-authenticated
            ``gcloud`` CLI session.

        Raises:
            RuntimeError: If ``gcloud auth print-access-token`` fails (e.g.
                not logged in).
        """
        import subprocess
        import time

        age = time.monotonic() - self._token_fetched_at
        if self._cached_token is None or age > (3600 - self._TOKEN_REFRESH_MARGIN_SECONDS):
            result = subprocess.run(
                ["gcloud", "auth", "print-access-token"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"gcloud auth print-access-token failed: {result.stderr}"
                )
            self._cached_token = result.stdout.strip()
            self._token_fetched_at = time.monotonic()
        return self._cached_token

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings with ``task_type="RETRIEVAL_DOCUMENT"``."""
        return self._embed(texts, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query with ``task_type="RETRIEVAL_QUERY"``."""
        return self._embed([text], task_type="RETRIEVAL_QUERY")[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document chunks with ``task_type="RETRIEVAL_DOCUMENT"``, batched."""
        return self._embed(texts, task_type="RETRIEVAL_DOCUMENT")

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        """Embed ``texts`` in sub-batches of at most ``_MAX_BATCH_SIZE`` AND
        an estimated ``_MAX_TOKENS_PER_BATCH`` -- confirmed live that the
        instance-count limit alone isn't enough: this API also rejects a
        request with ``INVALID_ARGUMENT`` if the combined input token count
        exceeds 20,000, which a handful of long chunks can reach well
        before 250 instances do (e.g. 44 real chunks hit 61,864 actual
        tokens; a separate 20-chunk real-content batch hit 22,276).

        Token count is estimated via ``_ESTIMATED_TOKENS_PER_WORD``, *not*
        the project's general ``settings.WORDS_PER_TOKEN`` -- confirmed
        live, twice, that this model's real tokenizer produces roughly
        **5 tokens per word** on real Hungarian legal text (22,276 actual
        tokens / 4,484 words), not the ~1.33 ``WORDS_PER_TOKEN=0.75``
        implies. That setting was calibrated for the local
        sentence-transformer model's tokenizer on an English-average
        assumption (see ``ingestion/chunker.py``'s module docstring) and
        is simply the wrong ratio for this specific API, not a matter of
        needing a better safety margin on the same number. This API has
        no local tokenizer to count exactly with (the error message
        itself points at a separate ``CountTokens`` API call, which would
        double the request count -- not worth it for an estimate this
        API already confirmed empirically close).
        """
        batches: list[list[str]] = []
        current: list[str] = []
        current_tokens = 0.0
        for text in texts:
            text_tokens = len(text.split()) * self._ESTIMATED_TOKENS_PER_WORD
            would_overflow = (
                current
                and (
                    len(current) >= self._MAX_BATCH_SIZE
                    or current_tokens + text_tokens > self._MAX_TOKENS_PER_BATCH
                )
            )
            if would_overflow:
                batches.append(current)
                current = []
                current_tokens = 0.0
            current.append(text)
            current_tokens += text_tokens
        if current:
            batches.append(current)

        embeddings: list[list[float]] = []
        for batch in batches:
            embeddings.extend(self._embed_one_batch(batch, task_type))
        return embeddings

    @retry_on_transient_error(max_attempts=3)
    def _embed_one_batch(self, texts: list[str], task_type: str) -> list[list[float]]:
        """Embed at most ``_MAX_BATCH_SIZE`` strings in one Vertex AI predict call.

        Sleeps ``EMBEDDING_REQUEST_DELAY_SECONDS`` before every attempt,
        same as :class:`JinaEmbeddingDriver`, though confirmed live that
        this project's default quota doesn't need it the way Jina's did.

        Raises:
            TransientAPIError: On a 429/5xx response or a network-level
                failure, triggering a retry with exponential backoff (up
                to 3 attempts).
        """
        import time

        import httpx

        if settings.EMBEDDING_REQUEST_DELAY_SECONDS > 0:
            time.sleep(settings.EMBEDDING_REQUEST_DELAY_SECONDS)

        try:
            response = httpx.post(
                self._endpoint,
                headers={
                    "Authorization": f"Bearer {self._get_access_token()}",
                    "Content-Type": "application/json",
                },
                json={
                    "instances": [
                        {"content": text, "task_type": task_type} for text in texts
                    ],
                    "parameters": {
                        "outputDimensionality": self.dimension,
                        "autoTruncate": True,
                    },
                },
                timeout=60.0,
            )
        except httpx.TransportError as exc:
            raise TransientAPIError(f"Vertex AI embeddings network error: {exc}") from exc

        if response.status_code == 429 or response.status_code >= 500:
            raise TransientAPIError(
                f"Vertex AI embeddings request status {response.status_code}: {response.text}"
            )
        response.raise_for_status()

        predictions = response.json()["predictions"]
        return [p["embeddings"]["values"] for p in predictions]


@lru_cache(maxsize=1)
def get_embedding_driver() -> EmbeddingDriver:
    """Factory function: return the active embedding driver from settings.

    Reads ``settings.EMBEDDING_DRIVER`` and instantiates the matching driver.
    Cached with ``@lru_cache(maxsize=1)`` so repeated calls reuse the same
    driver instance and its loaded in-memory model instead of reloading from disk.

    Returns:
        An :class:`EmbeddingDriver` instance ready to call.

    Raises:
        ValueError: If ``EMBEDDING_DRIVER`` is set to an unknown value.
    """
    driver_name = settings.EMBEDDING_DRIVER.lower()

    if driver_name == "local":
        return LocalSentenceTransformerDriver()
    if driver_name == "openai":
        return OpenAIEmbeddingDriver()
    if driver_name == "openrouter":
        return OpenRouterEmbeddingDriver()
    if driver_name == "gemini":
        return GeminiEmbeddingDriver()
    if driver_name == "jina":
        return JinaEmbeddingDriver()
    if driver_name == "vertex":
        return VertexEmbeddingDriver()

    raise ValueError(
        f"Unknown EMBEDDING_DRIVER: '{driver_name}'. "
        "Valid options are: 'local', 'openai', 'openrouter', 'gemini', 'jina', 'vertex'."
    )
