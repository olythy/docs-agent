"""Embedding driver abstractions.

Defines the common interface (``EmbeddingDriver``) that all embedding backends
must implement, following the Strategy / Driver pattern described in AGENTS.md.

The active driver is selected at runtime via ``settings.EMBEDDING_DRIVER``:
    - ``"local"``  → :class:`LocalSentenceTransformerDriver` (free, offline)
    - ``"openai"`` → :class:`OpenAIEmbeddingDriver` (paid API)

Usage::

    from drivers.embedding import get_embedding_driver
    driver = get_embedding_driver()
    vector = driver.embed_text("Mennyi az SZJA tartozásom?")
"""

from abc import ABC, abstractmethod

from config import settings


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

        Args:
            texts: A list of input strings.

        Returns:
            A list of float vectors, one per input string.
        """
        from openai import OpenAI

        client = OpenAI(api_key=settings.EMBEDDING_API_KEY)
        response = client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in response.data]


def get_embedding_driver() -> EmbeddingDriver:
    """Factory function: return the active embedding driver from settings.

    Reads ``settings.EMBEDDING_DRIVER`` and instantiates the matching driver.

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

    raise ValueError(
        f"Unknown EMBEDDING_DRIVER: '{driver_name}'. "
        "Valid options are: 'local', 'openai'."
    )
