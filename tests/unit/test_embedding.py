"""Tests for drivers.embedding: EmbeddingDriver ABC and concrete drivers.

No real model download or network call happens here — the local driver's
SentenceTransformer is monkeypatched, and the OpenAI driver's
max_sequence_length() never touches the network in the first place.
"""

from unittest.mock import MagicMock

import pytest

import drivers.embedding as embedding_module
from drivers.embedding import (
    EmbeddingDriver,
    GeminiEmbeddingDriver,
    LocalSentenceTransformerDriver,
    OpenAIEmbeddingDriver,
    OpenRouterEmbeddingDriver,
    get_embedding_driver,
)
from retry import TransientAPIError


class _FakeDriver(EmbeddingDriver):
    """Minimal concrete subclass to exercise the ABC's default behavior.

    Deliberately does NOT override embed_text — it should come for free
    from the ABC's default implementation.
    """

    @property
    def dimension(self) -> int:
        return 1

    def embed_batch(self, texts):
        return [[float(len(t))] for t in texts]


def test_default_max_sequence_length_is_none():
    assert _FakeDriver().max_sequence_length() is None


def test_default_count_tokens_is_none():
    assert _FakeDriver().count_tokens("hello") is None


def test_default_supports_token_counting_is_false():
    assert _FakeDriver().supports_token_counting() is False


def test_default_embed_text_delegates_to_embed_batch():
    """embed_text has no per-subclass override — it must come from the ABC."""
    assert _FakeDriver().embed_text("abc") == [3.0]


def test_default_embed_query_delegates_to_embed_text():
    assert _FakeDriver().embed_query("abc") == [3.0]


def test_default_embed_documents_delegates_to_embed_batch():
    assert _FakeDriver().embed_documents(["abc", "de"]) == [[3.0], [2.0]]


def test_openai_driver_max_sequence_length_is_none():
    assert OpenAIEmbeddingDriver().max_sequence_length() is None


def test_openai_driver_count_tokens_is_none():
    assert OpenAIEmbeddingDriver().count_tokens("hello") is None


def test_openai_driver_supports_token_counting_is_false():
    assert OpenAIEmbeddingDriver().supports_token_counting() is False


def test_local_driver_max_sequence_length_reads_from_model(monkeypatch):
    fake_model = MagicMock()
    fake_model.max_seq_length = 128
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", lambda name: fake_model
    )

    driver = LocalSentenceTransformerDriver()
    assert driver.max_sequence_length() == 128


def test_local_driver_count_tokens_uses_raw_tokenizer_without_truncation(monkeypatch):
    """count_tokens must call the tokenizer directly, not model.tokenize().

    Confirmed empirically against the real sentence-transformers 3.x API:
    model.tokenize() (what encode() uses internally) already truncates to
    max_seq_length, which would make overflow undetectable. The raw
    model.tokenizer(text) call has no such truncation.
    """
    fake_model = MagicMock()
    fake_model.tokenizer.return_value = {"input_ids": list(range(999))}
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", lambda name: fake_model
    )

    driver = LocalSentenceTransformerDriver()
    assert driver.count_tokens("a very long text") == 999
    fake_model.tokenizer.assert_called_once_with("a very long text")
    fake_model.tokenize.assert_not_called()


def test_suppress_token_length_warning_filter_drops_only_that_message():
    """The filter must drop transformers' known-benign overflow warning —
    and only that one, not unrelated log records that happen to pass
    through the same logger.
    """
    import logging

    from drivers.embedding import _SuppressTokenLengthWarning

    token_warning = logging.LogRecord(
        name="transformers.tokenization_utils_base",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="Token indices sequence length is longer than the specified "
        "maximum sequence length for this model (999 > 128). Running this "
        "sequence through the model will result in indexing errors",
        args=(),
        exc_info=None,
    )
    unrelated = logging.LogRecord(
        name="transformers.tokenization_utils_base",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="some unrelated warning",
        args=(),
        exc_info=None,
    )

    log_filter = _SuppressTokenLengthWarning()
    assert log_filter.filter(token_warning) is False
    assert log_filter.filter(unrelated) is True


def test_local_driver_count_tokens_registers_the_suppression_filter(monkeypatch):
    """count_tokens() must install the filter on transformers' own logger —
    otherwise the "Token indices sequence length..." warning interleaves
    with normal stdout output in scripts like inspect_chunks.py, since
    it's emitted via logging straight to stderr, not a catchable
    warnings.warn().
    """
    import logging

    from drivers.embedding import _SuppressTokenLengthWarning

    fake_model = MagicMock()
    fake_model.tokenizer.return_value = {"input_ids": [1, 2, 3]}
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", lambda name: fake_model
    )

    LocalSentenceTransformerDriver().count_tokens("hello")

    target_logger = logging.getLogger("transformers.tokenization_utils_base")
    assert any(
        isinstance(f, _SuppressTokenLengthWarning) for f in target_logger.filters
    )


def test_local_driver_supports_token_counting_is_true():
    assert LocalSentenceTransformerDriver().supports_token_counting() is True


def test_local_driver_loads_model_only_once(monkeypatch):
    load_calls = []
    fake_model = MagicMock()
    fake_model.max_seq_length = 128

    def fake_constructor(name):
        load_calls.append(name)
        return fake_model

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", fake_constructor)

    driver = LocalSentenceTransformerDriver()
    driver.max_sequence_length()
    driver.embed_batch(["hello"])

    assert len(load_calls) == 1


def test_get_embedding_driver_returns_local_by_default(monkeypatch, settings_override):
    monkeypatch.setattr(
        embedding_module, "settings", settings_override(EMBEDDING_DRIVER="local")
    )
    assert isinstance(get_embedding_driver(), LocalSentenceTransformerDriver)


def test_get_embedding_driver_returns_openai(monkeypatch, settings_override):
    monkeypatch.setattr(
        embedding_module, "settings", settings_override(EMBEDDING_DRIVER="openai")
    )
    assert isinstance(get_embedding_driver(), OpenAIEmbeddingDriver)


def test_get_embedding_driver_raises_on_unknown(monkeypatch, settings_override):
    monkeypatch.setattr(
        embedding_module, "settings", settings_override(EMBEDDING_DRIVER="bogus")
    )
    with pytest.raises(ValueError, match="Unknown EMBEDDING_DRIVER"):
        get_embedding_driver()


def test_get_embedding_driver_is_cached():
    driver1 = get_embedding_driver()
    driver2 = get_embedding_driver()
    assert driver1 is driver2


def test_local_driver_e5_adds_query_and_passage_prefixes(monkeypatch):
    fake_model = MagicMock()
    fake_item = MagicMock()
    fake_item.tolist.return_value = [0.1] * 384
    fake_model.encode.side_effect = lambda texts, **kw: [fake_item for _ in texts]
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", lambda name: fake_model
    )

    driver = LocalSentenceTransformerDriver(model_name="intfloat/multilingual-e5-small")
    driver.embed_query("What is this?")
    fake_model.encode.assert_called_with(
        ["query: What is this?"], convert_to_numpy=True
    )

    driver.embed_documents(["doc chunk 1", "doc chunk 2"])
    fake_model.encode.assert_called_with(
        ["passage: doc chunk 1", "passage: doc chunk 2"], convert_to_numpy=True
    )


def test_local_driver_non_e5_does_not_add_prefixes(monkeypatch):
    fake_model = MagicMock()
    fake_item = MagicMock()
    fake_item.tolist.return_value = [0.1] * 384
    fake_model.encode.side_effect = lambda texts, **kw: [fake_item for _ in texts]
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", lambda name: fake_model
    )

    driver = LocalSentenceTransformerDriver(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    driver.embed_query("What is this?")
    fake_model.encode.assert_called_with(["What is this?"], convert_to_numpy=True)

    driver.embed_documents(["doc chunk 1"])
    fake_model.encode.assert_called_with(["doc chunk 1"], convert_to_numpy=True)


# --- OpenRouterEmbeddingDriver ---


def _fake_openrouter_response(vectors: list[list[float]]) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"data": [{"embedding": v} for v in vectors]}
    return response


def test_openrouter_driver_dimension_reads_from_settings(
    monkeypatch, settings_override
):
    monkeypatch.setattr(
        embedding_module, "settings", settings_override(EMBEDDING_DIMENSION=384)
    )
    assert OpenRouterEmbeddingDriver().dimension == 384


def test_openrouter_driver_uses_given_model_over_settings_default():
    driver = OpenRouterEmbeddingDriver(model="google/gemini-embedding-001")
    assert driver._model == "google/gemini-embedding-001"


def test_openrouter_driver_defaults_to_settings_embedding_model(
    monkeypatch, settings_override
):
    monkeypatch.setattr(
        embedding_module,
        "settings",
        settings_override(EMBEDDING_MODEL="google/gemini-embedding-001"),
    )
    assert OpenRouterEmbeddingDriver()._model == "google/gemini-embedding-001"


def test_openrouter_driver_embed_batch_sends_correct_request(
    monkeypatch, settings_override
):
    monkeypatch.setattr(
        embedding_module,
        "settings",
        settings_override(
            EMBEDDING_DIMENSION=384,
            EMBEDDING_API_KEY="sk-or-v1-test",
        ),
    )
    fake_post = MagicMock(
        return_value=_fake_openrouter_response([[0.1, 0.2], [0.3, 0.4]])
    )
    monkeypatch.setattr("requests.post", fake_post)

    driver = OpenRouterEmbeddingDriver(model="google/gemini-embedding-001")
    result = driver.embed_batch(["first chunk", "second chunk"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    fake_post.assert_called_once_with(
        "https://openrouter.ai/api/v1/embeddings",
        headers={
            "Authorization": "Bearer sk-or-v1-test",
            "Content-Type": "application/json",
        },
        json={
            "model": "google/gemini-embedding-001",
            "input": ["first chunk", "second chunk"],
            "dimensions": 384,
        },
        timeout=90,
    )


def test_openrouter_driver_splits_batches_over_250_items(
    monkeypatch, settings_override
):
    """Regression test for a real, empirically-confirmed limit: Google's
    embedding API (via OpenRouter) rejects a batch of >250 items with an
    HTTP 400 ("supported range is from 1 (inclusive) to 251 (exclusive)").
    """
    monkeypatch.setattr(
        embedding_module, "settings", settings_override(EMBEDDING_DIMENSION=384)
    )
    call_batches: list[list[str]] = []

    def fake_post(url, headers, json, timeout):
        call_batches.append(json["input"])
        return _fake_openrouter_response([[0.0] * 384 for _ in json["input"]])

    monkeypatch.setattr("requests.post", fake_post)

    texts = [f"chunk {i}" for i in range(300)]
    result = OpenRouterEmbeddingDriver().embed_batch(texts)

    assert len(result) == 300
    assert [len(batch) for batch in call_batches] == [250, 50]
    # Order is preserved across the split.
    assert call_batches[0][0] == "chunk 0"
    assert call_batches[1][0] == "chunk 250"


def _fake_error_response(status_code: int) -> MagicMock:
    import requests

    response = MagicMock()
    response.status_code = status_code
    response.raise_for_status.side_effect = requests.HTTPError(f"HTTP {status_code}")
    return response


def test_openrouter_driver_embed_batch_raises_immediately_on_non_retryable_error(
    monkeypatch, settings_override
):
    """A 4xx other than 429 (e.g. the batch-size-limit 400) is a real
    request error, not a transient one — no point retrying it."""
    monkeypatch.setattr(
        embedding_module, "settings", settings_override(EMBEDDING_DIMENSION=384)
    )
    fake_post = MagicMock(return_value=_fake_error_response(400))
    monkeypatch.setattr("requests.post", fake_post)

    with pytest.raises(Exception, match="HTTP 400"):
        OpenRouterEmbeddingDriver().embed_batch(["text"])

    fake_post.assert_called_once()


def test_openrouter_driver_embed_batch_retries_on_429_then_raises_when_exhausted(
    monkeypatch, settings_override
):
    monkeypatch.setattr(
        embedding_module, "settings", settings_override(EMBEDDING_DIMENSION=384)
    )
    fake_post = MagicMock(return_value=_fake_error_response(429))
    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr("time.sleep", MagicMock())

    with pytest.raises(TransientAPIError, match="status 429"):
        OpenRouterEmbeddingDriver().embed_batch(["text"])

    assert fake_post.call_count == 3


def test_openrouter_driver_embed_batch_recovers_after_transient_failure(
    monkeypatch, settings_override
):
    """A 500 followed by a real network exception followed by success — the
    driver should retry through both and still return the right result."""
    import requests

    monkeypatch.setattr(
        embedding_module, "settings", settings_override(EMBEDDING_DIMENSION=384)
    )
    fake_post = MagicMock(
        side_effect=[
            _fake_error_response(500),
            requests.ConnectionError("network blip"),
            _fake_openrouter_response([[0.1, 0.2]]),
        ]
    )
    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr("time.sleep", MagicMock())

    result = OpenRouterEmbeddingDriver().embed_batch(["text"])

    assert result == [[0.1, 0.2]]
    assert fake_post.call_count == 3


def test_get_embedding_driver_returns_openrouter(monkeypatch, settings_override):
    monkeypatch.setattr(
        embedding_module, "settings", settings_override(EMBEDDING_DRIVER="openrouter")
    )
    assert isinstance(get_embedding_driver(), OpenRouterEmbeddingDriver)


# --- GeminiEmbeddingDriver ---


def _fake_gemini_client(vectors: list[list[float]]) -> MagicMock:
    embedding_items = [MagicMock(values=v) for v in vectors]
    response = MagicMock(embeddings=embedding_items)
    client = MagicMock()
    client.models.embed_content.return_value = response
    return client


def test_gemini_driver_dimension_reads_from_settings(monkeypatch, settings_override):
    monkeypatch.setattr(
        embedding_module, "settings", settings_override(EMBEDDING_DIMENSION=384)
    )
    assert GeminiEmbeddingDriver().dimension == 384


def test_gemini_driver_uses_given_model_over_settings_default():
    driver = GeminiEmbeddingDriver(model="gemini-embedding-001")
    assert driver._model == "gemini-embedding-001"


def test_gemini_driver_defaults_to_settings_embedding_model(
    monkeypatch, settings_override
):
    monkeypatch.setattr(
        embedding_module,
        "settings",
        settings_override(EMBEDDING_MODEL="gemini-embedding-001"),
    )
    assert GeminiEmbeddingDriver()._model == "gemini-embedding-001"


def test_gemini_driver_embed_batch_sends_correct_request(
    monkeypatch, settings_override
):
    monkeypatch.setattr(
        embedding_module,
        "settings",
        settings_override(
            EMBEDDING_DIMENSION=384,
            EMBEDDING_API_KEY="fake-key",
            EMBEDDING_REQUEST_DELAY_SECONDS=0.0,
        ),
    )
    fake_client = _fake_gemini_client([[0.1, 0.2], [0.3, 0.4]])
    monkeypatch.setattr("google.genai.Client", lambda api_key: fake_client)

    driver = GeminiEmbeddingDriver(model="gemini-embedding-001")
    result = driver.embed_batch(["first chunk", "second chunk"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    call = fake_client.models.embed_content.call_args
    assert call.kwargs["model"] == "gemini-embedding-001"
    assert call.kwargs["contents"] == ["first chunk", "second chunk"]
    assert call.kwargs["config"].output_dimensionality == 384


def test_gemini_driver_sleeps_before_each_request(monkeypatch, settings_override):
    monkeypatch.setattr(
        embedding_module,
        "settings",
        settings_override(EMBEDDING_DIMENSION=384, EMBEDDING_REQUEST_DELAY_SECONDS=5.0),
    )
    fake_client = _fake_gemini_client([[0.1] * 384])
    monkeypatch.setattr("google.genai.Client", lambda api_key: fake_client)
    fake_sleep = MagicMock()
    monkeypatch.setattr("time.sleep", fake_sleep)

    GeminiEmbeddingDriver().embed_batch(["text"])

    fake_sleep.assert_called_once_with(5.0)


def test_gemini_driver_does_not_sleep_when_delay_is_zero(
    monkeypatch, settings_override
):
    monkeypatch.setattr(
        embedding_module,
        "settings",
        settings_override(EMBEDDING_DIMENSION=384, EMBEDDING_REQUEST_DELAY_SECONDS=0.0),
    )
    fake_client = _fake_gemini_client([[0.1] * 384])
    monkeypatch.setattr("google.genai.Client", lambda api_key: fake_client)
    fake_sleep = MagicMock()
    monkeypatch.setattr("time.sleep", fake_sleep)

    GeminiEmbeddingDriver().embed_batch(["text"])

    fake_sleep.assert_not_called()


def test_gemini_driver_splits_batches_over_250_items(monkeypatch, settings_override):
    monkeypatch.setattr(
        embedding_module,
        "settings",
        settings_override(EMBEDDING_DIMENSION=384, EMBEDDING_REQUEST_DELAY_SECONDS=0.0),
    )
    call_batches: list[list[str]] = []

    def fake_embed_content(model, contents, config):
        call_batches.append(contents)
        return MagicMock(embeddings=[MagicMock(values=[0.0] * 384) for _ in contents])

    fake_client = MagicMock()
    fake_client.models.embed_content.side_effect = fake_embed_content
    monkeypatch.setattr("google.genai.Client", lambda api_key: fake_client)

    texts = [f"chunk {i}" for i in range(300)]
    result = GeminiEmbeddingDriver().embed_batch(texts)

    assert len(result) == 300
    assert [len(batch) for batch in call_batches] == [250, 50]


def test_gemini_driver_retries_on_429_then_succeeds(monkeypatch, settings_override):
    from google.genai.errors import APIError

    monkeypatch.setattr(
        embedding_module,
        "settings",
        settings_override(EMBEDDING_DIMENSION=384, EMBEDDING_REQUEST_DELAY_SECONDS=0.0),
    )
    fake_response = MagicMock(embeddings=[MagicMock(values=[0.1, 0.2])])
    fake_client = MagicMock()
    fake_client.models.embed_content.side_effect = [
        APIError(code=429, response_json={"error": {"message": "rate limited"}}),
        fake_response,
    ]
    monkeypatch.setattr("google.genai.Client", lambda api_key: fake_client)
    monkeypatch.setattr("time.sleep", MagicMock())

    result = GeminiEmbeddingDriver().embed_batch(["text"])

    assert result == [[0.1, 0.2]]
    assert fake_client.models.embed_content.call_count == 2


def test_gemini_driver_raises_immediately_on_non_retryable_error(
    monkeypatch, settings_override
):
    from google.genai.errors import APIError

    monkeypatch.setattr(
        embedding_module,
        "settings",
        settings_override(EMBEDDING_DIMENSION=384, EMBEDDING_REQUEST_DELAY_SECONDS=0.0),
    )
    fake_client = MagicMock()
    fake_client.models.embed_content.side_effect = APIError(
        code=400, response_json={"error": {"message": "bad request"}}
    )
    monkeypatch.setattr("google.genai.Client", lambda api_key: fake_client)

    with pytest.raises(APIError):
        GeminiEmbeddingDriver().embed_batch(["text"])

    fake_client.models.embed_content.assert_called_once()


def test_gemini_driver_raises_after_exhausting_retries_on_persistent_429(
    monkeypatch, settings_override
):
    from google.genai.errors import APIError

    monkeypatch.setattr(
        embedding_module,
        "settings",
        settings_override(EMBEDDING_DIMENSION=384, EMBEDDING_REQUEST_DELAY_SECONDS=0.0),
    )
    fake_client = MagicMock()
    fake_client.models.embed_content.side_effect = APIError(
        code=429, response_json={"error": {"message": "rate limited"}}
    )
    monkeypatch.setattr("google.genai.Client", lambda api_key: fake_client)
    monkeypatch.setattr("time.sleep", MagicMock())

    with pytest.raises(TransientAPIError, match="status 429"):
        GeminiEmbeddingDriver().embed_batch(["text"])

    assert fake_client.models.embed_content.call_count == 3


def test_get_embedding_driver_returns_gemini(monkeypatch, settings_override):
    monkeypatch.setattr(
        embedding_module, "settings", settings_override(EMBEDDING_DRIVER="gemini")
    )
    assert isinstance(get_embedding_driver(), GeminiEmbeddingDriver)
