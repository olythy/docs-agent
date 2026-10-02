"""Tests for drivers.reranker: RerankerDriver ABC and concrete drivers.

No real model download happens here — CrossEncoderRerankerDriver's
sentence_transformers.CrossEncoder is monkeypatched.
"""

from unittest.mock import MagicMock

import pytest

import drivers.reranker as reranker_module
from drivers.reranker import (
    CrossEncoderRerankerDriver,
    JinaRerankerDriver,
    NoopRerankerDriver,
    VertexRankerDriver,
    get_reranker_driver,
)
from models import ChunkMetadata, RetrievedChunk
from retry_policy import TransientAPIError


def _fake_gcloud_token(token="fake-access-token"):
    result = MagicMock()
    result.returncode = 0
    result.stdout = f"{token}\n"
    return result


def _chunk(content, page=1):
    return RetrievedChunk(
        id=1,
        content=content,
        metadata=ChunkMetadata(source_file="doc.pdf", page_number=page, chunk_index=0),
        score=0.5,
    )


def test_noop_driver_returns_chunks_unchanged():
    chunks = [_chunk("a"), _chunk("b")]
    assert NoopRerankerDriver().rerank("question", chunks) is chunks


def test_cross_encoder_driver_sorts_by_predicted_score(monkeypatch):
    fake_model = MagicMock()
    fake_model.predict.return_value = [0.1, 0.9]
    monkeypatch.setattr("sentence_transformers.CrossEncoder", lambda name: fake_model)

    chunks = [_chunk("low relevance"), _chunk("high relevance")]
    reranked = CrossEncoderRerankerDriver().rerank("question", chunks)

    assert [c.content for c in reranked] == ["high relevance", "low relevance"]
    assert reranked[0].score == 0.9
    assert reranked[1].score == 0.1


def test_cross_encoder_driver_scores_question_chunk_pairs(monkeypatch):
    fake_model = MagicMock()
    fake_model.predict.return_value = [0.5]
    monkeypatch.setattr("sentence_transformers.CrossEncoder", lambda name: fake_model)

    CrossEncoderRerankerDriver().rerank("my question", [_chunk("chunk text")])

    fake_model.predict.assert_called_once_with([("my question", "chunk text")])


def test_cross_encoder_driver_returns_empty_list_without_loading_model(monkeypatch):
    load_calls = []
    monkeypatch.setattr(
        "sentence_transformers.CrossEncoder",
        lambda name: load_calls.append(name),
    )

    assert CrossEncoderRerankerDriver().rerank("question", []) == []
    assert load_calls == []


def test_cross_encoder_driver_loads_model_only_once(monkeypatch):
    load_calls = []
    fake_model = MagicMock()
    fake_model.predict.return_value = [0.1]

    def fake_constructor(name):
        load_calls.append(name)
        return fake_model

    monkeypatch.setattr("sentence_transformers.CrossEncoder", fake_constructor)

    driver = CrossEncoderRerankerDriver()
    driver.rerank("q1", [_chunk("a")])
    driver.rerank("q2", [_chunk("b")])

    assert len(load_calls) == 1


def test_get_reranker_driver_returns_noop_by_default(monkeypatch, settings_override):
    monkeypatch.setattr(
        reranker_module, "settings", settings_override(RERANKER_DRIVER="none")
    )
    assert isinstance(get_reranker_driver(), NoopRerankerDriver)


def test_get_reranker_driver_returns_cross_encoder(monkeypatch, settings_override):
    monkeypatch.setattr(
        reranker_module, "settings", settings_override(RERANKER_DRIVER="cross_encoder")
    )
    assert isinstance(get_reranker_driver(), CrossEncoderRerankerDriver)


def test_get_reranker_driver_raises_on_unknown(monkeypatch, settings_override):
    monkeypatch.setattr(
        reranker_module, "settings", settings_override(RERANKER_DRIVER="bogus")
    )
    with pytest.raises(ValueError, match="Unknown RERANKER_DRIVER"):
        get_reranker_driver()


def _fake_jina_rerank_response(status_code: int, results: list[dict] | None = None):
    response = MagicMock()
    response.status_code = status_code
    if results is not None:
        response.json.return_value = {"results": results}
    else:
        response.text = "error"
    return response


def test_jina_driver_uses_given_model_over_settings_default():
    driver = JinaRerankerDriver(model_name="jina-reranker-v2-base-multilingual")
    assert driver._model_name == "jina-reranker-v2-base-multilingual"


def test_jina_driver_returns_empty_list_without_calling_api(monkeypatch):
    fake_post = MagicMock()
    monkeypatch.setattr("httpx.post", fake_post)

    assert JinaRerankerDriver().rerank("question", []) == []
    fake_post.assert_not_called()


def test_jina_driver_reorders_and_scores_by_relevance(monkeypatch, settings_override):
    monkeypatch.setattr(
        reranker_module,
        "settings",
        settings_override(RERANKER_API_KEY="fake-key", RERANKER_MODEL="jina-reranker-v2-base-multilingual"),
    )
    chunks = [_chunk("low relevance"), _chunk("high relevance")]
    fake_post = MagicMock(
        return_value=_fake_jina_rerank_response(
            200,
            [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.1},
            ],
        )
    )
    monkeypatch.setattr("httpx.post", fake_post)

    reranked = JinaRerankerDriver().rerank("question", chunks)

    assert [c.content for c in reranked] == ["high relevance", "low relevance"]
    assert reranked[0].score == 0.9
    assert reranked[1].score == 0.1
    call = fake_post.call_args
    assert call.kwargs["json"]["model"] == "jina-reranker-v2-base-multilingual"
    assert call.kwargs["json"]["query"] == "question"
    assert call.kwargs["json"]["documents"] == ["low relevance", "high relevance"]
    assert call.kwargs["headers"]["Authorization"] == "Bearer fake-key"


def test_jina_driver_retries_on_429_then_succeeds(monkeypatch):
    fake_post = MagicMock(
        side_effect=[
            _fake_jina_rerank_response(429),
            _fake_jina_rerank_response(200, [{"index": 0, "relevance_score": 0.5}]),
        ]
    )
    monkeypatch.setattr("httpx.post", fake_post)
    monkeypatch.setattr("time.sleep", MagicMock())

    reranked = JinaRerankerDriver().rerank("q", [_chunk("a")])

    assert reranked[0].score == 0.5
    assert fake_post.call_count == 2


def test_jina_driver_retries_network_error_then_succeeds(monkeypatch):
    import httpx

    fake_post = MagicMock(
        side_effect=[
            httpx.ConnectError("No route to host"),
            _fake_jina_rerank_response(200, [{"index": 0, "relevance_score": 0.5}]),
        ]
    )
    monkeypatch.setattr("httpx.post", fake_post)
    monkeypatch.setattr("time.sleep", MagicMock())

    reranked = JinaRerankerDriver().rerank("q", [_chunk("a")])

    assert reranked[0].score == 0.5
    assert fake_post.call_count == 2


def test_jina_driver_raises_after_exhausting_retries_on_persistent_429(monkeypatch):
    fake_post = MagicMock(return_value=_fake_jina_rerank_response(429))
    monkeypatch.setattr("httpx.post", fake_post)
    monkeypatch.setattr("time.sleep", MagicMock())

    with pytest.raises(TransientAPIError, match="status 429"):
        JinaRerankerDriver().rerank("q", [_chunk("a")])

    assert fake_post.call_count == 3


def test_get_reranker_driver_returns_jina(monkeypatch, settings_override):
    monkeypatch.setattr(
        reranker_module, "settings", settings_override(RERANKER_DRIVER="jina")
    )
    assert isinstance(get_reranker_driver(), JinaRerankerDriver)


def test_get_reranker_driver_is_cached():
    driver1 = get_reranker_driver()
    driver2 = get_reranker_driver()
    assert driver1 is driver2


def _fake_vertex_rank_response(status_code: int, records: list[dict] | None = None):
    response = MagicMock()
    response.status_code = status_code
    if records is not None:
        response.json.return_value = {"records": records}
    else:
        response.text = "error"
    return response


def test_vertex_ranker_uses_given_model_over_settings_default():
    driver = VertexRankerDriver(model_name="semantic-ranker-default@latest")
    assert driver._model_name == "semantic-ranker-default@latest"


def test_vertex_ranker_builds_endpoint_from_settings(monkeypatch, settings_override):
    monkeypatch.setattr(
        reranker_module, "settings", settings_override(VERTEX_PROJECT_ID="my-project")
    )
    driver = VertexRankerDriver()
    assert driver._endpoint == (
        "https://discoveryengine.googleapis.com/v1/projects/my-project/"
        "locations/global/rankingConfigs/default_ranking_config:rank"
    )


def test_vertex_ranker_returns_empty_list_without_calling_api(monkeypatch):
    fake_post = MagicMock()
    monkeypatch.setattr("httpx.post", fake_post)

    assert VertexRankerDriver().rerank("question", []) == []
    fake_post.assert_not_called()


def test_vertex_ranker_reorders_and_scores_by_relevance(monkeypatch, settings_override):
    monkeypatch.setattr(
        reranker_module,
        "settings",
        settings_override(
            VERTEX_PROJECT_ID="my-project",
            RERANKER_MODEL="semantic-ranker-default@latest",
        ),
    )
    monkeypatch.setattr("subprocess.run", MagicMock(return_value=_fake_gcloud_token()))
    chunks = [_chunk("low relevance"), _chunk("high relevance")]
    fake_post = MagicMock(
        return_value=_fake_vertex_rank_response(
            200,
            [
                {"id": "0", "score": 0.1},
                {"id": "1", "score": 0.9},
            ],
        )
    )
    monkeypatch.setattr("httpx.post", fake_post)

    reranked = VertexRankerDriver().rerank("question", chunks)

    assert [c.content for c in reranked] == ["high relevance", "low relevance"]
    assert reranked[0].score == 0.9
    assert reranked[1].score == 0.1
    call = fake_post.call_args
    assert call.kwargs["json"]["model"] == "semantic-ranker-default@latest"
    assert call.kwargs["json"]["query"] == "question"
    assert call.kwargs["json"]["records"] == [
        {"id": "0", "content": "low relevance"},
        {"id": "1", "content": "high relevance"},
    ]
    assert call.kwargs["headers"]["Authorization"] == "Bearer fake-access-token"


def test_vertex_ranker_retries_on_429_then_succeeds(monkeypatch):
    monkeypatch.setattr("subprocess.run", MagicMock(return_value=_fake_gcloud_token()))
    fake_post = MagicMock(
        side_effect=[
            _fake_vertex_rank_response(429),
            _fake_vertex_rank_response(200, [{"id": "0", "score": 0.5}]),
        ]
    )
    monkeypatch.setattr("httpx.post", fake_post)
    monkeypatch.setattr("time.sleep", MagicMock())

    reranked = VertexRankerDriver().rerank("q", [_chunk("a")])

    assert reranked[0].score == 0.5
    assert fake_post.call_count == 2


def test_vertex_ranker_retries_network_error_then_succeeds(monkeypatch):
    import httpx

    monkeypatch.setattr("subprocess.run", MagicMock(return_value=_fake_gcloud_token()))
    fake_post = MagicMock(
        side_effect=[
            httpx.ConnectError("No route to host"),
            _fake_vertex_rank_response(200, [{"id": "0", "score": 0.5}]),
        ]
    )
    monkeypatch.setattr("httpx.post", fake_post)
    monkeypatch.setattr("time.sleep", MagicMock())

    reranked = VertexRankerDriver().rerank("q", [_chunk("a")])

    assert reranked[0].score == 0.5
    assert fake_post.call_count == 2


def test_vertex_ranker_raises_after_exhausting_retries_on_persistent_429(monkeypatch):
    monkeypatch.setattr("subprocess.run", MagicMock(return_value=_fake_gcloud_token()))
    fake_post = MagicMock(return_value=_fake_vertex_rank_response(429))
    monkeypatch.setattr("httpx.post", fake_post)
    monkeypatch.setattr("time.sleep", MagicMock())

    with pytest.raises(TransientAPIError, match="status 429"):
        VertexRankerDriver().rerank("q", [_chunk("a")])

    assert fake_post.call_count == 3


def test_get_reranker_driver_returns_vertex(monkeypatch, settings_override):
    monkeypatch.setattr(
        reranker_module, "settings", settings_override(RERANKER_DRIVER="vertex")
    )
    assert isinstance(get_reranker_driver(), VertexRankerDriver)
