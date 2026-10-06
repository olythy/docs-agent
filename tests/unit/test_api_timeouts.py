"""Every outbound SDK client gets a request time limit, and a timeout is retried.

Confirmed live: one LLM request that never got an answer blocked an eval run for 21
minutes (idle, 0% CPU). The google-genai SDK takes the limit in *milliseconds*, the
OpenAI SDK in *seconds*; both come from ``settings.API_REQUEST_TIMEOUT_SECONDS``.
"""

from unittest.mock import MagicMock

import httpx
import pytest

import drivers.embedding as embedding_module
import drivers.llm as llm_module
from drivers.embedding import GeminiEmbeddingDriver, OpenAIEmbeddingDriver
from drivers.llm import (
    GeminiAnswerDriver,
    OpenAIAnswerDriver,
    OpenRouterAnswerDriver,
    VertexAnswerDriver,
)
from retry_policy import TransientAPIError


@pytest.fixture
def limit(monkeypatch, settings_override):
    """A 7-second limit in both driver modules, so a wrong unit would show."""
    modified = settings_override(
        API_REQUEST_TIMEOUT_SECONDS=7.0,
        VERTEX_PROJECT_ID="p",
        VERTEX_LOCATION="l",
        LLM_REQUEST_DELAY_SECONDS=0.0,
    )
    monkeypatch.setattr(llm_module, "settings", modified)
    monkeypatch.setattr(embedding_module, "settings", modified)
    return modified


def _capture(monkeypatch, target: str, built=None) -> dict:
    """Replace an SDK client constructor; record the keyword arguments it got."""
    seen: dict = {}

    def constructor(**kwargs):
        seen.update(kwargs)
        return built if built is not None else MagicMock()

    monkeypatch.setattr(target, constructor)
    return seen


def test_the_gemini_client_is_built_with_the_limit_in_milliseconds(monkeypatch, limit):
    seen = _capture(monkeypatch, "google.genai.Client")

    GeminiAnswerDriver(model="m")._get_client()

    assert seen["http_options"].timeout == 7000


def test_the_vertex_client_is_built_with_the_limit_in_milliseconds(monkeypatch, limit):
    seen = _capture(monkeypatch, "google.genai.Client")
    monkeypatch.setattr("drivers.gcloud_auth.get_credentials", lambda: "creds")

    VertexAnswerDriver(model="m")._get_client()

    assert seen["vertexai"] is True and seen["http_options"].timeout == 7000


@pytest.mark.parametrize("driver", [OpenAIAnswerDriver, OpenRouterAnswerDriver])
def test_the_openai_compatible_llm_clients_get_the_limit_in_seconds(
    monkeypatch, limit, driver
):
    seen = _capture(monkeypatch, "openai.OpenAI")

    driver(model="m")._get_client()

    assert seen["timeout"] == 7.0


def test_the_embedding_clients_get_the_limit_too(monkeypatch, limit):
    gemini = _capture(monkeypatch, "google.genai.Client")
    GeminiEmbeddingDriver()._get_client()
    assert gemini["http_options"].timeout == 7000

    fake = MagicMock()
    fake.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1, 0.2])]
    )
    openai = _capture(monkeypatch, "openai.OpenAI", built=fake)
    OpenAIEmbeddingDriver().embed_batch(["x"])
    assert openai["timeout"] == 7.0


def _timing_out_driver(monkeypatch, replies):
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = replies
    monkeypatch.setattr("google.genai.Client", lambda **kwargs: fake_client)
    monkeypatch.setattr("time.sleep", MagicMock())  # no real back-off waits
    return GeminiAnswerDriver(model="m"), fake_client


def test_a_request_that_times_out_is_retried_and_can_then_succeed(monkeypatch, limit):
    ok = MagicMock(text="ok", candidates=[MagicMock(content=None)])
    driver, client = _timing_out_driver(
        monkeypatch, [httpx.ReadTimeout("no answer"), ok]
    )

    result = driver.run_tool_calling_turn([{"role": "user", "content": "hi"}])

    assert result.content == "ok"
    assert client.models.generate_content.call_count == 2


def test_a_request_that_always_times_out_raises_after_three_tries_not_forever(
    monkeypatch, limit
):
    driver, client = _timing_out_driver(
        monkeypatch, [httpx.ReadTimeout("no answer")] * 3
    )

    with pytest.raises(TransientAPIError, match="network error"):
        driver.run_tool_calling_turn([{"role": "user", "content": "hi"}])

    assert client.models.generate_content.call_count == 3
