"""Tests for drivers.llm: AnswerDriver ABC (template method) and concrete drivers.

No real network calls happen here — chat clients are mocked or replaced by
a fake driver subclass.
"""

from unittest.mock import MagicMock

import pytest

import drivers.llm as llm_module
from drivers.llm import (
    AnswerDriver,
    OpenAIAnswerDriver,
    OpenRouterAnswerDriver,
    _build_prompt,
    get_answer_driver,
)
from models import ChunkMetadata, RetrievedChunk


def _chunk(
    content: str, source_file: str | None = None, page_number: int | None = None
):
    return RetrievedChunk(
        id=1,
        content=content,
        metadata=ChunkMetadata(
            source_file=source_file or "", page_number=page_number, chunk_index=0
        ),
        score=0.0,
    )


def test_build_prompt_includes_numbered_sources_and_question():
    chunks = [
        _chunk("first chunk", source_file="a.pdf", page_number=1),
        _chunk("second chunk", source_file="b.pdf", page_number=2),
    ]
    system_prompt, user_message = _build_prompt("What happened?", chunks)

    assert "based strictly on" in system_prompt
    assert "[1] Source: a.pdf, page 1" in user_message
    assert "[2] Source: b.pdf, page 2" in user_message
    assert "first chunk" in user_message
    assert "second chunk" in user_message
    assert "Question: What happened?" in user_message


def test_build_prompt_forbids_outside_knowledge_and_requires_partial_answer_honesty():
    system_prompt, _ = _build_prompt("q", [])

    assert "training knowledge" in system_prompt
    assert "traceable to a specific excerpt" in system_prompt
    assert "partially answer" in system_prompt


def test_build_prompt_handles_missing_metadata_gracefully():
    chunks = [_chunk("x")]
    _, user_message = _build_prompt("q", chunks)
    assert "Source: unknown, page ?" in user_message


class _FakeAnswerDriver(AnswerDriver):
    """Minimal concrete subclass to exercise the ABC's template-method answer()."""

    def __init__(self, model, client):
        super().__init__(model)
        self._fake_client = client

    def _get_client(self):
        return self._fake_client


def _client_returning(text: str | None) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content=text))
    ]
    return client


def test_answer_calls_chat_completions_with_built_prompt_and_returns_content():
    client = _client_returning("the answer")
    driver = _FakeAnswerDriver(model="some-model", client=client)

    result = driver.answer("What is X?", [_chunk("X is Y")])

    assert result == "the answer"
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "some-model"
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_tokens"] == 1024
    assert kwargs["messages"][0]["role"] == "system"
    assert kwargs["messages"][1]["role"] == "user"
    assert "What is X?" in kwargs["messages"][1]["content"]


def test_answer_returns_empty_string_when_content_is_none():
    client = _client_returning(None)
    driver = _FakeAnswerDriver(model="some-model", client=client)

    assert driver.answer("q", []) == ""


def test_model_property_exposes_configured_model():
    driver = _FakeAnswerDriver(model="some-model", client=MagicMock())
    assert driver.model == "some-model"


def _fake_message(content=None, tool_calls=None):
    return MagicMock(content=content, tool_calls=tool_calls)


def _fake_response(message) -> MagicMock:
    return MagicMock(choices=[MagicMock(message=message)])


def _fake_function_tool_call(call_id: str, name: str, arguments: str) -> MagicMock:
    # MagicMock(name=...) is a footgun: `name` sets the mock's own repr, not
    # an attribute — build .function separately and assign .name after.
    function = MagicMock(arguments=arguments)
    function.name = name
    return MagicMock(id=call_id, function=function)


def test_run_tool_calling_turn_returns_direct_reply_with_no_tool_calls():
    client = MagicMock()
    client.chat.completions.create.return_value = _fake_response(
        _fake_message(content="hi", tool_calls=None)
    )
    driver = _FakeAnswerDriver(model="some-model", client=client)

    result = driver.run_tool_calling_turn([{"role": "user", "content": "hey"}])

    assert result.content == "hi"
    assert result.tool_calls == []
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "some-model"
    assert "tools" not in kwargs


def test_run_tool_calling_turn_passes_tools_and_returns_tool_call_requests():
    tool_call = _fake_function_tool_call("call_1", "query_knowledge_base", '{"q": 1}')
    client = MagicMock()
    client.chat.completions.create.return_value = _fake_response(
        _fake_message(content=None, tool_calls=[tool_call])
    )
    driver = _FakeAnswerDriver(model="some-model", client=client)

    result = driver.run_tool_calling_turn(
        [{"role": "user", "content": "hey"}], tools=[{"type": "function"}]
    )

    assert result.content is None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "call_1"
    assert result.tool_calls[0].name == "query_knowledge_base"
    assert result.tool_calls[0].arguments == '{"q": 1}'
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["tools"] == [{"type": "function"}]


def test_openai_driver_get_client_caches_across_calls(monkeypatch):
    created = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

    driver = OpenAIAnswerDriver(model="gpt-x")
    client1 = driver._get_client()
    client2 = driver._get_client()

    assert client1 is client2
    assert len(created) == 1


def test_openrouter_driver_uses_base_url_and_referer_header(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

    driver = OpenRouterAnswerDriver(model="openrouter/free")
    driver._get_client()

    assert captured["base_url"] == "https://openrouter.ai/api/v1"
    assert "HTTP-Referer" in captured["default_headers"]


def test_get_answer_driver_returns_openrouter_by_default(
    monkeypatch, settings_override
):
    monkeypatch.setattr(
        llm_module, "settings", settings_override(LLM_DRIVER="openrouter")
    )
    assert isinstance(get_answer_driver(), OpenRouterAnswerDriver)


def test_get_answer_driver_returns_openai(monkeypatch, settings_override):
    monkeypatch.setattr(llm_module, "settings", settings_override(LLM_DRIVER="openai"))
    assert isinstance(get_answer_driver(), OpenAIAnswerDriver)


def test_get_answer_driver_raises_on_unknown(monkeypatch, settings_override):
    monkeypatch.setattr(llm_module, "settings", settings_override(LLM_DRIVER="bogus"))
    with pytest.raises(ValueError, match="Unknown LLM_DRIVER"):
        get_answer_driver()


def test_get_answer_driver_is_cached():
    driver1 = get_answer_driver()
    driver2 = get_answer_driver()
    assert driver1 is driver2
