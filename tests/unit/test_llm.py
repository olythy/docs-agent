"""Tests for drivers.llm: AnswerDriver ABC (template method) and concrete drivers.

No real network calls happen here — chat clients are mocked or replaced by
a fake driver subclass.
"""

import json
from unittest.mock import ANY, MagicMock

import pytest

import drivers.llm as llm_module
from drivers.llm import (
    GeminiAnswerDriver,
    OpenAIAnswerDriver,
    OpenRouterAnswerDriver,
    VertexAnswerDriver,
    _build_prompt,
    _OpenAICompatibleAnswerDriver,
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
    assert "Source: a.pdf, page 1" in user_message
    assert "Source: b.pdf, page 2" in user_message
    assert "first chunk" in user_message
    assert "second chunk" in user_message
    assert "Question: What happened?" in user_message


def test_build_prompt_places_best_ranked_chunk_last_before_the_question():
    """ "Lost in the middle" mitigation (see docs/decisions.md): the
    highest-ranked chunk (first in context_chunks) must end up physically
    last in the context block, right before "Question:", not first."""
    chunks = [
        _chunk("best match", source_file="best.pdf", page_number=1),
        _chunk("middling match", source_file="mid.pdf", page_number=1),
        _chunk("weakest match", source_file="weak.pdf", page_number=1),
    ]
    _, user_message = _build_prompt("What happened?", chunks)

    weakest_pos = user_message.index("weakest match")
    middling_pos = user_message.index("middling match")
    best_pos = user_message.index("best match")
    question_pos = user_message.index("Question:")

    assert weakest_pos < middling_pos < best_pos < question_pos


def test_build_prompt_forbids_outside_knowledge_and_requires_partial_answer_honesty():
    system_prompt, _ = _build_prompt("q", [])

    assert "training knowledge" in system_prompt
    assert "traceable to a specific excerpt" in system_prompt
    assert "partially answer" in system_prompt


def test_build_prompt_handles_missing_metadata_gracefully():
    chunks = [_chunk("x")]
    _, user_message = _build_prompt("q", chunks)
    assert "Source: unknown, page ?" in user_message


class _FakeAnswerDriver(_OpenAICompatibleAnswerDriver):
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


def _fake_function_call_part(call_id, name, args, thought_signature=None) -> MagicMock:
    # MagicMock(name=...) is a footgun: `name` sets the mock's own repr, not
    # an attribute -- set .name after construction instead.
    function_call = MagicMock(id=call_id, args=args)
    function_call.name = name
    return MagicMock(function_call=function_call, thought_signature=thought_signature)


def _fake_gemini_client(*, text=None, function_call_parts=None) -> MagicMock:
    content = MagicMock(parts=function_call_parts) if function_call_parts else None
    candidate = MagicMock(content=content)
    response = MagicMock(text=text, candidates=[candidate])
    client = MagicMock()
    client.models.generate_content.return_value = response
    return client


def test_gemini_driver_answer_sends_prompt_and_returns_text(
    monkeypatch, settings_override
):
    monkeypatch.setattr(
        llm_module, "settings", settings_override(LLM_REQUEST_DELAY_SECONDS=0.0)
    )
    fake_client = _fake_gemini_client(text="the answer")
    monkeypatch.setattr("google.genai.Client", lambda api_key, **kwargs: fake_client)

    driver = GeminiAnswerDriver(model="gemini-2.5-flash")
    result = driver.answer("What is X?", [_chunk("X is Y")])

    assert result == "the answer"
    call = fake_client.models.generate_content.call_args
    assert call.kwargs["model"] == "gemini-2.5-flash"
    assert call.kwargs["config"].system_instruction is not None
    assert call.kwargs["config"].max_output_tokens == 1024


def test_gemini_driver_disables_thinking_by_default(monkeypatch, settings_override):
    """Regression test for a real, live MAX_TOKENS truncation: a "thinking"
    model (gemini-2.5-flash) shares max_output_tokens between hidden
    reasoning and the visible answer -- a real case spent 981 tokens
    thinking and left only 39 for the answer, truncating it mid-sentence
    with no citation at all. LLM_THINKING_BUDGET defaults to 0 (disabled)
    so the full budget goes to the visible answer."""
    monkeypatch.setattr(
        llm_module,
        "settings",
        settings_override(LLM_REQUEST_DELAY_SECONDS=0.0, LLM_THINKING_BUDGET=0),
    )
    fake_client = _fake_gemini_client(text="the answer")
    monkeypatch.setattr("google.genai.Client", lambda api_key, **kwargs: fake_client)

    GeminiAnswerDriver(model="gemini-2.5-flash").answer(
        "What is X?", [_chunk("X is Y")]
    )

    call = fake_client.models.generate_content.call_args
    assert call.kwargs["config"].thinking_config.thinking_budget == 0


def test_gemini_driver_answer_returns_empty_string_when_text_is_none(
    monkeypatch, settings_override
):
    monkeypatch.setattr(
        llm_module, "settings", settings_override(LLM_REQUEST_DELAY_SECONDS=0.0)
    )
    fake_client = _fake_gemini_client(text=None)
    monkeypatch.setattr("google.genai.Client", lambda api_key, **kwargs: fake_client)

    driver = GeminiAnswerDriver(model="gemini-2.5-flash")
    assert driver.answer("q", []) == ""


def test_gemini_driver_run_tool_calling_turn_returns_direct_reply(
    monkeypatch, settings_override
):
    monkeypatch.setattr(
        llm_module, "settings", settings_override(LLM_REQUEST_DELAY_SECONDS=0.0)
    )
    fake_client = _fake_gemini_client(text="hi")
    monkeypatch.setattr("google.genai.Client", lambda api_key, **kwargs: fake_client)

    driver = GeminiAnswerDriver(model="gemini-2.5-flash")
    result = driver.run_tool_calling_turn([{"role": "user", "content": "hey"}])

    assert result.content == "hi"
    assert result.tool_calls == []
    assert "tools" not in fake_client.models.generate_content.call_args.kwargs[
        "config"
    ].model_dump(exclude_none=True)


def test_gemini_driver_run_tool_calling_turn_returns_tool_call_requests(
    monkeypatch, settings_override
):
    monkeypatch.setattr(
        llm_module, "settings", settings_override(LLM_REQUEST_DELAY_SECONDS=0.0)
    )
    part = _fake_function_call_part(
        "call_461665",
        "query_knowledge_base",
        {"question": "What is X?"},
        thought_signature=b"sig-bytes",
    )
    fake_client = _fake_gemini_client(text=None, function_call_parts=[part])
    monkeypatch.setattr("google.genai.Client", lambda api_key, **kwargs: fake_client)

    driver = GeminiAnswerDriver(model="gemini-2.5-flash")
    result = driver.run_tool_calling_turn(
        [{"role": "user", "content": "hey"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "query_knowledge_base",
                    "description": "d",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    assert result.content is None
    assert len(result.tool_calls) == 1
    # Confirmed live: Gemini's generate_content DOES have a real per-call id
    # (contrary to the SDK docs' single-call examples never showing one).
    assert result.tool_calls[0].id == "call_461665"
    assert result.tool_calls[0].name == "query_knowledge_base"
    assert json.loads(result.tool_calls[0].arguments) == {"question": "What is X?"}
    # thought_signature must be carried in provider_data for the next turn
    # to re-attach it -- confirmed live newer models reject its absence.
    assert result.tool_calls[0].provider_data == {"thought_signature": b"sig-bytes"}


def test_gemini_driver_run_tool_calling_turn_round_trips_tool_result(
    monkeypatch, settings_override
):
    """Feeding a tool-result message back (as agent.py's second call does)
    must translate correctly -- correlated by name, not by an id Gemini
    doesn't have."""
    monkeypatch.setattr(
        llm_module, "settings", settings_override(LLM_REQUEST_DELAY_SECONDS=0.0)
    )
    fake_client = _fake_gemini_client(text="Final answer")
    monkeypatch.setattr("google.genai.Client", lambda api_key, **kwargs: fake_client)

    driver = GeminiAnswerDriver(model="gemini-2.5-flash")
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "What is X?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "query_knowledge_base",
                    "type": "function",
                    "function": {
                        "name": "query_knowledge_base",
                        "arguments": '{"question": "What is X?"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "query_knowledge_base",
            "content": "KB ANSWER",
        },
    ]

    result = driver.run_tool_calling_turn(messages)

    assert result.content == "Final answer"
    contents = fake_client.models.generate_content.call_args.kwargs["contents"]
    assert len(contents) == 3  # user, model (tool call), user (function response)
    # role="user", not "tool" -- confirmed live that the real API rejects
    # role="tool" outright, despite the SDK's own docs example using it.
    assert contents[-1].role == "user"


def test_gemini_driver_run_tool_calling_turn_retries_on_429_then_succeeds(
    monkeypatch, settings_override
):
    from google.genai.errors import APIError

    monkeypatch.setattr(
        llm_module, "settings", settings_override(LLM_REQUEST_DELAY_SECONDS=0.0)
    )
    fake_response = MagicMock(text="ok", candidates=[MagicMock(content=None)])
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = [
        APIError(code=429, response_json={"error": {"message": "rate limited"}}),
        fake_response,
    ]
    monkeypatch.setattr("google.genai.Client", lambda api_key, **kwargs: fake_client)
    monkeypatch.setattr("time.sleep", MagicMock())

    driver = GeminiAnswerDriver(model="gemini-2.5-flash")
    result = driver.run_tool_calling_turn([{"role": "user", "content": "hey"}])

    assert result.content == "ok"
    assert fake_client.models.generate_content.call_count == 2


def test_gemini_driver_retries_network_error_then_succeeds(
    monkeypatch, settings_override
):
    """Regression test for a real, live "No route to host" mid-ingestion
    crash: httpx.TransportError is a completely different exception
    hierarchy from google.genai.errors.APIError and was previously not
    retried at all."""
    import httpx

    monkeypatch.setattr(
        llm_module, "settings", settings_override(LLM_REQUEST_DELAY_SECONDS=0.0)
    )
    fake_response = MagicMock(text="ok", candidates=[MagicMock(content=None)])
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = [
        httpx.ConnectError("No route to host"),
        fake_response,
    ]
    monkeypatch.setattr("google.genai.Client", lambda api_key, **kwargs: fake_client)
    monkeypatch.setattr("time.sleep", MagicMock())

    driver = GeminiAnswerDriver(model="gemini-2.5-flash")
    result = driver.run_tool_calling_turn([{"role": "user", "content": "hey"}])

    assert result.content == "ok"
    assert fake_client.models.generate_content.call_count == 2


def test_get_answer_driver_returns_gemini(monkeypatch, settings_override):
    monkeypatch.setattr(llm_module, "settings", settings_override(LLM_DRIVER="gemini"))
    assert isinstance(get_answer_driver(), GeminiAnswerDriver)


def test_vertex_driver_get_client_uses_vertexai_mode(monkeypatch, settings_override):
    """VertexAnswerDriver only overrides _get_client() -- confirms it builds
    the google-genai client in Vertex AI mode (project/location/credentials)
    instead of GeminiAnswerDriver's api_key mode."""
    monkeypatch.setattr(
        llm_module,
        "settings",
        settings_override(
            VERTEX_PROJECT_ID="my-project", VERTEX_LOCATION="us-central1"
        ),
    )
    fake_client = MagicMock()
    fake_constructor = MagicMock(return_value=fake_client)
    monkeypatch.setattr("google.genai.Client", fake_constructor)
    monkeypatch.setattr(
        "drivers.gcloud_auth.get_credentials", lambda: "fake-credentials"
    )

    driver = VertexAnswerDriver(model="gemini-2.5-flash")
    client = driver._get_client()

    assert client is fake_client
    fake_constructor.assert_called_once_with(
        vertexai=True,
        project="my-project",
        location="us-central1",
        credentials="fake-credentials",
        http_options=ANY,  # the request time limit; see test_api_timeouts.py
    )


def test_vertex_driver_get_client_caches_across_calls(monkeypatch, settings_override):
    monkeypatch.setattr(
        llm_module,
        "settings",
        settings_override(
            VERTEX_PROJECT_ID="my-project", VERTEX_LOCATION="us-central1"
        ),
    )
    fake_constructor = MagicMock(return_value=MagicMock())
    monkeypatch.setattr("google.genai.Client", fake_constructor)
    monkeypatch.setattr(
        "drivers.gcloud_auth.get_credentials", lambda: "fake-credentials"
    )

    driver = VertexAnswerDriver(model="gemini-2.5-flash")
    driver._get_client()
    driver._get_client()

    fake_constructor.assert_called_once()


def test_vertex_driver_answer_sends_prompt_and_returns_text(
    monkeypatch, settings_override
):
    """Reuses GeminiAnswerDriver.answer() unchanged -- only _get_client()
    differs, so this mainly confirms the inherited method still works
    through the overridden client construction."""
    monkeypatch.setattr(
        llm_module, "settings", settings_override(LLM_REQUEST_DELAY_SECONDS=0.0)
    )
    fake_client = _fake_gemini_client(text="the answer")
    monkeypatch.setattr("google.genai.Client", lambda **kwargs: fake_client)
    monkeypatch.setattr(
        "drivers.gcloud_auth.get_credentials", lambda: "fake-credentials"
    )

    driver = VertexAnswerDriver(model="gemini-2.5-flash")
    result = driver.answer("What is X?", [_chunk("X is Y")])

    assert result == "the answer"


def test_get_answer_driver_returns_vertex(monkeypatch, settings_override):
    monkeypatch.setattr(llm_module, "settings", settings_override(LLM_DRIVER="vertex"))
    assert isinstance(get_answer_driver(), VertexAnswerDriver)


def test_vertex_driver_invalidates_token_and_retries_on_401(
    monkeypatch, settings_override
):
    """Regression test for a real, live 401 mid-bulk-ingest -- see
    test_embedding.py's matching test for the full explanation. Here the
    401 arrives as google.genai.errors.APIError(code=401), not a raw
    httpx status, since VertexAnswerDriver goes through the SDK."""
    from google.genai.errors import APIError

    monkeypatch.setattr(
        llm_module,
        "settings",
        settings_override(
            LLM_REQUEST_DELAY_SECONDS=0.0,
            VERTEX_PROJECT_ID="my-project",
            VERTEX_LOCATION="us-central1",
        ),
    )
    fake_response = _fake_gemini_client(text="ok").models.generate_content.return_value
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = [
        APIError(code=401, response_json={"error": {"message": "unauthorized"}}),
        fake_response,
    ]
    monkeypatch.setattr("google.genai.Client", lambda **kwargs: fake_client)
    fake_invalidate = MagicMock()
    monkeypatch.setattr("drivers.gcloud_auth.invalidate", fake_invalidate)
    monkeypatch.setattr(
        "drivers.gcloud_auth.get_credentials", lambda: "fake-credentials"
    )
    monkeypatch.setattr("time.sleep", MagicMock())

    driver = VertexAnswerDriver(model="gemini-2.5-flash")
    result = driver.run_tool_calling_turn([{"role": "user", "content": "hey"}])

    assert result.content == "ok"
    fake_invalidate.assert_called_once()
    assert fake_client.models.generate_content.call_count == 2


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


def test_gemini_driver_tolerates_a_function_call_without_an_id(
    monkeypatch, settings_override
):
    """Regression: Vertex AI mode does not fill FunctionCall.id (it is optional
    in the SDK), and an `assert id is not None` crashed `make chat` on the first
    question that triggered a tool call."""
    monkeypatch.setattr(
        llm_module, "settings", settings_override(LLM_REQUEST_DELAY_SECONDS=0.0)
    )
    part = _fake_function_call_part(
        None, "query_knowledge_base", {"question": "What is X?"}
    )
    fake_client = _fake_gemini_client(text=None, function_call_parts=[part])
    monkeypatch.setattr("google.genai.Client", lambda api_key, **kwargs: fake_client)

    result = GeminiAnswerDriver(model="gemini-2.5-flash").run_tool_calling_turn(
        [{"role": "user", "content": "hey"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "query_knowledge_base",
                    "description": "d",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    call = result.tool_calls[0]
    assert call.id  # made up, but present and usable for pairing the result
    assert call.provider_data == {"synthesized_id": True}


def test_a_synthesized_call_id_is_not_sent_back_to_the_api():
    from drivers.llm import _messages_to_gemini_contents

    messages = [
        {"role": "user", "content": "What is X?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_0_abcd1234",
                    "type": "function",
                    "function": {
                        "name": "query_knowledge_base",
                        "arguments": '{"question": "What is X?"}',
                    },
                    "provider_data": {"synthesized_id": True},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_0_abcd1234", "content": "KB ANSWER"},
    ]

    _, contents = _messages_to_gemini_contents(messages)

    assert contents[1].parts[0].function_call.id is None
    assert contents[2].parts[0].function_response.id is None
    assert contents[2].parts[0].function_response.name == "query_knowledge_base"


def test_a_real_call_id_is_still_sent_back_to_the_api():
    from drivers.llm import _messages_to_gemini_contents

    messages = [
        {"role": "user", "content": "What is X?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_461665",
                    "type": "function",
                    "function": {
                        "name": "query_knowledge_base",
                        "arguments": "{}",
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_461665", "content": "KB ANSWER"},
    ]

    _, contents = _messages_to_gemini_contents(messages)

    assert contents[1].parts[0].function_call.id == "call_461665"
    assert contents[2].parts[0].function_response.id == "call_461665"


def _prompt_chunks():
    from models import ChunkMetadata, RetrievedChunk

    return [
        RetrievedChunk(
            id=1,
            content="szoveg",
            metadata=ChunkMetadata(source_file="a.docx", page_number=1, chunk_index=0),
            score=0.5,
        )
    ]


def test_build_prompt_keeps_the_strict_refusal_by_default(
    monkeypatch, settings_override
):
    from drivers.llm import _build_prompt

    monkeypatch.setattr(
        llm_module, "settings", settings_override(ANSWER_PARTIAL_COVERAGE=False)
    )

    system, _ = _build_prompt("q?", _prompt_chunks())

    assert "If the answer cannot be found in the excerpts at all" in system
    assert "sample selected from a much larger collection" not in system


def test_build_prompt_partial_coverage_refuses_only_without_a_relevant_excerpt(
    monkeypatch, settings_override
):
    """Regression: with the strict wording a broad question with partial context
    was refused 3/3; the exact refusal sentence must stay (decline detection and
    the adversarial questions depend on it)."""
    from drivers.llm import _build_prompt

    monkeypatch.setattr(
        llm_module, "settings", settings_override(ANSWER_PARTIAL_COVERAGE=True)
    )

    system, _ = _build_prompt("q?", _prompt_chunks())

    assert "Refuse only when NONE of the excerpts is relevant" in system
    assert "sample selected from a much larger collection" in system
    assert "I could not find this information in the provided documents." in system
    assert "If the answer cannot be found in the excerpts at all" not in system
