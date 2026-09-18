"""Tests for agent.py's tool dispatch and the tool-calling loop.

No real LLM call happens here — the OpenAI-compatible client returned by
AnswerDriver.get_client() is faked with plain objects shaped like the
openai SDK's response (choices[0].message.content/.tool_calls).
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import agent
from agent import _call_tool, run_agent


def _fake_message(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _fake_response(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _fake_tool_call(call_id, name, arguments: dict):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def test_call_tool_add_document_calls_ingest_and_returns_confirmation(monkeypatch):
    fake_add_document = MagicMock()
    monkeypatch.setattr(agent, "add_document", fake_add_document)

    result = _call_tool("add_document", {"file_path": "notes.md"})

    fake_add_document.assert_called_once_with("notes.md")
    assert "notes.md" in result


def test_call_tool_query_knowledge_base_returns_answer(monkeypatch):
    fake_query = MagicMock(return_value="ANSWER")
    monkeypatch.setattr(agent, "query_knowledge_base", fake_query)

    result = _call_tool("query_knowledge_base", {"question": "What is X?"})

    fake_query.assert_called_once_with("What is X?")
    assert result == "ANSWER"


def test_call_tool_raises_on_unknown_tool():
    with pytest.raises(ValueError, match="Unknown tool"):
        _call_tool("delete_everything", {})


def _fake_driver(client):
    driver = MagicMock()
    driver.model = "test-model"
    driver.get_client.return_value = client
    return driver


def test_run_agent_returns_direct_reply_when_no_tool_call(monkeypatch):
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _fake_response(
        _fake_message(content="Hello there", tool_calls=None)
    )
    monkeypatch.setattr(agent, "get_answer_driver", lambda: _fake_driver(fake_client))

    result = run_agent("hi")

    assert result == "Hello there"
    assert fake_client.chat.completions.create.call_count == 1


def test_run_agent_executes_tool_call_and_returns_final_reply(monkeypatch):
    tool_call = _fake_tool_call("call_1", "query_knowledge_base", {"question": "What is X?"})
    first_response = _fake_response(_fake_message(content=None, tool_calls=[tool_call]))
    final_response = _fake_response(_fake_message(content="Final answer", tool_calls=None))

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = [first_response, final_response]
    monkeypatch.setattr(agent, "get_answer_driver", lambda: _fake_driver(fake_client))

    fake_query = MagicMock(return_value="KB ANSWER")
    monkeypatch.setattr(agent, "query_knowledge_base", fake_query)

    result = run_agent("What is X?")

    assert result == "Final answer"
    fake_query.assert_called_once_with("What is X?")

    second_call_messages = fake_client.chat.completions.create.call_args_list[1].kwargs["messages"]
    tool_messages = [m for m in second_call_messages if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["content"] == "KB ANSWER"
    assert tool_messages[0]["tool_call_id"] == "call_1"


def test_run_agent_reports_tool_execution_errors_to_the_model(monkeypatch):
    tool_call = _fake_tool_call("call_1", "add_document", {"file_path": "missing.pdf"})
    first_response = _fake_response(_fake_message(content=None, tool_calls=[tool_call]))
    final_response = _fake_response(_fake_message(content="Could not add it.", tool_calls=None))

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = [first_response, final_response]
    monkeypatch.setattr(agent, "get_answer_driver", lambda: _fake_driver(fake_client))

    fake_add_document = MagicMock(side_effect=FileNotFoundError("no such file"))
    monkeypatch.setattr(agent, "add_document", fake_add_document)

    result = run_agent("Add missing.pdf")

    assert result == "Could not add it."
    second_call_messages = fake_client.chat.completions.create.call_args_list[1].kwargs["messages"]
    tool_messages = [m for m in second_call_messages if m["role"] == "tool"]
    assert tool_messages[0]["content"].startswith("Error:")
