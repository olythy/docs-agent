"""Tests for agent.py's tool dispatch and the tool-calling loop.

No real LLM call happens here — AnswerDriver.run_tool_calling_turn() is
faked directly, returning plain AgentTurnResult/ToolCallRequest values
(this project's own shape, not the openai SDK's).
"""

import json
from unittest.mock import MagicMock

import pytest

import agent
from agent import _call_tool, run_agent
from drivers.llm import AgentTurnResult, ToolCallRequest


def _tool_call(call_id, name, arguments: dict):
    return ToolCallRequest(id=call_id, name=name, arguments=json.dumps(arguments))


def test_call_tool_add_document_calls_ingest_and_returns_confirmation(monkeypatch):
    fake_add_document = MagicMock()
    monkeypatch.setattr(agent, "add_document", fake_add_document)

    result = _call_tool("add_document", {"file_path": "notes.md"})

    fake_add_document.assert_called_once_with("notes.md")
    assert "notes.md" in result


def test_call_tool_add_directory_calls_ingest_and_returns_summary(monkeypatch):
    fake_add_directory = MagicMock(
        return_value={
            "ingested": ["a.md"],
            "skipped": [],
            "failed": [],
            "total_found": 1,
        }
    )
    monkeypatch.setattr(agent, "add_directory", fake_add_directory)

    result = _call_tool("add_directory", {"dir_path": "docs/", "recursive": True})

    fake_add_directory.assert_called_once_with("docs/", recursive=True, force=False)
    assert "Directory 'docs/' processed" in result
    assert "1 file(s) ingested" in result


def test_call_tool_query_knowledge_base_returns_answer(monkeypatch):
    fake_query = MagicMock(return_value="ANSWER")
    monkeypatch.setattr(agent, "query_knowledge_base", fake_query)

    result = _call_tool("query_knowledge_base", {"question": "What is X?"})

    fake_query.assert_called_once_with("What is X?", source_file=None)
    assert result == "ANSWER"


def test_call_tool_query_knowledge_base_passes_the_source_file_on(monkeypatch):
    fake_query = MagicMock(return_value="ANSWER")
    monkeypatch.setattr(agent, "query_knowledge_base", fake_query)

    _call_tool(
        "query_knowledge_base", {"question": "What is X?", "source_file": "a.docx"}
    )

    fake_query.assert_called_once_with("What is X?", source_file="a.docx")


def test_call_tool_raises_on_unknown_tool():
    with pytest.raises(ValueError, match="Unknown tool"):
        _call_tool("delete_everything", {})


def _fake_driver(*turns: AgentTurnResult) -> MagicMock:
    driver = MagicMock()
    driver.model = "test-model"
    driver.run_tool_calling_turn.side_effect = list(turns)
    return driver


def test_run_agent_returns_direct_reply_when_no_tool_call(monkeypatch):
    fake_driver = _fake_driver(AgentTurnResult(content="Hello there"))
    monkeypatch.setattr(agent, "get_answer_driver", lambda: fake_driver)

    result = run_agent("hi")

    assert result == "Hello there"
    assert fake_driver.run_tool_calling_turn.call_count == 1


def test_run_agent_executes_tool_call_and_returns_final_reply(monkeypatch):
    tool_call = _tool_call("call_1", "query_knowledge_base", {"question": "What is X?"})
    fake_driver = _fake_driver(
        AgentTurnResult(content=None, tool_calls=[tool_call]),
        AgentTurnResult(content="Final answer"),
    )
    monkeypatch.setattr(agent, "get_answer_driver", lambda: fake_driver)

    fake_query = MagicMock(return_value="KB ANSWER")
    monkeypatch.setattr(agent, "query_knowledge_base", fake_query)

    result = run_agent("What is X?")

    assert result == "Final answer"
    fake_query.assert_called_once_with("What is X?", source_file=None)

    second_call_messages = fake_driver.run_tool_calling_turn.call_args_list[1].args[0]
    tool_messages = [m for m in second_call_messages if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["content"] == "KB ANSWER"
    assert tool_messages[0]["tool_call_id"] == "call_1"


def test_run_agent_reports_tool_execution_errors_to_the_model(monkeypatch):
    tool_call = _tool_call("call_1", "add_document", {"file_path": "missing.pdf"})
    fake_driver = _fake_driver(
        AgentTurnResult(content=None, tool_calls=[tool_call]),
        AgentTurnResult(content="Could not add it."),
    )
    monkeypatch.setattr(agent, "get_answer_driver", lambda: fake_driver)

    fake_add_document = MagicMock(side_effect=FileNotFoundError("no such file"))
    monkeypatch.setattr(agent, "add_document", fake_add_document)

    result = run_agent("Add missing.pdf")

    assert result == "Could not add it."
    second_call_messages = fake_driver.run_tool_calling_turn.call_args_list[1].args[0]
    tool_messages = [m for m in second_call_messages if m["role"] == "tool"]
    assert tool_messages[0]["content"].startswith("Error:")
