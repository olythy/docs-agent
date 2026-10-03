"""Tests for ingestion.summarize.generate_document_summary.

No real LLM call here -- the answer driver is a plain mock exercising the
generic run_tool_calling_turn() interface every concrete driver implements.
"""

from unittest.mock import MagicMock

from drivers.llm import AgentTurnResult
from ingestion.summarize import generate_document_summary


def test_generate_document_summary_returns_stripped_content():
    driver = MagicMock()
    driver.run_tool_calling_turn.return_value = AgentTurnResult(
        content="  Tárgy: X. Eredmény: Y.  \n"
    )

    result = generate_document_summary("some document text", driver)

    assert result == "Tárgy: X. Eredmény: Y."


def test_generate_document_summary_calls_with_no_tools():
    """A plain single-prompt completion -- no function-calling tools
    offered, since this is a one-shot summarization, not an agentic turn."""
    driver = MagicMock()
    driver.run_tool_calling_turn.return_value = AgentTurnResult(content="summary")

    generate_document_summary("some document text", driver)

    call = driver.run_tool_calling_turn.call_args
    assert "tools" not in call.kwargs
    messages = call.kwargs["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "some document text" in messages[0]["content"]


def test_generate_document_summary_truncates_to_max_chars():
    driver = MagicMock()
    driver.run_tool_calling_turn.return_value = AgentTurnResult(content="summary")

    long_text = "x" * 10000
    generate_document_summary(long_text, driver, max_chars=100)

    call = driver.run_tool_calling_turn.call_args
    sent_prompt = call.kwargs["messages"][0]["content"]
    # Only the first 100 chars of the document text should appear, not all 10000.
    assert "x" * 100 in sent_prompt
    assert "x" * 101 not in sent_prompt


def test_generate_document_summary_returns_empty_string_when_no_content():
    driver = MagicMock()
    driver.run_tool_calling_turn.return_value = AgentTurnResult(content=None)

    result = generate_document_summary("some document text", driver)

    assert result == ""
