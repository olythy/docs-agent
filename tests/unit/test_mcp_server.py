"""Tests for mcp_server.py's tool implementations.

No real MCP protocol/transport involved — the @mcp.tool()-decorated
functions are plain Python underneath, so they're called directly. No
real DB or embedding model either: search_knowledge_base()/add_document() are
monkeypatched, the same pattern tests/unit/test_agent.py uses for its
tool dispatch.
"""

from unittest.mock import MagicMock

import pytest
from mcp.server.mcpserver.exceptions import ToolError

import mcp_server
from models import ChunkMetadata, RetrievedChunk


def _chunk(chunk_id, content="c", source="a.pdf", page=1, score=0.9):
    return RetrievedChunk(
        id=chunk_id,
        content=content,
        metadata=ChunkMetadata(source_file=source, page_number=page, chunk_index=0),
        score=score,
    )


def test_to_search_result_maps_expected_fields():
    result = mcp_server._to_search_result(
        _chunk(1, content="hello", source="a.pdf", page=3)
    )

    assert result == {"content": "hello", "source_file": "a.pdf", "page_number": 3}


def test_to_search_result_drops_id_and_score():
    result = mcp_server._to_search_result(_chunk(1))

    assert "id" not in result
    assert "score" not in result


def test_to_search_result_defaults_missing_metadata():
    chunk = RetrievedChunk(
        id=1,
        content="x",
        metadata=ChunkMetadata(source_file="", page_number=None, chunk_index=0),
        score=0.5,
    )

    result = mcp_server._to_search_result(chunk)

    assert result == {"content": "x", "source_file": "unknown", "page_number": "?"}


def test_search_knowledge_base_returns_mapped_chunks(monkeypatch):
    fake_retrieve = MagicMock(
        return_value=[_chunk(1, content="first"), _chunk(2, content="second")]
    )
    monkeypatch.setattr(mcp_server, "_search", fake_retrieve)

    result = mcp_server.search_knowledge_base("What is X?")

    fake_retrieve.assert_called_once_with("What is X?", None)
    assert [r["content"] for r in result] == ["first", "second"]


def test_search_knowledge_base_passes_the_source_file_on(monkeypatch):
    fake_search = MagicMock(return_value=[])
    monkeypatch.setattr(mcp_server, "_search", fake_search)

    mcp_server.search_knowledge_base("What is X?", source_file="a.pdf")

    fake_search.assert_called_once_with("What is X?", "a.pdf")


def test_search_knowledge_base_returns_empty_list_when_nothing_found(monkeypatch):
    monkeypatch.setattr(mcp_server, "_search", MagicMock(return_value=[]))

    assert mcp_server.search_knowledge_base("Anything?") == []


def test_add_document_delegates_and_confirms(monkeypatch):
    fake_add_document = MagicMock()
    monkeypatch.setattr(mcp_server, "_add_document", fake_add_document)

    result = mcp_server.add_document("notes.md")

    fake_add_document.assert_called_once_with("notes.md")
    assert "notes.md" in result


def test_add_document_reraises_value_error_as_tool_error_with_message(monkeypatch):
    """Found the hard way: a plain ValueError from a tool is an *unexpected
    crash* as far as the MCP SDK is concerned, and its message is hidden
    from the client — only ToolError's message actually reaches the model.
    """
    monkeypatch.setattr(
        mcp_server,
        "_add_document",
        MagicMock(
            side_effect=ValueError("'notes.md' is already in the knowledge base.")
        ),
    )

    with pytest.raises(ToolError, match="already in the knowledge base"):
        mcp_server.add_document("notes.md")


def test_add_document_reraises_file_not_found_as_tool_error_with_message(monkeypatch):
    monkeypatch.setattr(
        mcp_server,
        "_add_document",
        MagicMock(side_effect=FileNotFoundError("no such file: x.pdf")),
    )

    with pytest.raises(ToolError, match="no such file"):
        mcp_server.add_document("x.pdf")


def test_add_directory_delegates_and_confirms(monkeypatch):
    fake_add_directory = MagicMock(
        return_value={
            "ingested": ["a.md", "b.pdf"],
            "skipped": ["c.md"],
            "failed": [],
            "total_found": 3,
        }
    )
    monkeypatch.setattr(mcp_server, "_add_directory", fake_add_directory)

    result = mcp_server.add_directory("docs/", recursive=True)

    fake_add_directory.assert_called_once_with("docs/", recursive=True)
    assert "2 file(s) ingested" in result
    assert "1 skipped" in result


def test_add_directory_reraises_as_tool_error(monkeypatch):
    monkeypatch.setattr(
        mcp_server,
        "_add_directory",
        MagicMock(side_effect=NotADirectoryError("Path is not a directory: f.txt")),
    )

    with pytest.raises(ToolError, match="not a directory"):
        mcp_server.add_directory("f.txt")


def test_my_docs_prompt_instructs_the_model_to_call_the_tool_and_includes_the_question():
    rendered = mcp_server.my_docs("What is X?")

    assert "search_knowledge_base" in rendered
    assert "Question: What is X?" in rendered


def test_my_docs_prompt_tells_the_model_not_to_use_its_own_memory():
    rendered = mcp_server.my_docs("q")

    assert "memory" in rendered.lower()


def test_my_docs_prompt_forbids_other_tools_and_requires_citations():
    rendered = mcp_server.my_docs("q")

    assert "ONLY" in rendered
    assert "no other tool" in rendered
    assert "cite" in rendered.lower()
