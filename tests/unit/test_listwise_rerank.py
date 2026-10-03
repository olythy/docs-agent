"""Tests for query.listwise_rerank.listwise_rerank.

No real LLM call here -- the answer driver is a plain mock exercising the
generic run_tool_calling_turn() interface every concrete driver implements.
"""

from unittest.mock import MagicMock

from drivers.llm import AgentTurnResult
from models import ChunkMetadata, RetrievedChunk
from query.listwise_rerank import _parse_choice, listwise_rerank


def _chunk(chunk_id, source, summary=None, chunk_index=0):
    return RetrievedChunk(
        id=chunk_id,
        content=f"chunk {chunk_id}",
        metadata=ChunkMetadata(
            source_file=source,
            page_number=None,
            chunk_index=chunk_index,
            document_summary=summary,
        ),
        score=0.0,
    )


class TestParseChoice:
    def test_parses_a_plain_number(self):
        assert _parse_choice("3", num_candidates=5) == 2

    def test_parses_a_number_embedded_in_prose(self):
        assert _parse_choice("A válasz: 2.", num_candidates=5) == 1

    def test_returns_none_for_out_of_range_choice(self):
        assert _parse_choice("7", num_candidates=5) is None

    def test_returns_none_when_unparseable(self):
        assert _parse_choice("nem tudom", num_candidates=5) is None


def test_listwise_rerank_promotes_chosen_document_to_the_front():
    chunks = [
        _chunk(1, "docA.pdf", summary="Tárgy: A ügy. Eredmény: elutasítva."),
        _chunk(2, "docB.pdf", summary="Tárgy: B ügy. Eredmény: megsemmisítve."),
        _chunk(3, "docC.pdf", summary="Tárgy: C ügy. Eredmény: elutasítva."),
    ]
    driver = MagicMock()
    driver.run_tool_calling_turn.return_value = AgentTurnResult(content="2")

    result = listwise_rerank("some question", chunks, driver)

    assert [c.metadata.source_file for c in result] == ["docB.pdf", "docA.pdf", "docC.pdf"]


def test_listwise_rerank_preserves_multiple_chunks_of_the_chosen_document():
    chunks = [
        _chunk(1, "docA.pdf", summary="Tárgy: A.", chunk_index=0),
        _chunk(2, "docB.pdf", summary="Tárgy: B.", chunk_index=0),
        _chunk(3, "docB.pdf", summary="Tárgy: B.", chunk_index=1),
    ]
    driver = MagicMock()
    driver.run_tool_calling_turn.return_value = AgentTurnResult(content="2")

    result = listwise_rerank("some question", chunks, driver)

    assert [c.id for c in result] == [2, 3, 1]


def test_listwise_rerank_returns_unchanged_with_fewer_than_two_documents():
    chunks = [
        _chunk(1, "docA.pdf", summary="Tárgy: A."),
        _chunk(2, "docA.pdf", summary="Tárgy: A.", chunk_index=1),
    ]
    driver = MagicMock()

    result = listwise_rerank("some question", chunks, driver)

    assert result == chunks
    driver.run_tool_calling_turn.assert_not_called()


def test_listwise_rerank_returns_unchanged_without_any_document_summary():
    """No candidate has a document_summary yet (e.g. ingested before this
    feature existed) -- nothing for the LLM to compare, so skip the call."""
    chunks = [_chunk(1, "docA.pdf"), _chunk(2, "docB.pdf")]
    driver = MagicMock()

    result = listwise_rerank("some question", chunks, driver)

    assert result == chunks
    driver.run_tool_calling_turn.assert_not_called()


def test_listwise_rerank_returns_unchanged_when_llm_reply_unparseable():
    chunks = [
        _chunk(1, "docA.pdf", summary="Tárgy: A."),
        _chunk(2, "docB.pdf", summary="Tárgy: B."),
    ]
    driver = MagicMock()
    driver.run_tool_calling_turn.return_value = AgentTurnResult(content="nem tudom")

    result = listwise_rerank("some question", chunks, driver)

    assert result == chunks


def test_listwise_rerank_respects_max_candidates():
    chunks = [
        _chunk(1, "docA.pdf", summary="Tárgy: A."),
        _chunk(2, "docB.pdf", summary="Tárgy: B."),
        _chunk(3, "docC.pdf", summary="Tárgy: C."),
    ]
    driver = MagicMock()
    driver.run_tool_calling_turn.return_value = AgentTurnResult(content="1")

    listwise_rerank("some question", chunks, driver, max_candidates=2)

    call = driver.run_tool_calling_turn.call_args
    prompt = call.kwargs["messages"][0]["content"]
    assert "Tárgy: A." in prompt
    assert "Tárgy: B." in prompt
    assert "Tárgy: C." not in prompt
