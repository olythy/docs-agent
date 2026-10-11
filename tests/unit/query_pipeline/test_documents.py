"""Ranking the documents of a ranked chunk list."""

from models import ChunkMetadata, RetrievedChunk
from query.retrieval.documents import document_scores


def chunk(chunk_id, source):
    return RetrievedChunk(chunk_id, "text", ChunkMetadata(source, None, 0), 0.5)


def test_document_scores_prefer_several_good_chunks_to_one_lucky_one():
    ranked = [chunk(1, "a"), chunk(2, "b"), chunk(3, "b"), chunk(4, "b"), chunk(5, "a")]

    assert [source for source, _ in document_scores(ranked)] == ["b", "a"]


def test_no_chunks_no_documents():
    assert document_scores([]) == []
