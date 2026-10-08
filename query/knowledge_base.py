"""The question-answering entry point of the agent: ``query_knowledge_base``.

A thin function over :class:`query.query_service.QueryService`: it builds the service from
the settings and asks it. It decides nothing and carries nothing out itself; the service and
the classes behind it do (see ``docs/query-pipeline-design.md``).

Key exports:
    query_knowledge_base  -- Answers a question from the ingested documents.
    search_knowledge_base -- The best passages only, with no answer and no decision (MCP).
"""

from config import settings
from models import RetrievedChunk
from query.composition import build_query_service, build_retrieval_service
from query.decision import scope_of_source_file
from query.outcome import Answerable
from query.service import RetrievalRequest
from store import VectorStore


def query_knowledge_base(
    question: str,
    top_k: int | None = None,
    min_score: float | None = None,
    source_file: str | None = None,
    store: VectorStore | None = None,
) -> str:
    """Answer a question using the RAG knowledge base.

    The service decides how the question is answered (exactly from the metadata, by reading
    the best passages of the documents it names or its filters select, or not at all) and
    returns the answer text. If nothing relevant is found the caller gets an honest
    "I don't know" instead of a hallucinated answer.

    Args:
        question: The user's natural-language question.
        top_k: Override for ``settings.RETRIEVAL_TOP_K``. Maximum chunks passed to the LLM.
        min_score: Override for ``settings.RETRIEVAL_MIN_SCORE``. Similarity threshold (0-1)
            below which the relevance gate fails.
        source_file: Restrict the answer to this ingested file. The question is then read
            inside that file, not decided.
        store: Optional :class:`store.VectorStore` (default: the configured one).

    Returns:
        The answer text.

    Raises:
        RuntimeError: If the embedding driver's dimension does not match the stored
            vectors, or the planner is on and no document type is approved.
    """
    return (
        build_query_service(settings)
        .answer(
            question,
            top_k=top_k,
            min_score=min_score,
            scope=scope_of_source_file(source_file) if source_file else None,
            store=store,
        )
        .text
    )


def search_knowledge_base(
    question: str, source_file: str | None = None
) -> list[RetrievedChunk]:
    """The best passages for a question, without an answer.

    For a caller that writes the answer itself (the MCP host's own model). Nothing is
    decided here: no planner is asked, so the passages come from all documents (or from
    ``source_file``), read with the configured profile.

    Args:
        question: The user's natural-language question.
        source_file: Restrict the search to this ingested file.

    Returns:
        The selected chunks, best first; empty when nothing cleared the relevance gate.
    """
    result = build_retrieval_service(settings).retrieve(
        RetrievalRequest(
            question,
            profile=settings.RETRIEVAL_STRATEGY,
            scope=scope_of_source_file(source_file) if source_file else None,
        ),
        VectorStore(),
    )
    return list(result.outcome.chunks) if isinstance(result.outcome, Answerable) else []
