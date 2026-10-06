"""Putting the retrieval together from the settings and the drivers.

The one place that reads ``Settings`` and the driver factories to build a
:class:`query.service.RetrievalService`: the embedding and reranker drivers, the
listwise reranker (bound to its language model and limits, made lazily because the
model client is only needed when that step is switched on) and the observers that write
the progress lines and the audit log. Steps and the service themselves never see
``Settings``.

Key exports:
    build_retrieval_service -- Builds the service.
"""

from config import Settings
from drivers.embedding import get_embedding_driver
from drivers.llm import get_answer_driver
from drivers.reranker import get_reranker_driver
from logger import get_logger
from models import RetrievedChunk
from query.facts import QueryFactsReader
from query.listwise_rerank import listwise_rerank
from query.observers import AuditLogObserver, ProgressLogObserver
from query.profiles import PipelineFactory, ProfileResolver
from query.service import RetrievalService


def build_retrieval_service(settings: Settings) -> RetrievalService:
    """Build a :class:`RetrievalService` from ``settings``.

    Args:
        settings: Where the numbers and the choice of drivers come from.
    """
    embedding = get_embedding_driver()

    def listwise(question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return listwise_rerank(
            question,
            chunks,
            get_answer_driver(),
            max_candidates=settings.LISTWISE_RERANK_MAX_CANDIDATES,
        )

    return RetrievalService(
        QueryFactsReader(),
        ProfileResolver(settings),
        PipelineFactory(embedding, get_reranker_driver(), listwise),
        embedding,
        observers=(
            ProgressLogObserver(),
            AuditLogObserver(get_logger(), settings.RERANKER_MODEL),
        ),
    )
