"""Putting the retrieval together from the settings and the drivers.

The one place that reads ``Settings`` and the driver factories to build a
:class:`query.service.RetrievalService`: the embedding and reranker drivers, the
listwise reranker (bound to its language model and limits, made lazily because the
model client is only needed when that step is switched on) and the observers that write
the progress lines and the audit log. Steps and the service themselves never see
``Settings``.

Key exports:
    build_retrieval_service -- Builds the retrieval service.
    build_planning          -- Builds the planner-driven decider and its exact answerer.
    build_query_service     -- Builds the end-to-end service (decide, retrieve, answer).
"""

from config import Settings
from drivers.embedding import get_embedding_driver
from drivers.llm import AnswerDriver, get_answer_driver
from drivers.reranker import get_reranker_driver
from logger import get_logger
from models import RetrievedChunk
from query.answering import AnswerPolicy, ExactAnswerer, GroundedAnswerer, ResultPhraser
from query.decision import (
    PlanningDecider,
    ProfileSelector,
    ScopeResolver,
)
from query.facts import QueryFactsReader
from query.listwise_rerank import listwise_rerank
from query.observers import AnswerAuditObserver, AuditLogObserver, ProgressLogObserver
from query.outcome import RefusalRenderer
from query.profiles import DEFAULT_PROFILE, PipelineFactory, ProfileResolver
from query.query_service import QueryService
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
        PipelineFactory(
            embedding, get_reranker_driver(settings.RERANKER_DRIVER), listwise
        ),
        embedding,
        observers=(
            ProgressLogObserver(),
            AuditLogObserver(get_logger(), settings.RERANKER_MODEL),
        ),
    )


def build_planning(
    settings: Settings, llm: AnswerDriver
) -> tuple[PlanningDecider, ExactAnswerer]:
    """Build the planner-driven decider and the answerer of its exact decisions.

    The two share one plan executor and one database connection holder.

    Args:
        settings: Where the choice of profile comes from.
        llm: The language model that plans and words exact results.
    """
    from document_store import DocumentStore
    from metadata.clock import SystemClock
    from metadata.compiler import PlanCompiler
    from metadata.date_ranges import DateRangeResolver
    from metadata.executor import PlanExecutor
    from metadata.identifier_resolver import IdentifierResolver
    from metadata.planner import LLMQueryPlanner

    clock = SystemClock()
    compiler = PlanCompiler(DateRangeResolver(clock))
    store = DocumentStore()
    executor = PlanExecutor(store, compiler)
    decider = PlanningDecider(
        LLMQueryPlanner(llm, compiler, clock),
        store,
        ScopeResolver(executor, IdentifierResolver(store)),
        ProfileSelector(DEFAULT_PROFILE),
    )
    return decider, ExactAnswerer(executor, ResultPhraser(llm))


def build_query_service(settings: Settings) -> QueryService:
    """Build a :class:`QueryService` from ``settings``.

    A planner decides how every question is answered (exactly from the metadata, by
    reading documents, or not at all), so the metadata catalog must be loaded
    (``load-catalog``); without an approved document type the first question fails loudly.

    Args:
        settings: Where the numbers, the choice of drivers and the answer policy come from.
    """
    llm = get_answer_driver()
    decider, exact_answerer = build_planning(settings, llm)
    return QueryService(
        QueryFactsReader(),
        decider,
        build_retrieval_service(settings),
        exact_answerer,
        GroundedAnswerer(
            llm,
            AnswerPolicy(expose_document_date=settings.EXPOSE_DOCUMENT_DATE),
            observers=(
                AnswerAuditObserver(
                    get_logger(), settings.LLM_DRIVER, settings.LLM_MODEL
                ),
            ),
        ),
        RefusalRenderer(),
        default_profile=DEFAULT_PROFILE,
    )
