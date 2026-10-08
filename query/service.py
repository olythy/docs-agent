"""The entry point of the new retrieval: a question in, chunks (or a refusal) out.

:class:`RetrievalService` coordinates, and nothing else: it reads the facts, resolves
the profile, opens the store for the run, builds the pipeline and runs it. It owns the
store session (open, check the embedding dimension), so the pipeline itself stays
ignorant of the store's lifecycle.

Key exports:
    RetrievalRequest, RetrievalResult -- What goes in and comes out.
    RetrievalService                  -- Runs a retrieval.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from drivers.embedding import EmbeddingDriver
from models import RetrievalTrace
from query.context import RetrievalContext
from query.decision import Scope
from query.facts import QueryFactsReader
from query.legacy_trace import LegacyTraceProjection
from query.observers import CompositeObserver
from query.outcome import Answerable, Declined
from query.profiles import PipelineFactory, ProfileResolver
from query.runner import StageRecord, StepObserver, TraceRecorder
from store import VectorStore


@dataclass(frozen=True)
class RetrievalRequest:
    """A question to retrieve chunks for.

    Attributes:
        question: The user's question.
        profile: The name of the profile to run.
        metadata_filter: A key/value restriction on chunk metadata.
        query_vector: A precomputed embedding of the question.
        top_k: Overrides ``RETRIEVAL_TOP_K``.
        min_score: Overrides ``RETRIEVAL_MIN_SCORE``.
        scope: Which documents the retrieval may look at (decided beforehand, from the
            metadata). The service restricts the store to it, so no caller has to remember
            to; when it has a selection it **replaces** any restriction the given store has.
            ``None`` means unrestricted.
    """

    question: str
    profile: str
    metadata_filter: Mapping[str, object] | None = None
    query_vector: Sequence[float] | None = None
    top_k: int | None = None
    min_score: float | None = None
    scope: Scope | None = None


@dataclass(frozen=True)
class RetrievalResult:
    """How a retrieval ended.

    Attributes:
        outcome: The chunks, or the refusal (which says which stage refused).
        records: What every step that ran held going in and coming out.
        trace: The same as the original pipeline's trace keys (for the diagnostics).
        scope: The scope the retrieval ran under, with its note (what was left out, what
            matched only approximately), so the answer can carry it.
    """

    outcome: Answerable | Declined
    records: tuple[StageRecord, ...]
    trace: RetrievalTrace
    scope: Scope = field(default_factory=Scope)


class RetrievalService:
    """Runs a retrieval from a request.

    Args:
        facts_reader: Reads the identifiers and years of the question.
        resolver: Turns a profile name and settings into numbers.
        factory: Builds the pipeline.
        embedding: The embedding driver (its dimension is checked against the store).
        observers: Told after every step (progress and audit logging); the trace is
            always recorded.
    """

    def __init__(
        self,
        facts_reader: QueryFactsReader,
        resolver: ProfileResolver,
        factory: PipelineFactory,
        embedding: EmbeddingDriver,
        observers: Sequence[StepObserver] = (),
    ) -> None:
        self._facts_reader = facts_reader
        self._resolver = resolver
        self._factory = factory
        self._embedding = embedding
        self._observers = tuple(observers)
        self._projection = LegacyTraceProjection()

    def retrieve(
        self, request: RetrievalRequest, store: VectorStore
    ) -> RetrievalResult:
        """Run the retrieval of ``request`` over ``store``.

        If the request has a scope with a selection, the retrieval runs over the store
        restricted to it (replacing any restriction ``store`` had), and the scope comes back
        in the result with its note.

        Raises:
            RuntimeError: If the embedding driver's dimension does not match the store.
        """
        facts = self._facts_reader.read(request.question)
        scope = request.scope if request.scope is not None else Scope()
        if scope.selection is not None:
            store = store.restricted_to(scope.selection)
        profile = self._resolver.resolve(
            request.profile, top_k=request.top_k, min_score=request.min_score
        )
        context = RetrievalContext(
            facts=facts,
            metadata_filter=request.metadata_filter,
            spread_documents=scope.names_several_documents,
            query_vector=(
                tuple(request.query_vector)
                if request.query_vector is not None
                else None
            ),
        )
        recorder = TraceRecorder()
        with store:
            store.assert_dimension_matches(self._embedding.dimension)
            pipeline = self._factory.build(profile, store)
            run = pipeline.run(context, CompositeObserver([recorder, *self._observers]))
        return RetrievalResult(
            outcome=run.outcome,
            records=tuple(recorder.records),
            trace=self._projection.project(recorder.records, run),
            scope=scope,
        )
