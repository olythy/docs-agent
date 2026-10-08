"""Which steps a retrieval runs, and building them.

A **profile** is data: an ordered list of step kinds with a name and a note of how it
was measured. Only the profiles registered in :data:`PROFILES` are valid (every
combination of steps would otherwise be an unmeasured pipeline). Settings choose a
profile by *name* and supply the numbers (``top_k``, the similarity threshold, the pool
size); the :class:`ProfileResolver` is the one place that turns settings into those
numbers. The :class:`PipelineFactory` builds a pipeline from a resolved profile
**per query**, because the store is restricted per query (to the documents a structured
filter selected) and so cannot be fixed in a step at start-up.

Key exports:
    StepSpec, ProfileSpec, PROFILES -- The profiles.
    RetrievalParams, ResolvedProfile, ProfileResolver -- Settings made into numbers.
    PipelineFactory -- Builds the pipeline of a profile for one store.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from config import Settings
from drivers.embedding import EmbeddingDriver
from drivers.reranker import RerankerDriver
from query.candidate_steps import (
    CslsReorderStep,
    DenseSearchStep,
    EmbedQueryStep,
    KeywordSearchStep,
    YearDenseWideningStep,
    YearKeywordWideningStep,
)
from query.gate_steps import RelevanceGateStep, RerankScoreGateStep
from query.ranking_steps import (
    ListwiseRanker,
    ListwiseRerankStep,
    RerankStep,
    RrfFusionStep,
)
from query.runner import PipelineError, RetrievalPipeline
from query.selection_steps import CosineCutStep, TopKWithGuaranteesStep
from query.step import RetrievalStep, StepName
from store import VectorStore


class Condition(StrEnum):
    """When an optional step belongs to a profile (decided from the settings)."""

    PERIOD_FILTER = "period_filter"  # RETRIEVAL_PERIOD_FILTER
    CROSS_ENCODER = "cross_encoder"  # the reranker's scores are calibrated logits
    LISTWISE = "listwise"  # LISTWISE_RERANK_ENABLED


@dataclass(frozen=True)
class StepSpec:
    """One step of a profile.

    Attributes:
        kind: Which step.
        when: The condition under which it is part of the profile (always, if ``None``).
    """

    kind: StepName
    when: Condition | None = None


@dataclass(frozen=True)
class ProfileSpec:
    """A named, ordered list of steps.

    Attributes:
        name: What ``RETRIEVAL_STRATEGY`` selects.
        steps: The steps, in order.
        measured: Where this profile was measured (a ``docs/decisions.md`` entry, a
            snapshot); a profile nobody measured does not belong here.
    """

    name: str
    steps: tuple[StepSpec, ...]
    measured: str


#: The valid profiles. Adding one means adding it here and measuring it.
PROFILES: dict[str, ProfileSpec] = {
    "hybrid": ProfileSpec(
        name="hybrid",
        steps=(
            StepSpec(StepName.EMBED_QUERY),
            StepSpec(StepName.DENSE_SEARCH),
            StepSpec(StepName.RELEVANCE_GATE),
            StepSpec(StepName.YEAR_DENSE_WIDENING, Condition.PERIOD_FILTER),
            StepSpec(StepName.CSLS_REORDER),
            StepSpec(StepName.KEYWORD_SEARCH),
            StepSpec(StepName.YEAR_KEYWORD_WIDENING, Condition.PERIOD_FILTER),
            StepSpec(StepName.RRF_FUSION),
            StepSpec(StepName.RERANK),
            StepSpec(StepName.RERANK_SCORE_GATE, Condition.CROSS_ENCODER),
            StepSpec(StepName.LISTWISE_RERANK, Condition.LISTWISE),
            StepSpec(StepName.TOP_K_SELECTION),
        ),
        measured=(
            "the default since hybrid search was adopted (docs/decisions.md); pinned "
            "by the characterization scenarios and retrieval-snapshot"
        ),
    ),
    "vector": ProfileSpec(
        name="vector",
        steps=tuple(
            StepSpec(kind)
            for kind in (
                StepName.EMBED_QUERY,
                StepName.DENSE_SEARCH,
                StepName.RELEVANCE_GATE,
                StepName.COSINE_CUT,
            )
        ),
        measured="the pre-hybrid baseline; pinned by the characterization scenarios",
    ),
}


@dataclass(frozen=True)
class RetrievalParams:
    """The numbers a retrieval runs with.

    Attributes:
        top_k: How many chunks the context may hold (also the depth of the relevance gate).
        min_score: The cosine similarity threshold of the relevance gate.
        pool_size: How many candidates each search fetches.
        diversify: Share the guaranteed (in-period) slots across documents.
        rerank_min_score: The lowest reranker score the score gate accepts.
        period_filter: Whether the question's years narrow the retrieval.
    """

    top_k: int
    min_score: float
    pool_size: int
    diversify: bool = True
    rerank_min_score: float = 0.0
    period_filter: bool = False


@dataclass(frozen=True)
class ResolvedProfile:
    """A profile with its numbers settled and its optional steps decided.

    Attributes:
        spec: The profile.
        params: The numbers.
        steps: The kinds of the steps that apply, in order.
    """

    spec: ProfileSpec
    params: RetrievalParams
    steps: tuple[StepName, ...]


class ProfileResolver:
    """Turns a profile name and the settings into a :class:`ResolvedProfile`.

    Args:
        settings: Where the numbers come from.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def resolve(
        self,
        name: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> ResolvedProfile:
        """Resolve ``name``.

        Args:
            name: A key of :data:`PROFILES`.
            top_k: Overrides ``RETRIEVAL_TOP_K``.
            min_score: Overrides ``RETRIEVAL_MIN_SCORE``.

        Raises:
            ValueError: If there is no such profile.
        """
        spec = PROFILES.get(name.lower())
        if spec is None:
            raise ValueError(
                f"Unknown retrieval profile: '{name}'. Valid options are: "
                f"{', '.join(repr(n) for n in PROFILES)}."
            )
        s = self._settings
        k = top_k if top_k is not None else s.RETRIEVAL_TOP_K
        enabled = {
            Condition.PERIOD_FILTER: s.RETRIEVAL_PERIOD_FILTER,
            Condition.CROSS_ENCODER: s.RERANKER_DRIVER.lower() == "cross_encoder",
            Condition.LISTWISE: s.LISTWISE_RERANK_ENABLED,
        }
        return ResolvedProfile(
            spec=spec,
            params=RetrievalParams(
                top_k=k,
                min_score=min_score if min_score is not None else s.RETRIEVAL_MIN_SCORE,
                pool_size=max(k, s.RETRIEVAL_CANDIDATE_POOL_SIZE),
                diversify=s.RETRIEVAL_DIVERSIFY_GUARANTEES,
                rerank_min_score=s.RERANKER_MIN_SCORE,
                period_filter=s.RETRIEVAL_PERIOD_FILTER,
            ),
            steps=tuple(
                step.kind
                for step in spec.steps
                if step.when is None or enabled[step.when]
            ),
        )


@dataclass(frozen=True)
class StepDeps:
    """What a step may need when it is built."""

    store: VectorStore
    embedding: EmbeddingDriver
    reranker: RerankerDriver
    listwise: ListwiseRanker
    params: RetrievalParams


#: step kind -> how to build it.
_BUILDERS: dict[StepName, Callable[[StepDeps], RetrievalStep]] = {
    StepName.EMBED_QUERY: lambda d: EmbedQueryStep(d.embedding),
    StepName.DENSE_SEARCH: lambda d: DenseSearchStep(d.store, d.params.pool_size),
    StepName.RELEVANCE_GATE: lambda d: RelevanceGateStep(
        d.params.top_k, d.params.min_score
    ),
    StepName.COSINE_CUT: lambda d: CosineCutStep(d.params.min_score, d.params.top_k),
    StepName.YEAR_DENSE_WIDENING: lambda d: YearDenseWideningStep(
        d.store, d.params.pool_size
    ),
    StepName.CSLS_REORDER: lambda d: CslsReorderStep(),
    StepName.YEAR_KEYWORD_WIDENING: lambda d: YearKeywordWideningStep(d.store),
    StepName.LISTWISE_RERANK: lambda d: ListwiseRerankStep(d.listwise),
    StepName.KEYWORD_SEARCH: lambda d: KeywordSearchStep(d.store),
    StepName.RRF_FUSION: lambda d: RrfFusionStep(),
    StepName.RERANK: lambda d: RerankStep(d.reranker),
    StepName.RERANK_SCORE_GATE: lambda d: RerankScoreGateStep(
        d.params.rerank_min_score
    ),
    StepName.TOP_K_SELECTION: lambda d: TopKWithGuaranteesStep(
        d.params.top_k, d.params.diversify, d.params.period_filter
    ),
}


class PipelineFactory:
    """Builds the pipeline of a resolved profile.

    Args:
        embedding: The embedding driver the steps use.
        reranker: The reranker driver the steps use.
        listwise: The listwise reranker (called only when a profile includes the step,
            so whatever it needs, such as a language-model client, is made lazily).
    """

    def __init__(
        self,
        embedding: EmbeddingDriver,
        reranker: RerankerDriver,
        listwise: ListwiseRanker,
    ) -> None:
        self._embedding = embedding
        self._reranker = reranker
        self._listwise = listwise

    def build(self, profile: ResolvedProfile, store: VectorStore) -> RetrievalPipeline:
        """Build the pipeline for ``store``.

        Raises:
            PipelineError: If a step kind has no builder, or the chain is invalid.
        """
        deps = StepDeps(
            store, self._embedding, self._reranker, self._listwise, profile.params
        )
        steps = []
        for kind in profile.steps:
            builder = _BUILDERS.get(kind)
            if builder is None:
                raise PipelineError(f"no builder for step {kind!r}")
            steps.append(builder(deps))
        return RetrievalPipeline(steps)
