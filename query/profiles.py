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

from config import Settings
from drivers.embedding import EmbeddingDriver
from query.candidate_steps import DenseSearchStep, EmbedQueryStep
from query.gate_steps import RelevanceGateStep
from query.runner import PipelineError, RetrievalPipeline
from query.selection_steps import CosineCutStep
from query.step import RetrievalStep, StepName
from store import VectorStore


@dataclass(frozen=True)
class StepSpec:
    """One step of a profile.

    Attributes:
        kind: Which step.
    """

    kind: StepName


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
    """

    top_k: int
    min_score: float
    pool_size: int


@dataclass(frozen=True)
class ResolvedProfile:
    """A profile with its numbers settled."""

    spec: ProfileSpec
    params: RetrievalParams


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
        return ResolvedProfile(
            spec=spec,
            params=RetrievalParams(
                top_k=k,
                min_score=min_score if min_score is not None else s.RETRIEVAL_MIN_SCORE,
                pool_size=max(k, s.RETRIEVAL_CANDIDATE_POOL_SIZE),
            ),
        )


@dataclass(frozen=True)
class StepDeps:
    """What a step may need when it is built."""

    store: VectorStore
    embedding: EmbeddingDriver
    params: RetrievalParams


#: step kind -> how to build it.
_BUILDERS: dict[StepName, Callable[[StepDeps], RetrievalStep]] = {
    StepName.EMBED_QUERY: lambda d: EmbedQueryStep(d.embedding),
    StepName.DENSE_SEARCH: lambda d: DenseSearchStep(d.store, d.params.pool_size),
    StepName.RELEVANCE_GATE: lambda d: RelevanceGateStep(
        d.params.top_k, d.params.min_score
    ),
    StepName.COSINE_CUT: lambda d: CosineCutStep(d.params.min_score, d.params.top_k),
}


class PipelineFactory:
    """Builds the pipeline of a resolved profile.

    Args:
        embedding: The embedding driver the steps use.
    """

    def __init__(self, embedding: EmbeddingDriver) -> None:
        self._embedding = embedding

    def build(self, profile: ResolvedProfile, store: VectorStore) -> RetrievalPipeline:
        """Build the pipeline for ``store``.

        Raises:
            PipelineError: If a step kind has no builder, or the chain is invalid.
        """
        deps = StepDeps(store, self._embedding, profile.params)
        steps = []
        for spec in profile.spec.steps:
            builder = _BUILDERS.get(spec.kind)
            if builder is None:
                raise PipelineError(f"no builder for step {spec.kind!r}")
            steps.append(builder(deps))
        return RetrievalPipeline(steps)
