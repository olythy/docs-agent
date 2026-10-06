"""The steps of the vector profile, one at a time (what the scenarios cannot reach)."""

from models import ChunkMetadata, RetrievedChunk
from query.context import RetrievalContext
from query.facts import QueryFacts
from query.gate_steps import RelevanceGateStep
from query.step import Continue, Halt


def chunk(chunk_id: int, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        content="",
        metadata=ChunkMetadata(source_file="a.docx", page_number=None, chunk_index=0),
        score=score,
    )


def context(*scores: float) -> RetrievalContext:
    return RetrievalContext(
        facts=QueryFacts("q"),
        dense_pool=tuple(chunk(i, s) for i, s in enumerate(scores, start=1)),
    )


class TestRelevanceGate:
    def test_it_passes_when_one_of_the_top_candidates_is_similar_enough(self):
        result = RelevanceGateStep(depth=2, min_score=0.5).run(context(0.1, 0.6))

        assert isinstance(result, Continue)
        assert result.notes["gate_passed"] is True

    def test_it_only_looks_at_the_top_candidates(self):
        """A similar candidate beyond the depth does not count (the pool is not
        assumed to be sorted, the gate looks at the first ``depth`` as given)."""
        result = RelevanceGateStep(depth=2, min_score=0.5).run(context(0.1, 0.2, 0.9))

        assert isinstance(result, Halt)
        assert result.notes["gate_passed"] is False

    def test_a_refusal_says_which_stage_and_why_and_keeps_the_numbers(self):
        result = RelevanceGateStep(depth=3, min_score=0.5).run(context(0.3, 0.2))

        assert isinstance(result, Halt)
        assert result.declined.reason == "not_relevant"
        assert result.declined.stage == "relevance_gate"
        assert result.notes == {
            "gate_passed": False,
            "gate_top_score": 0.3,
            "gate_min_score": 0.5,
            "gate_depth": 3,
            "gate_candidates": 2,
        }

    def test_an_empty_pool_is_a_refusal_with_no_top_score(self):
        result = RelevanceGateStep(depth=3, min_score=0.5).run(context())

        assert isinstance(result, Halt)
        assert result.notes["gate_top_score"] is None
