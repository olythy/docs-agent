"""The service wires the pieces: it runs the profile and tells the observers."""

from test_retrieval_characterization import (  # type: ignore[import-not-found]
    FakeEmbedding,
    FakeStore,
)

from logger import EventLogger
from query.facts import QueryFactsReader
from query.observers import AuditLogObserver
from query.outcome import Answerable, Declined, DeclineReason
from query.profiles import PipelineFactory, ProfileResolver
from query.service import RetrievalRequest, RetrievalService


def service(settings_override, observers=()):
    config = settings_override(
        RETRIEVAL_TOP_K=4,
        RETRIEVAL_CANDIDATE_POOL_SIZE=6,
        RETRIEVAL_MIN_SCORE=0.25,
    )
    embedding = FakeEmbedding()
    return RetrievalService(
        QueryFactsReader(),
        ProfileResolver(config),
        PipelineFactory(embedding),  # type: ignore[arg-type]
        embedding,  # type: ignore[arg-type]
        observers,
    )


class TestRetrieve:
    def test_it_returns_the_chunks_the_profile_selects(self, settings_override):
        result = service(settings_override).retrieve(
            RetrievalRequest("what about costs?", profile="vector"),
            FakeStore(),  # type: ignore[arg-type]
        )

        assert isinstance(result.outcome, Answerable)
        assert [c.id for c in result.outcome.chunks] == [1, 3, 5, 7]

    def test_a_refusal_comes_back_as_a_value_naming_the_stage(self, settings_override):
        result = service(settings_override).retrieve(
            RetrievalRequest("what about costs?", profile="vector", min_score=0.95),
            FakeStore(),  # type: ignore[arg-type]
        )

        assert isinstance(result.outcome, Declined)
        assert result.outcome.reason is DeclineReason.NOT_RELEVANT
        assert result.outcome.stage == "relevance_gate"

    def test_an_unknown_profile_is_refused_loudly(self, settings_override):
        try:
            service(settings_override).retrieve(
                RetrievalRequest("q", profile="nope"),
                FakeStore(),  # type: ignore[arg-type]
            )
        except ValueError as error:
            assert "Unknown retrieval profile" in str(error)
        else:
            raise AssertionError("an unknown profile must raise")

    def test_the_observers_are_told_about_the_run(self, settings_override, tmp_path):
        path = tmp_path / "log.jsonl"

        service(settings_override, [AuditLogObserver(EventLogger(path))]).retrieve(
            RetrievalRequest("what about costs?", profile="vector"),
            FakeStore(),  # type: ignore[arg-type]
        )

        assert "relevance_gate_checked" in path.read_text()
