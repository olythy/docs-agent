"""The service wires the pieces: it runs the profile and tells the observers."""

from test_retrieval_characterization import (  # type: ignore[import-not-found]
    FakeCrossEncoder,
    FakeEmbedding,
    FakeStore,
)

from logger import EventLogger
from models import DocumentSelection
from query.decision import Scope
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
        PipelineFactory(
            embedding,  # type: ignore[arg-type]
            FakeCrossEncoder(),
            lambda q, chunks: chunks,
        ),
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


def only(*files: str) -> Scope:
    """A scope over some documents (the fake store reads the files from the selection)."""
    return Scope(selection=DocumentSelection("fake", (set(files),)), note="a note")


def files_of(result) -> set[str]:
    assert isinstance(result.outcome, Answerable)
    return {c.metadata.source_file for c in result.outcome.chunks}


class TestScope:
    def retrieve(self, settings_override, scope=None, store=None):
        return service(settings_override).retrieve(
            RetrievalRequest(
                "what about costs?", profile="vector", scope=scope, min_score=0.0
            ),
            store or FakeStore(),  # type: ignore[arg-type]
        )

    def test_without_a_scope_the_retrieval_is_unrestricted(self, settings_override):
        result = self.retrieve(settings_override)

        assert len(files_of(result)) > 1
        assert result.scope == Scope()

    def test_the_retrieval_only_sees_the_documents_of_the_scope(
        self, settings_override
    ):
        result = self.retrieve(settings_override, only("c.docx"))

        assert files_of(result) == {"c.docx"}

    def test_the_service_restricts_the_store_so_no_caller_has_to(
        self, settings_override
    ):
        """A caller passes the plain store and the scope, and gets the restricted retrieval."""
        everything = FakeStore()

        result = self.retrieve(settings_override, only("a.docx", "b.docx"), everything)

        assert files_of(result) == {"a.docx", "b.docx"}

    def test_the_scope_replaces_a_restriction_the_given_store_had(
        self, settings_override
    ):
        already = FakeStore({"a.docx"})

        result = self.retrieve(settings_override, only("c.docx"), already)

        assert files_of(result) == {"c.docx"}  # the scope decides

    def test_the_scope_and_its_note_come_back_in_the_result(self, settings_override):
        scope = only("c.docx")

        result = self.retrieve(settings_override, scope)

        assert result.scope is scope
        assert result.scope.note == "a note"

    def test_a_scope_with_only_a_note_does_not_restrict_but_is_kept(
        self, settings_override
    ):
        scope = Scope(note="the identifier matched no document")

        result = self.retrieve(settings_override, scope)

        assert len(files_of(result)) > 1  # unrestricted
        assert result.scope.note == "the identifier matched no document"

    def test_an_empty_selection_sees_nothing_not_everything(self, settings_override):
        result = self.retrieve(settings_override, only())

        assert isinstance(
            result.outcome, Declined
        )  # the relevance gate: nothing to look at
