"""Tests for metadata.classification_runner.ClassificationRunner (fakes only)."""

from metadata.classification_runner import ClassificationRunner
from metadata.classifier import Classification, TypeClassifier, TypeProposal
from models import (
    ChunkMetadata,
    Document,
    DocumentType,
    RetrievedChunk,
    TypeStatus,
)

APPROVED = DocumentType(
    "court_decision", "Court decision", "A ruling.", TypeStatus.APPROVED
)
PROPOSED = DocumentType("invoice", "Invoice", "A bill.", TypeStatus.PROPOSED)
RETIRED = DocumentType("memo", "Memo", "A note.", TypeStatus.RETIRED)

HEAD = "A BÍRÓSÁG ÍTÉLETET HOZOTT a felperes keresetében."


class FakeDocuments:
    def __init__(self, types, documents):
        self.types = {t.type: t for t in types}
        self.documents = list(documents)
        self.assigned: dict[str, str] = {}

    def list_types(self, status=None):
        return list(self.types.values())

    def get_type(self, name):
        return self.types.get(name)

    def upsert_type(self, doc_type):
        self.types[doc_type.type] = doc_type

    def unclassified_documents(self, limit=None, seed=None):
        return self.documents[:limit]

    def set_document_type(self, content_hash, type_name):
        self.assigned[content_hash] = type_name
        return True


class FakeChunks:
    def __init__(self, body=HEAD):
        self.body = body

    def get_document_chunks(self, content_hash):
        if self.body is None:
            return []
        meta = ChunkMetadata(source_file="a.docx", page_number=1, chunk_index=0)
        return [RetrievedChunk(id=1, content=self.body, metadata=meta, score=1.0)]


class ScriptedClassifier(TypeClassifier):
    def __init__(self, result):
        self.result, self.seen = result, []

    def classify(self, text, types):
        self.seen.append((text, [t.type for t in types]))
        return self.result


def _doc(n=1, summary="Summary."):
    return Document(f"hash{n}", f"f{n}.docx", summary)


def _run(result, types=(APPROVED,), chunks=None, docs=None):
    repo = FakeDocuments(types, docs or [_doc()])
    classifier = ScriptedClassifier(result)
    report = ClassificationRunner(repo, chunks or FakeChunks(), classifier).run()
    return report, repo, classifier


def test_a_verified_choice_of_an_approved_type_is_recorded():
    report, repo, _ = _run(
        Classification(type="court_decision", evidence="ÍTÉLETET HOZOTT")
    )

    assert repo.assigned == {"hash1": "court_decision"}
    assert (report.documents, report.classified, report.failed) == (1, 1, 0)


def test_a_quote_that_is_not_in_the_text_leaves_the_document_unclassified_and_says_why():
    report, repo, _ = _run(
        Classification(type="court_decision", evidence="THIS IS NOT THERE")
    )

    assert repo.assigned == {}
    assert report.failed == 1 and report.reasons == (
        ("quote not found in the text", 1),
    )


def test_a_missing_quote_is_not_enough():
    report, repo, _ = _run(Classification(type="court_decision", evidence=None))

    assert repo.assigned == {} and report.failed == 1


def test_an_unusable_answer_is_counted_and_retried_next_run_not_recorded():
    report, repo, _ = _run(Classification(failed=True))

    assert repo.assigned == {} and report.reasons == (("unusable answer", 1),)


def test_an_unknown_type_name_is_a_failure():
    report, repo, _ = _run(Classification(type="spaceship", evidence="ÍTÉLETET"))

    assert repo.assigned == {} and report.reasons == (("unknown or retired type", 1),)


def test_retired_types_are_not_offered_and_cannot_be_chosen():
    report, repo, classifier = _run(
        Classification(type="memo", evidence="ÍTÉLETET"), types=(APPROVED, RETIRED)
    )

    assert classifier.seen[0][1] == ["court_decision"]  # memo was not shown
    assert repo.assigned == {} and report.failed == 1


def test_a_new_proposal_is_stored_as_proposed_and_the_document_points_at_it():
    proposal = TypeProposal("contract", "Contract", "An agreement.")

    report, repo, _ = _run(Classification(proposal=proposal, evidence="ÍTÉLETET"))

    assert repo.types["contract"] == DocumentType(
        "contract", "Contract", "An agreement.", TypeStatus.PROPOSED
    )
    assert repo.assigned == {"hash1": "contract"}
    assert (report.proposed, report.new_types, report.classified) == (1, 1, 0)


def test_a_proposal_for_a_type_that_already_exists_reuses_it_and_creates_nothing():
    proposal = TypeProposal("invoice", "Invoice again", "Different words.")

    report, repo, _ = _run(
        Classification(proposal=proposal, evidence="ÍTÉLETET"),
        types=(APPROVED, PROPOSED),
    )

    assert repo.types["invoice"] == PROPOSED  # untouched
    assert repo.assigned == {"hash1": "invoice"}
    assert (report.proposed, report.new_types) == (1, 0)


def test_a_proposal_may_not_revive_a_retired_type_or_use_a_non_english_name():
    revive = _run(
        Classification(proposal=TypeProposal("memo", "M", "d"), evidence="ÍTÉLETET"),
        types=(APPROVED, RETIRED),
    )[0]
    foreign = _run(
        Classification(proposal=TypeProposal("Számla", "S", "d"), evidence="ÍTÉLETET")
    )[0]

    assert revive.failed == 1 and foreign.reasons == (
        ("proposed type is not English snake_case", 1),
    )


def test_choosing_an_already_proposed_type_counts_as_proposed_not_classified():
    report, repo, _ = _run(
        Classification(type="invoice", evidence="ÍTÉLETET"), types=(APPROVED, PROPOSED)
    )

    assert repo.assigned == {"hash1": "invoice"}
    assert (report.classified, report.proposed, report.new_types) == (0, 1, 0)


def test_a_document_without_text_is_reported_not_guessed():
    report, repo, classifier = _run(
        Classification(type="court_decision", evidence="x"),
        chunks=FakeChunks(body=None),
    )

    assert classifier.seen == [] and repo.assigned == {}
    assert report.reasons == (("no text to read", 1),)


def test_the_classifier_sees_the_summary_and_only_the_opening_of_the_document():
    long_body = "WORD " * 5000
    _, _, classifier = _run(
        Classification(type="court_decision", evidence="WORD"),
        chunks=FakeChunks(body=long_body),
        docs=[_doc(summary="The summary.")],
    )

    text = classifier.seen[0][0]
    assert text.startswith("The summary.\n\n") and len(text) < 2200


def test_several_documents_are_counted_separately_and_progress_is_reported():
    docs = [_doc(1), _doc(2), _doc(3)]
    repo = FakeDocuments([APPROVED], docs)
    answers = iter(
        [
            Classification(type="court_decision", evidence="ÍTÉLETET"),
            Classification(failed=True),
            Classification(type="court_decision", evidence="NOT THERE"),
        ]
    )

    class Sequence(TypeClassifier):
        def classify(self, text, types):
            return next(answers)

    ticks = []
    report = ClassificationRunner(
        repo, FakeChunks(), Sequence(), on_progress=lambda d, t: ticks.append((d, t))
    ).run()

    assert (report.documents, report.classified, report.failed) == (3, 1, 2)
    assert dict(report.reasons) == {
        "unusable answer": 1,
        "quote not found in the text": 1,
    }
    assert ticks == [(1, 3), (2, 3), (3, 3)]
