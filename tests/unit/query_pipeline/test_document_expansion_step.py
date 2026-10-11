"""The step that reads the whole of the document that clearly leads the ranking."""

from models import ChunkMetadata, RetrievedChunk
from query.facts import QueryFacts
from query.retrieval.context import RetrievalContext
from query.retrieval.documents import document_scores
from query.retrieval.profiles import ProfileResolver
from query.retrieval.step import Continue
from query.retrieval.steps.selection import DocumentExpansionStep


def chunk(chunk_id, source, index):
    return RetrievedChunk(
        id=chunk_id,
        content="text",
        metadata=ChunkMetadata(source_file=source, page_number=None, chunk_index=index),
        score=0.5,
    )


class Store:
    """Holds the chunks of each document, in reading order."""

    def __init__(self, documents):
        self.documents, self.asked = documents, []

    def chunks_of_source(self, source_file, limit=50):
        self.asked.append((source_file, limit))
        return self.documents.get(source_file, [])[:limit]


A = [chunk(10 + i, "a.docx", i) for i in range(6)]
B = [chunk(20 + i, "b.docx", i) for i in range(6)]
STORE = lambda: Store({"a.docx": A, "b.docx": B})


def run(ranked, margin=1.5, max_chunks=30, store=None):
    store = store or STORE()
    step = DocumentExpansionStep(store, margin, max_chunks)  # type: ignore[arg-type]
    context = RetrievalContext(
        facts=QueryFacts("q"), ranked=tuple(ranked), selected=tuple(ranked[:2])
    )
    result = step.run(context)
    assert isinstance(result, Continue)
    return result.context, store


def test_a_clear_leader_is_read_whole_best_ranked_chunks_first():
    ranked = [A[3], A[1], A[4], B[0]]  # a.docx leads: three of the four, near the top

    context, store = run(ranked)

    ids = [c.id for c in context.selected or ()]
    assert ids[:3] == [A[3].id, A[1].id, A[4].id]  # ranked ones first
    assert sorted(ids) == sorted(c.id for c in A)  # the whole document, no other
    assert store.asked == [("a.docx", 30)]


def test_when_no_document_leads_by_the_margin_the_cut_is_kept():
    ranked = [A[0], B[0], A[1], B[1]]  # two documents, close together

    context, store = run(ranked, margin=1.5)

    assert [c.id for c in context.selected or ()] == [A[0].id, B[0].id]
    assert store.asked == []


def test_one_document_in_the_ranking_is_always_a_leader():
    context, _ = run([A[2], A[0]], margin=5.0)

    assert len(context.selected or ()) == len(A)


def test_the_document_is_cut_at_the_chunk_limit():
    context, store = run([A[0], A[1]], max_chunks=4)

    assert len(context.selected or ()) == 4 and store.asked == [("a.docx", 4)]


def test_an_empty_ranking_or_an_unknown_document_changes_nothing():
    empty, _ = run([])
    unknown, _ = run([A[0]], store=Store({}))

    assert empty.selected == () and [c.id for c in unknown.selected or ()] == [A[0].id]


def test_document_scores_prefer_several_good_chunks_to_one_lucky_one():
    scores = document_scores([A[0], B[0], B[1], B[2], A[1]])

    assert [source for source, _ in scores] == ["b.docx", "a.docx"]


def test_the_step_is_in_the_profile_only_when_the_margin_is_set(settings_override):
    on = ProfileResolver(settings_override(DOCUMENT_EXPANSION_MARGIN=1.5)).resolve(
        "best_chunks"
    )
    off = ProfileResolver(settings_override(DOCUMENT_EXPANSION_MARGIN=0)).resolve(
        "best_chunks"
    )

    assert "document_expansion" in [str(s) for s in on.steps]
    assert "document_expansion" not in [str(s) for s in off.steps]
    assert on.params.expansion_margin == 1.5
