"""The step that adds the opening chunk of the one document a question names."""

import pytest

from models import ChunkMetadata, RetrievedChunk
from query.facts import QueryFacts
from query.retrieval.context import RetrievalContext
from query.retrieval.profiles import PROFILES
from query.retrieval.step import Continue, StepName
from query.retrieval.steps.selection import OpeningChunkStep


def chunk(chunk_id, index, content="text"):
    return RetrievedChunk(
        id=chunk_id,
        content=content,
        metadata=ChunkMetadata(
            source_file="a.docx", page_number=None, chunk_index=index
        ),
        score=0.5,
    )


class Store:
    def __init__(self, opening=None):
        self.opening, self.calls = opening, []

    def opening_chunks(self, limit=1):
        self.calls.append(limit)
        return [] if self.opening is None else [self.opening]


def make_step(store):
    """The step over a scripted store (which is not a VectorStore)."""
    return OpeningChunkStep(store)  # type: ignore[arg-type]


def context(selected, single=True):
    return RetrievalContext(
        facts=QueryFacts("q"), single_document=single, selected=tuple(selected)
    )


def run(step, ctx):
    result = step.run(ctx)
    assert isinstance(result, Continue)
    return result.context.selected


OPENING = chunk(99, 0, "the ruling")
MIDDLE = (chunk(1, 7), chunk(2, 14))


def test_the_opening_is_added_as_the_last_ranked_chunk_when_one_document_is_named_and_it_is_missing():
    selected = run(make_step(Store(OPENING)), context(MIDDLE))

    assert selected is not None and [c.id for c in selected] == [1, 2, 99]


def test_nothing_is_added_when_the_opening_is_already_selected_and_the_store_is_not_asked():
    store = Store(OPENING)
    ctx = context([chunk(5, 0), *MIDDLE])

    assert run(make_step(store), ctx) == ctx.selected
    assert store.calls == []


@pytest.mark.parametrize("single", [False])
def test_a_question_that_does_not_name_exactly_one_document_is_untouched(single):
    store = Store(OPENING)
    ctx = context(MIDDLE, single=single)

    assert run(make_step(store), ctx) == ctx.selected and store.calls == []


def test_a_store_without_an_opening_leaves_the_selection_as_it_is():
    ctx = context(MIDDLE)

    assert run(make_step(Store(None)), ctx) == ctx.selected


def test_with_nothing_selected_there_is_nothing_to_add_to():
    store = Store(OPENING)

    assert run(make_step(store), context([])) == ()
    assert store.calls == []


def test_the_step_is_part_of_the_default_profile_after_the_final_cut():
    kinds = [spec.kind for spec in PROFILES["best_chunks"].steps]

    assert kinds[-2:] == [StepName.TOP_K_SELECTION, StepName.OPENING_CHUNK]
