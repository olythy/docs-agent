"""The steps of the hybrid profile, one at a time.

The scenarios (``test_retrieval_characterization.py``) pin what the whole chain does on
a small corpus. These tests cover what a fake store cannot show (the arguments a step
passes to the store) and the rules of the two steps that were ported from the original
functions (the hubness reorder and the final cut with its year quota).
"""

from logger import EventLogger
from models import ChunkMetadata, RetrievedChunk
from query.facts import QueryFacts
from query.observers import AuditLogObserver
from query.retrieval.context import RetrievalContext
from query.retrieval.step import Continue, Halt
from query.retrieval.steps.candidates import (
    CslsReorderStep,
    KeywordSearchStep,
    YearDenseWideningStep,
    YearKeywordWideningStep,
)
from query.retrieval.steps.gates import RerankScoreGateStep
from query.retrieval.steps.selection import TopKWithGuaranteesStep


def chunk(
    chunk_id: int,
    score: float = 0.5,
    source: str = "a.docx",
    date: str | None = None,
    hub: float | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        content="",
        metadata=ChunkMetadata(
            source_file=source,
            page_number=None,
            chunk_index=0,
            document_date=date,
            hub_score=hub,
        ),
        score=score,
    )


def ids(chunks) -> list[int]:
    return [c.id for c in chunks]


class SpyStore:
    """Records how it was searched and answers with what it was given."""

    def __init__(self, fulltext=(), by_identifier=(), dense=()):
        self.fulltext, self.by_identifier, self.dense = fulltext, by_identifier, dense
        self.search_calls: list[dict] = []
        self.fulltext_calls: list[dict] = []
        self.identifier_calls: list[dict] = []

    def search_fulltext(self, query_text, top_k, **kwargs):
        self.fulltext_calls.append({"top_k": top_k, **kwargs})
        return list(self.fulltext)

    def search(self, query_embedding, top_k, min_score, **kwargs):
        self.search_calls.append({"top_k": top_k, "min_score": min_score, **kwargs})
        return list(self.dense)

    def search_by_identifier(self, tokens, top_k, per_token=False):
        self.identifier_calls.append(
            {"tokens": tokens, "top_k": top_k, "per_token": per_token}
        )
        return list(self.by_identifier)


def context(**slots) -> RetrievalContext:
    facts = slots.pop("facts", QueryFacts("q"))
    return RetrievalContext(facts=facts, **slots)


class TestKeywordSearch:
    def test_it_fetches_as_many_candidates_as_the_vector_pool_holds(self):
        store = SpyStore()

        KeywordSearchStep(store).run(  # type: ignore[arg-type]
            context(dense_pool=(chunk(1), chunk(2), chunk(3)))
        )

        assert store.fulltext_calls == [{"top_k": 3}]

    def test_the_metadata_filter_is_passed_only_when_there_is_one(self):
        store = SpyStore()
        step = KeywordSearchStep(store)  # type: ignore[arg-type]

        step.run(context(dense_pool=(chunk(1),), metadata_filter={"k": "v"}))

        assert store.fulltext_calls == [{"top_k": 1, "metadata_filter": {"k": "v"}}]


FACTS_2021 = QueryFacts("q", years=(2021,))


class TestYearDenseWidening:
    def widen(self, store, pool, facts=FACTS_2021, **extra):
        step = YearDenseWideningStep(store, pool_size=7)  # type: ignore[arg-type]
        return step.run(
            context(facts=facts, query_vector=(0.1,), dense_pool=pool, **extra)
        )

    def test_it_searches_the_years_with_the_pool_size_and_no_threshold(self):
        store = SpyStore(dense=(chunk(9, 0.4),))

        self.widen(store, (chunk(1, 0.9),))

        assert store.search_calls == [{"top_k": 7, "min_score": 0.0, "years": [2021]}]

    def test_the_metadata_filter_goes_along(self):
        store = SpyStore(dense=(chunk(9, 0.4),))

        self.widen(store, (chunk(1, 0.9),), metadata_filter={"k": "v"})

        assert store.search_calls[0]["years"] == [2021]
        assert store.search_calls[0]["metadata_filter"] == {"k": "v"}

    def test_the_union_is_sorted_by_similarity_and_a_tie_keeps_the_original_first(self):
        store = SpyStore(dense=(chunk(9, 0.5), chunk(8, 0.7), chunk(1, 0.5)))

        result = self.widen(store, (chunk(1, 0.5), chunk(2, 0.9)))

        assert isinstance(result, Continue)
        # 1 is already there (not added twice); 1 (0.5) keeps its place before 9 (0.5)
        assert ids(result.context.dense_pool or ()) == [2, 8, 1, 9]

    def test_it_reports_the_years_and_the_year_pool(self):
        store = SpyStore(dense=(chunk(9, 0.4),))

        result = self.widen(store, (chunk(1, 0.9),))

        assert isinstance(result, Continue)
        assert result.notes == {"years": [2021]}
        assert [(r.label, ids(r.chunks)) for r in result.records] == [
            ("year_pool", [9])
        ]

    def test_without_years_nothing_is_searched_or_changed(self):
        store = SpyStore(dense=(chunk(9, 0.4),))

        result = self.widen(store, (chunk(1, 0.9),), facts=QueryFacts("q"))

        assert store.search_calls == []
        assert isinstance(result, Continue)
        assert ids(result.context.dense_pool or ()) == [1]
        assert result.records == () and dict(result.notes) == {}


class TestYearKeywordWidening:
    def widen(self, store, facts=FACTS_2021, **extra):
        return YearKeywordWideningStep(store).run(  # type: ignore[arg-type]
            context(
                facts=facts,
                dense_pool=(chunk(1), chunk(2), chunk(3)),
                keyword_pool=(chunk(5, 3.0), chunk(6, 1.0)),
                **extra,
            )
        )

    def test_the_limit_is_the_size_of_the_vector_pool_and_the_years_are_passed(self):
        store = SpyStore(fulltext=(chunk(7, 9.0),))

        self.widen(store)

        assert store.fulltext_calls == [{"top_k": 3, "years": [2021]}]

    def test_the_metadata_filter_goes_along(self):
        store = SpyStore(fulltext=(chunk(7, 9.0),))

        self.widen(store, metadata_filter={"k": "v"})

        assert store.fulltext_calls == [
            {"top_k": 3, "years": [2021], "metadata_filter": {"k": "v"}}
        ]

    def test_the_year_candidates_are_appended_not_sorted_in(self):
        """Even a year candidate with a higher score goes after the existing ones."""
        store = SpyStore(fulltext=(chunk(7, 9.0), chunk(5, 3.0)))

        result = self.widen(store)

        assert isinstance(result, Continue)
        assert ids(result.context.keyword_pool or ()) == [5, 6, 7]
        assert [(r.label, ids(r.chunks)) for r in result.records] == [
            ("year_pool", [7, 5])
        ]

    def test_without_years_nothing_is_searched(self):
        store = SpyStore(fulltext=(chunk(7, 9.0),))

        result = self.widen(store, facts=QueryFacts("q"))

        assert store.fulltext_calls == []
        assert isinstance(result, Continue)
        assert ids(result.context.keyword_pool or ()) == [5, 6]


class TestRerankScoreGate:
    def test_a_low_score_is_dropped_and_the_numbers_are_kept(self):
        ranked = (chunk(1, 0.9), chunk(2, 0.1), chunk(3, 0.5))

        result = RerankScoreGateStep(0.5).run(context(ranked=ranked))

        assert isinstance(result, Continue)
        assert ids(result.context.ranked or ()) == [1, 3]  # 0.5 itself is accepted
        assert result.notes == {
            "rerank_threshold": 0.5,
            "rerank_candidates": 3,
            "rerank_accepted": 2,
            "rerank_top_score": 0.9,
        }

    def test_nothing_accepted_is_a_refusal_that_keeps_the_numbers(self):
        result = RerankScoreGateStep(0.5).run(context(ranked=(chunk(1, 0.1),)))

        assert isinstance(result, Halt)
        assert result.declined.reason == "rerank_rejected"
        assert result.declined.stage == "rerank_score_gate"
        assert result.notes["rerank_accepted"] == 0

    def test_the_audit_event_is_what_the_original_logged(self, tmp_path):
        import json

        path = tmp_path / "log.jsonl"
        gate = RerankScoreGateStep(0.5)
        before = context(ranked=(chunk(1, 0.9), chunk(2, 0.1)))

        AuditLogObserver(EventLogger(path), reranker_model="m").on_step(
            gate, before, gate.run(before), 0.0
        )

        event = json.loads(path.read_text())
        assert event["action"] == "rerank_applied"
        assert event["data"] == {
            "question": "q",
            "reranker_model": "m",
            "threshold": 0.5,
            "candidates_count": 2,
            "accepted_count": 1,
            "top_score": 0.9,
        }


class TestCslsReorder:
    def reorder(self, pool):
        result = CslsReorderStep().run(context(dense_pool=tuple(pool)))
        assert isinstance(result, Continue)
        return result.context.dense_pool or ()

    def test_a_less_generic_chunk_is_promoted_above_a_higher_raw_score(self):
        """The real near-duplicate-dilution case (docs/decisions.md): a lower raw score
        with a *much* lower hub score beats a higher one that sits in the generic centre
        of the embedding space."""
        pool = [
            chunk(1, 0.87, hub=0.98),  # generic, high raw score
            chunk(2, 0.84, hub=0.90),  # less generic
        ]

        assert ids(self.reorder(pool)) == [2, 1]

    def test_without_a_hub_score_the_raw_score_decides(self):
        """No hub score yet (compute-hub-scores has not run, or a new chunk): neither a
        crash nor a place at the bottom."""
        pool = [chunk(1, 0.5), chunk(2, 0.9)]

        assert ids(self.reorder(pool)) == [2, 1]

    def test_the_raw_scores_are_untouched(self):
        pool = [chunk(1, 0.87, hub=0.98), chunk(2, 0.84, hub=0.90), chunk(3, 0.5)]

        out = self.reorder(pool)

        assert {c.id: c.score for c in out} == {c.id: c.score for c in pool}


class TestTopKYearQuota:
    """The reranker knows nothing about dates, so off-year chunks kept the whole top_k."""

    def select(self, chunks, top_k=4, years=(2022,), diversify=False):
        result = TopKWithGuaranteesStep(top_k, diversify, year_quota=True).run(
            context(facts=QueryFacts("q", years=tuple(years)), ranked=tuple(chunks))
        )
        assert isinstance(result, Continue)
        return result.context.selected or ()

    def test_half_the_slots_are_reserved_for_the_questions_years(self):
        chunks = [
            chunk(1, 0.9, "a.docx", "2019-01-01"),
            chunk(2, 0.8, "b.docx", "2019-02-01"),
            chunk(3, 0.7, "c.docx", "2018-03-01"),
            chunk(4, 0.6, "d.docx", "2017-04-01"),
            chunk(5, 0.5, "e.docx", "2022-05-01"),
            chunk(6, 0.4, "f.docx", "2022-06-01"),
        ]

        result = self.select(chunks)

        assert sorted(
            c.id for c in result if (c.metadata.document_date or "")[:4] == "2022"
        ) == [5, 6]
        assert len(result) == 4

    def test_nothing_changes_without_an_in_period_chunk(self):
        chunks = [chunk(i, 1.0 - i / 10, "a.docx", "2019-01-01") for i in range(1, 7)]

        assert ids(self.select(chunks)) == [1, 2, 3, 4]

    def test_the_reserved_slots_take_turns_across_documents_when_diversifying(self):
        chunks = [
            chunk(1, 0.9, "x.docx", "2019-01-01"),
            chunk(2, 0.8, "x.docx", "2019-01-01"),
            chunk(3, 0.7, "a.docx", "2022-01-01"),
            chunk(4, 0.6, "a.docx", "2022-01-01"),
            chunk(5, 0.5, "b.docx", "2022-02-01"),
        ]

        result = self.select(chunks, diversify=True)

        reserved = {c.metadata.source_file for c in result if c.id in {3, 4, 5}}
        assert reserved == {"a.docx", "b.docx"}

    def test_without_diversifying_one_document_may_take_every_reserved_slot(self):
        chunks = [
            chunk(1, 0.9, "x.docx", "2019-01-01"),
            chunk(2, 0.8, "x.docx", "2019-01-01"),
            chunk(3, 0.7, "a.docx", "2022-01-01"),
            chunk(4, 0.6, "a.docx", "2022-01-01"),
            chunk(5, 0.5, "b.docx", "2022-02-01"),
        ]

        result = self.select(chunks, diversify=False)

        assert {c.metadata.source_file for c in result if c.id in {3, 4, 5}} == {
            "a.docx"
        }


def test_the_year_quota_only_applies_when_the_profile_asks_for_it():
    chunks = tuple(chunk(i, date="2020-01-01") for i in range(1, 5)) + (
        chunk(9, date="2021-01-01"),
    )
    facts = QueryFacts("q", years=(2021,))

    with_quota = TopKWithGuaranteesStep(4, False, year_quota=True).run(
        context(facts=facts, ranked=chunks)
    )
    without = TopKWithGuaranteesStep(4, False, year_quota=False).run(
        context(facts=facts, ranked=chunks)
    )

    assert isinstance(with_quota, Continue) and isinstance(without, Continue)
    assert 9 in ids(with_quota.context.selected or ())
    assert 9 not in ids(without.context.selected or ())


class TestSpreadOverNamedDocuments:
    """A question that names several documents must show each of them."""

    def chunks(self):
        # document a holds the four best chunks, b only a weak one
        return tuple(
            [chunk(i, 1.0 - i / 10, source="a.docx") for i in range(1, 5)]
            + [chunk(9, 0.2, source="b.docx")]
        )

    def select(self, spread, top_k=3):
        result = TopKWithGuaranteesStep(top_k, False, year_quota=False).run(
            context(ranked=self.chunks(), spread_documents=spread)
        )
        assert isinstance(result, Continue)
        return ids(result.context.selected or ())

    def test_without_it_the_best_document_fills_every_slot(self):
        assert self.select(spread=False) == [1, 2, 3]

    def test_with_it_the_documents_take_turns_best_first(self):
        assert self.select(spread=True) == [1, 9, 2]

    def test_one_document_is_unaffected(self):
        only_a = tuple(c for c in self.chunks() if c.metadata.source_file == "a.docx")

        result = TopKWithGuaranteesStep(3, False, year_quota=False).run(
            context(ranked=only_a, spread_documents=True)
        )

        assert isinstance(result, Continue)
        assert ids(result.context.selected or ()) == [1, 2, 3]
