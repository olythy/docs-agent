"""Characterization tests: what the retrieval pipeline does, pinned end to end.

These do not say the behaviour is *right*; they say it is *this*. The retrieval path is the
most heavily measured part of the system (see docs/decisions.md), so a change to a step,
a profile or the order of the chain must leave these results identical; when a result is
*meant* to change, the change is made here on purpose, in the same commit, and explained.

The numbers of the scenarios without an identifier were first measured on the original
engine (``retrieve_chunks`` with the strategy classes) and found identical on the new one;
they were then carried over here unchanged under the step names (2026-10-08, the commit that
deleted the original). The scenarios that name an identifier describe the new behaviour only:
the original pulled the named document's chunks in with a text search, the new pipeline does
not (the scope restricts the store to the named documents instead).

A small deterministic corpus, a fake ``VectorStore`` that really filters by year /
metadata, and a fake reranker run the whole retrieval service. For each scenario both the
ids every step left (``<step>.<list>``: what ``funnel`` shows) and the final context are
pinned.

The corpus (id: cosine, document, year, what the text is about):

    1: .90 a 2020 costs       2: .60 a 2020 other
    3: .85 b 2021 costs       4: .55 b 2021 other
    5: .80 c 2022 costs       6: .50 c 2022 other
    7: .75 d 2022 other       8: .45 d 2022 other
    9-12: .30-.27 e 2023, every chunk carries the case number 10.P.20.100/2022/5
    13-14: .26/.25 f 2021, every chunk carries the case number 20.P.20.200/2021/3
"""

from dataclasses import replace

import pytest

from corpus.commands.retrieval_snapshot import stage_ids
from drivers.reranker import CrossEncoderRerankerDriver, RerankerDriver
from models import ChunkMetadata, RetrievedChunk
from query.facts import QueryFactsReader
from query.outcome import Answerable
from query.profiles import DEFAULT_PROFILE, PipelineFactory, ProfileResolver
from query.service import RetrievalRequest, RetrievalService

CASE = "10.P.20.100/2022/5"
CASE2 = "20.P.20.200/2021/3"

#: (id, cosine, source_file, year, text)
_CORPUS = [
    (1, 0.90, "a.docx", "2020", "the costs of the proceedings were awarded"),
    (2, 0.60, "a.docx", "2020", "the claim concerns a lease"),
    (3, 0.85, "b.docx", "2021", "costs of the proceedings and fees"),
    (4, 0.55, "b.docx", "2021", "the claim concerns a boundary"),
    (5, 0.80, "c.docx", "2022", "costs follow the outcome of the proceedings"),
    (6, 0.50, "c.docx", "2022", "the claim concerns a tenancy"),
    (7, 0.75, "d.docx", "2022", "the court heard the witnesses"),
    (8, 0.45, "d.docx", "2022", "the court rejected the objection"),
    (9, 0.30, "e.docx", "2023", f"case {CASE} the plaintiff claims damages"),
    (10, 0.29, "e.docx", "2023", f"case {CASE} the defendant disputes it"),
    (11, 0.28, "e.docx", "2023", f"case {CASE} the expert reported"),
    (12, 0.27, "e.docx", "2023", f"case {CASE} the judgment was announced"),
    (13, 0.26, "f.docx", "2021", f"case {CASE2} the tenant appealed"),
    (14, 0.25, "f.docx", "2021", f"case {CASE2} the appeal was dismissed"),
]


def _chunk(row, hubs: dict[int, float] | None = None) -> RetrievedChunk:
    chunk_id, cosine, source, year, text = row
    return RetrievedChunk(
        id=chunk_id,
        content=text,
        metadata=ChunkMetadata(
            source_file=source,
            page_number=None,
            chunk_index=chunk_id,
            document_date=f"{year}-06-01",
            document_identifiers=tuple(c for c in (CASE, CASE2) if c in text),
            hub_score=(hubs or {}).get(chunk_id),
        ),
        score=cosine,
    )


class FakeStore:
    """A VectorStore with a tiny in-memory corpus and real filtering semantics."""

    def __init__(
        self,
        documents: set[str] | None = None,
        hubs: dict[int, float] | None = None,
    ) -> None:
        self._documents = documents  # a restriction, like VectorStore(selection=...)
        self._hubs = hubs  # chunk id -> hub score (how generic a chunk is)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def restricted_to(self, selection):
        """Like the real store: a store that only sees the selected documents (here, the
        selection's first parameter is the set of source files)."""
        return FakeStore(set(selection.params[0]), self._hubs)

    def assert_dimension_matches(self, expected):
        return None

    def _visible(self, metadata_filter=None, years=None):
        rows = [
            _chunk(r, self._hubs)
            for r in _CORPUS
            if self._documents is None or r[2] in self._documents
        ]
        if metadata_filter:
            rows = [
                c
                for c in rows
                if all(
                    getattr(c.metadata, k, None) == v
                    for k, v in metadata_filter.items()
                )
            ]
        if years:
            wanted = {str(y) for y in years}
            rows = [c for c in rows if (c.metadata.document_date or "")[:4] in wanted]
        return rows

    def search(
        self, query_embedding, top_k, min_score, metadata_filter=None, years=None
    ):
        rows = [
            c for c in self._visible(metadata_filter, years) if c.score >= min_score
        ]
        return sorted(rows, key=lambda c: -c.score)[:top_k]

    def search_fulltext(self, query_text, top_k, metadata_filter=None, years=None):
        words = {w for w in query_text.lower().replace("?", "").split() if len(w) > 3}
        scored = [
            (sum(w in c.content.lower() for w in words), c)
            for c in self._visible(metadata_filter, years)
        ]
        hits = sorted(((n, c) for n, c in scored if n), key=lambda p: (-p[0], p[1].id))
        return [replace(c, score=float(n)) for n, c in hits][:top_k]

    def search_by_identifier(self, tokens, top_k, per_token=False):
        rows = [
            replace(c, score=1.0)
            for c in self._visible()
            if any(t.lower() in c.content.lower() for t in tokens)
        ]
        return sorted(rows, key=lambda c: c.id)[:top_k]


class FakeEmbedding:
    dimension = 384

    def embed_query(self, question):
        return [0.0] * 384


class FakeCrossEncoder(CrossEncoderRerankerDriver):
    """Ranks the chunks that talk about costs first (a stand-in for a real ranker)."""

    def __init__(self) -> None:
        pass

    def rerank(self, question, chunks):
        scored = [
            replace(c, score=(0.9 if "costs" in c.content else 0.2) + c.score / 100)
            for c in chunks
        ]
        return sorted(scored, key=lambda c: -c.score)


class RejectsEverything(FakeCrossEncoder):
    def rerank(self, question, chunks):
        return [replace(c, score=0.01) for c in chunks]


class FakeOtherReranker(RerankerDriver):
    """A reranker whose scores are not calibrated logits (not a cross-encoder)."""

    def rerank(self, question, chunks):
        scored = [
            replace(c, score=(0.9 if "costs" in c.content else 0.2) + c.score / 100)
            for c in chunks
        ]
        return sorted(scored, key=lambda c: -c.score)


class FakeListwise:
    def __call__(self, question, chunks, driver, **kwargs):
        return list(reversed(chunks))


_BASE_SETTINGS = {
    "RERANKER_DRIVER": "cross_encoder",
    "RETRIEVAL_TOP_K": 4,
    "RETRIEVAL_CANDIDATE_POOL_SIZE": 6,
    "RETRIEVAL_MIN_SCORE": 0.25,
    "RERANKER_MIN_SCORE": 0.5,
    "RETRIEVAL_PERIOD_FILTER": False,
    "RETRIEVAL_DIVERSIFY_GUARANTEES": True,
    "LISTWISE_RERANK_ENABLED": False,
}


def run_scenario(
    settings_override,
    question,
    *,
    store=None,
    reranker=None,
    metadata_filter=None,
    **settings,
):
    """Run the retrieval service on a scenario; the ids each step left, and the final ids."""
    config = settings_override(**{**_BASE_SETTINGS, **settings})
    embedding = FakeEmbedding()
    service = RetrievalService(
        QueryFactsReader(),
        ProfileResolver(config),
        PipelineFactory(
            embedding,  # type: ignore[arg-type]
            reranker or FakeCrossEncoder(),
            lambda q, chunks: FakeListwise()(q, chunks, None),
        ),
        embedding,  # type: ignore[arg-type]
    )
    result = service.retrieve(
        RetrievalRequest(
            question,
            profile=DEFAULT_PROFILE,
            metadata_filter=metadata_filter,
        ),
        store or FakeStore(),  # type: ignore[arg-type]
    )
    final = (
        [c.id for c in result.outcome.chunks]
        if isinstance(result.outcome, Answerable)
        else []
    )
    return stage_ids(result.records), final


QUESTION = "What about the costs of the proceedings?"

#: name -> (what it pins and why it matters, question, how to run it)
SCENARIOS = {
    "plain": (
        (
            "no identifier, no year: vector + full-text fused, reranked; the cross-encoder "
            "threshold drops the chunks that are not about costs"
        ),
        QUESTION,
        dict,
    ),
    "years": (
        (
            "a question that names years widens both pools with year-restricted searches "
            "and reserves half of the slots for those years"
        ),
        "costs of the proceedings in 2021 and 2022",
        lambda: {"RETRIEVAL_PERIOD_FILTER": True},
    ),
    "period_filter_without_years": (
        "the period filter is on but the question names no year: nothing is widened",
        QUESTION,
        lambda: {"RETRIEVAL_PERIOD_FILTER": True},
    ),
    "years_with_metadata_filter": (
        "the year pools and the metadata filter work together",
        "costs of the proceedings in 2021",
        lambda: {
            "RETRIEVAL_PERIOD_FILTER": True,
            "metadata_filter": {"source_file": "b.docx"},
        },
    ),
    "another_reranker_has_no_score_gate": (
        (
            "a reranker that is not a cross-encoder has no calibrated scores, so nothing is "
            "dropped by a threshold: the whole reranked list goes to the final cut"
        ),
        QUESTION,
        lambda: {"RERANKER_DRIVER": "vertex", "reranker": FakeOtherReranker()},
    ),
    "one_identifier": (
        (
            "an identifier pulls every chunk of its document in as a guaranteed match, and "
            "they fill the whole context (the reason a 'five similar cases to X' question "
            "gets nothing but X)"
        ),
        f"What happened in case {CASE}?",
        dict,
    ),
    "two_identifiers_diversified": (
        (
            "two identifiers with RETRIEVAL_DIVERSIFY_GUARANTEES: the slots are shared "
            "round-robin between the two documents"
        ),
        f"Compare {CASE} and {CASE2}",
        lambda: {"RETRIEVAL_DIVERSIFY_GUARANTEES": True},
    ),
    "two_identifiers_not_diversified": (
        "the same without it: the first document takes every slot",
        f"Compare {CASE} and {CASE2}",
        lambda: {"RETRIEVAL_DIVERSIFY_GUARANTEES": False},
    ),
    "identifier_and_topic": (
        (
            "an identifier plus a topic: the identifier match is put in front of the fused "
            "list and survives the cut"
        ),
        f"costs of the proceedings in case {CASE}",
        dict,
    ),
    "cross_encoder_rejects_everything": (
        (
            "a second, hidden 'no result': when the cross-encoder scores every chunk below "
            "the threshold the final list is empty although the cosine gate passed"
        ),
        QUESTION,
        lambda: {"reranker": RejectsEverything()},
    ),
    "cosine_gate_fails": (
        (
            "the relevance gate on the raw vector similarity stops everything: only the "
            "vector stage is recorded"
        ),
        QUESTION,
        lambda: {"RETRIEVAL_MIN_SCORE": 0.95},
    ),
    "listwise_rerank": (
        "the optional LLM listwise rerank reorders the list just before the top_k cut",
        QUESTION,
        lambda: {"LISTWISE_RERANK_ENABLED": True},
    ),
    "metadata_filter": (
        "a metadata filter restricts every search it is passed to",
        QUESTION,
        lambda: {"metadata_filter": {"source_file": "b.docx"}},
    ),
    "restricted_store": (
        (
            "a store restricted to some documents (what a structured filter selects) sees "
            "only those, in every search"
        ),
        QUESTION,
        lambda: {"store": FakeStore({"c.docx", "d.docx"})},
    ),
    "csls_demotes_a_generic_chunk": (
        (
            "CSLS re-orders the vector pool by 2 * similarity - hub score, so a chunk "
            "that is close to everything (chunk 1) falls below a more specific one "
            "(chunk 3); the stage records the new order"
        ),
        QUESTION,
        lambda: {"store": FakeStore(hubs={1: 0.99, 3: 0.5})},
    ),
    "top_k_two": (
        "top_k limits the final context",
        QUESTION,
        lambda: {"RETRIEVAL_TOP_K": 2},
    ),
}

#: name -> (the ids each recorded stage held, the final ids). Note the stage called
#: "listwise" is recorded even when listwise is off: it then holds the list after
#: the cross-encoder threshold (a misleading name worth knowing about).
EXPECTED = {
    "plain": (
        {
            "dense_search.dense_pool": [1, 3, 5, 7, 2, 4],
            "csls_reorder.dense_pool": [1, 3, 5, 7, 2, 4],
            "keyword_search.keyword_pool": [1, 3, 5],
            "rrf_fusion.ranked": [1, 3, 5, 7, 2, 4],
            "rerank.ranked": [1, 3, 5, 7, 2, 4],
            "rerank_score_gate.ranked": [1, 3, 5],
            "top_k_selection.selected": [1, 3, 5],
        },
        [1, 3, 5],
    ),
    "years": (
        {
            "dense_search.dense_pool": [1, 3, 5, 7, 2, 4],
            "year_dense_widening.dense_pool": [1, 3, 5, 7, 2, 4, 6, 8],
            "year_dense_widening.year_pool": [3, 5, 7, 4, 6, 8],
            "csls_reorder.dense_pool": [1, 3, 5, 7, 2, 4, 6, 8],
            "keyword_search.keyword_pool": [1, 3, 5, 9, 10, 11, 12, 13],
            "year_keyword_widening.keyword_pool": [1, 3, 5, 9, 10, 11, 12, 13, 14],
            "year_keyword_widening.year_pool": [3, 5, 13, 14],
            "rrf_fusion.ranked": [1, 3, 5, 7, 9, 2, 10, 4, 11, 6, 12, 8, 13, 14],
            "rerank.ranked": [1, 3, 5, 7, 9, 2, 10, 4, 11, 6, 12, 8, 13, 14],
            "rerank_score_gate.ranked": [1, 3, 5],
            "top_k_selection.selected": [3, 5, 1],
        },
        [3, 5, 1],
    ),
    "period_filter_without_years": (
        {
            "dense_search.dense_pool": [1, 3, 5, 7, 2, 4],
            "year_dense_widening.dense_pool": [1, 3, 5, 7, 2, 4],
            "csls_reorder.dense_pool": [1, 3, 5, 7, 2, 4],
            "keyword_search.keyword_pool": [1, 3, 5],
            "year_keyword_widening.keyword_pool": [1, 3, 5],
            "rrf_fusion.ranked": [1, 3, 5, 7, 2, 4],
            "rerank.ranked": [1, 3, 5, 7, 2, 4],
            "rerank_score_gate.ranked": [1, 3, 5],
            "top_k_selection.selected": [1, 3, 5],
        },
        [1, 3, 5],
    ),
    "years_with_metadata_filter": (
        {
            "dense_search.dense_pool": [3, 4],
            "year_dense_widening.dense_pool": [3, 4],
            "year_dense_widening.year_pool": [3, 4],
            "csls_reorder.dense_pool": [3, 4],
            "keyword_search.keyword_pool": [3],
            "year_keyword_widening.keyword_pool": [3],
            "year_keyword_widening.year_pool": [3],
            "rrf_fusion.ranked": [3, 4],
            "rerank.ranked": [3, 4],
            "rerank_score_gate.ranked": [3],
            "top_k_selection.selected": [3],
        },
        [3],
    ),
    "another_reranker_has_no_score_gate": (
        {
            "dense_search.dense_pool": [1, 3, 5, 7, 2, 4],
            "csls_reorder.dense_pool": [1, 3, 5, 7, 2, 4],
            "keyword_search.keyword_pool": [1, 3, 5],
            "rrf_fusion.ranked": [1, 3, 5, 7, 2, 4],
            "rerank.ranked": [1, 3, 5, 7, 2, 4],
            "top_k_selection.selected": [1, 3, 5, 7],
        },
        [1, 3, 5, 7],
    ),
    "one_identifier": (
        {
            "dense_search.dense_pool": [1, 3, 5, 7, 2, 4],
            "csls_reorder.dense_pool": [1, 3, 5, 7, 2, 4],
            "keyword_search.keyword_pool": [9, 10, 11, 12, 13, 14],
            "rrf_fusion.ranked": [1, 9, 3, 10, 5, 11, 7, 12, 2, 13, 4, 14],
            "rerank.ranked": [1, 3, 5, 9, 10, 11, 7, 12, 2, 13, 4, 14],
            "rerank_score_gate.ranked": [1, 3, 5],
            "top_k_selection.selected": [1, 3, 5],
        },
        [1, 3, 5],
    ),
    "two_identifiers_diversified": (
        {
            "dense_search.dense_pool": [1, 3, 5, 7, 2, 4],
            "csls_reorder.dense_pool": [1, 3, 5, 7, 2, 4],
            "keyword_search.keyword_pool": [9, 10, 11, 12, 13, 14],
            "rrf_fusion.ranked": [1, 9, 3, 10, 5, 11, 7, 12, 2, 13, 4, 14],
            "rerank.ranked": [1, 3, 5, 9, 10, 11, 7, 12, 2, 13, 4, 14],
            "rerank_score_gate.ranked": [1, 3, 5],
            "top_k_selection.selected": [1, 3, 5],
        },
        [1, 3, 5],
    ),
    "two_identifiers_not_diversified": (
        {
            "dense_search.dense_pool": [1, 3, 5, 7, 2, 4],
            "csls_reorder.dense_pool": [1, 3, 5, 7, 2, 4],
            "keyword_search.keyword_pool": [9, 10, 11, 12, 13, 14],
            "rrf_fusion.ranked": [1, 9, 3, 10, 5, 11, 7, 12, 2, 13, 4, 14],
            "rerank.ranked": [1, 3, 5, 9, 10, 11, 7, 12, 2, 13, 4, 14],
            "rerank_score_gate.ranked": [1, 3, 5],
            "top_k_selection.selected": [1, 3, 5],
        },
        [1, 3, 5],
    ),
    "identifier_and_topic": (
        {
            "dense_search.dense_pool": [1, 3, 5, 7, 2, 4],
            "csls_reorder.dense_pool": [1, 3, 5, 7, 2, 4],
            "keyword_search.keyword_pool": [1, 3, 5, 9, 10, 11],
            "rrf_fusion.ranked": [1, 3, 5, 7, 9, 2, 10, 4, 11],
            "rerank.ranked": [1, 3, 5, 7, 9, 2, 10, 4, 11],
            "rerank_score_gate.ranked": [1, 3, 5],
            "top_k_selection.selected": [1, 3, 5],
        },
        [1, 3, 5],
    ),
    "cross_encoder_rejects_everything": (
        {
            "dense_search.dense_pool": [1, 3, 5, 7, 2, 4],
            "csls_reorder.dense_pool": [1, 3, 5, 7, 2, 4],
            "keyword_search.keyword_pool": [1, 3, 5],
            "rrf_fusion.ranked": [1, 3, 5, 7, 2, 4],
            "rerank.ranked": [1, 3, 5, 7, 2, 4],
        },
        [],
    ),
    "cosine_gate_fails": ({"dense_search.dense_pool": [1, 3, 5, 7, 2, 4]}, []),
    "listwise_rerank": (
        {
            "dense_search.dense_pool": [1, 3, 5, 7, 2, 4],
            "csls_reorder.dense_pool": [1, 3, 5, 7, 2, 4],
            "keyword_search.keyword_pool": [1, 3, 5],
            "rrf_fusion.ranked": [1, 3, 5, 7, 2, 4],
            "rerank.ranked": [1, 3, 5, 7, 2, 4],
            "rerank_score_gate.ranked": [1, 3, 5],
            "listwise_rerank.ranked": [5, 3, 1],
            "top_k_selection.selected": [5, 3, 1],
        },
        [5, 3, 1],
    ),
    "metadata_filter": (
        {
            "dense_search.dense_pool": [3, 4],
            "csls_reorder.dense_pool": [3, 4],
            "keyword_search.keyword_pool": [3],
            "rrf_fusion.ranked": [3, 4],
            "rerank.ranked": [3, 4],
            "rerank_score_gate.ranked": [3],
            "top_k_selection.selected": [3],
        },
        [3],
    ),
    "restricted_store": (
        {
            "dense_search.dense_pool": [5, 7, 6, 8],
            "csls_reorder.dense_pool": [5, 7, 6, 8],
            "keyword_search.keyword_pool": [5],
            "rrf_fusion.ranked": [5, 7, 6, 8],
            "rerank.ranked": [5, 7, 6, 8],
            "rerank_score_gate.ranked": [5],
            "top_k_selection.selected": [5],
        },
        [5],
    ),
    "csls_demotes_a_generic_chunk": (
        {
            "dense_search.dense_pool": [1, 3, 5, 7, 2, 4],
            "csls_reorder.dense_pool": [3, 1, 5, 7, 2, 4],
            "keyword_search.keyword_pool": [1, 3, 5],
            "rrf_fusion.ranked": [3, 1, 5, 7, 2, 4],
            "rerank.ranked": [3, 1, 5, 7, 2, 4],
            "rerank_score_gate.ranked": [3, 1, 5],
            "top_k_selection.selected": [3, 1, 5],
        },
        [3, 1, 5],
    ),
    "top_k_two": (
        {
            "dense_search.dense_pool": [1, 3, 5, 7, 2, 4],
            "csls_reorder.dense_pool": [1, 3, 5, 7, 2, 4],
            "keyword_search.keyword_pool": [1, 3, 5],
            "rrf_fusion.ranked": [1, 3, 5, 7, 2, 4],
            "rerank.ranked": [1, 3, 5, 7, 2, 4],
            "rerank_score_gate.ranked": [1, 3, 5],
            "top_k_selection.selected": [1, 3],
        },
        [1, 3],
    ),
}


#: The scenarios that name an identifier: the original pulled the named document's chunks
#: in (a text search), the new pipeline does not; see the module docstring.
IDENTIFIER_SCENARIOS = {
    "one_identifier",
    "two_identifiers_diversified",
    "two_identifiers_not_diversified",
    "identifier_and_topic",
}


@pytest.mark.parametrize("name", SCENARIOS)
def test_the_pipeline_still_does_exactly_what_it_did(name, settings_override):
    """The ids every step left and the final context of every scenario are unchanged."""
    _, question, how = SCENARIOS[name]

    assert run_scenario(settings_override, question, **how()) == EXPECTED[name]


@pytest.mark.parametrize("name", sorted(IDENTIFIER_SCENARIOS))
def test_an_identifier_in_the_question_does_not_pull_its_chunks_in(
    name, settings_override
):
    """Naming a document restricts the *store* (the scope), not the ranking: the fixture's
    identifier chunks (ids 9-14) are not forced into the context."""
    _, question, how = SCENARIOS[name]

    stages, final = run_scenario(settings_override, question, **how())

    assert not any(key.startswith("identifier_pin") for key in stages)
    assert not set(final) & {9, 10, 11, 12, 13, 14}


def test_two_runs_of_a_scenario_give_the_same_result(settings_override):
    """The pinned results only mean something if the fixtures are deterministic."""
    _, question, how = SCENARIOS["years"]

    first = run_scenario(settings_override, question, **how())
    second = run_scenario(settings_override, question, **how())

    assert first == second


def test_the_scenarios_cover_every_step_the_funnel_describes(settings_override):
    """Step names are a contract with corpus/commands/funnel.py: if one stops being
    produced, the funnel silently loses a row."""
    seen: set[str] = set()
    for _, question, how in SCENARIOS.values():
        stages, _ = run_scenario(settings_override, question, **how())
        seen |= {key.split(".")[0] for key in stages}

    assert seen >= {
        "dense_search",
        "year_dense_widening",
        "csls_reorder",
        "keyword_search",
        "year_keyword_widening",
        "rrf_fusion",
        "rerank",
        "rerank_score_gate",
        "listwise_rerank",
        "top_k_selection",
    }
