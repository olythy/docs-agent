"""Characterization tests: what the retrieval pipeline does today, pinned end to end.

These do not say the behaviour is *right*; they say it is *this*. The retrieval
path is the most heavily measured part of the system (see docs/decisions.md), and
it is about to be restructured (steps pulled out of ``HybridRetrievalStrategy``,
a shared state, profiles). Every such step must leave these results identical;
when a result is *meant* to change, the change is made here on purpose, in the
same commit, and explained.

A small deterministic corpus, a fake ``VectorStore`` that really filters by year /
metadata and really searches by identifier, and a fake reranker run the whole of
:func:`query.retrieval.retrieve_chunks`. For each scenario both the recorded stages
(the ids each stage held: what ``funnel`` shows) and the final context are pinned.

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

import query.retrieval as retrieval_module
from drivers.reranker import CrossEncoderRerankerDriver, RerankerDriver
from models import ChunkMetadata, RetrievalTrace, RetrievedChunk
from query.facts import QueryFactsReader
from query.outcome import Answerable
from query.profiles import PipelineFactory, ProfileResolver
from query.retrieval import retrieve_chunks
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
    "RETRIEVAL_STRATEGY": "hybrid",
    "RERANKER_DRIVER": "cross_encoder",
    "RETRIEVAL_TOP_K": 4,
    "RETRIEVAL_CANDIDATE_POOL_SIZE": 6,
    "RETRIEVAL_MIN_SCORE": 0.25,
    "RERANKER_MIN_SCORE": 0.5,
    "RETRIEVAL_PERIOD_FILTER": False,
    "RETRIEVAL_DIVERSIFY_GUARANTEES": True,
    "LISTWISE_RERANK_ENABLED": False,
}


def _run_legacy(
    monkeypatch,
    settings_override,
    question,
    *,
    store=None,
    reranker=None,
    metadata_filter=None,
    **settings,
):
    """Run the original pipeline (``retrieve_chunks`` with the strategy classes).

    A scenario is described engine-neutrally (settings overrides, a store, a reranker,
    a metadata filter); an engine turns that into a run and returns the ids each
    recorded stage held and the final ids.
    """
    base = {**_BASE_SETTINGS, **settings}
    monkeypatch.setattr(retrieval_module, "settings", settings_override(**base))
    monkeypatch.setattr(
        retrieval_module, "get_embedding_driver", lambda: FakeEmbedding()
    )
    monkeypatch.setattr(
        retrieval_module,
        "get_reranker_driver",
        lambda *a, **k: reranker or FakeCrossEncoder(),
    )
    monkeypatch.setattr(retrieval_module, "get_answer_driver", lambda: object())
    monkeypatch.setattr(retrieval_module, "listwise_rerank", FakeListwise())
    trace = RetrievalTrace()
    chunks = retrieve_chunks(
        question,
        store=store or FakeStore(),  # type: ignore[arg-type]
        trace=trace,
        metadata_filter=metadata_filter,
    )
    return {k: [c.id for c in v] for k, v in trace.stages.items()}, [
        c.id for c in chunks
    ]


def _run_v2(
    monkeypatch,
    settings_override,
    question,
    *,
    store=None,
    reranker=None,
    metadata_filter=None,
    **settings,
):
    """Run the new pipeline (``RetrievalService``) on the same neutral description."""
    merged = {**_BASE_SETTINGS, **settings}
    config = settings_override(**merged)
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
            profile=config.RETRIEVAL_STRATEGY,
            metadata_filter=metadata_filter,
        ),
        store or FakeStore(),  # type: ignore[arg-type]
    )
    final = (
        [c.id for c in result.outcome.chunks]
        if isinstance(result.outcome, Answerable)
        else []
    )
    return {k: [c.id for c in v] for k, v in result.trace.stages.items()}, final


#: engine name -> how to run a scenario on it. A new pipeline implementation is added
#: here and must reproduce every ``EXPECTED`` result unchanged.
ENGINES = {"legacy": _run_legacy, "v2": _run_v2}


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
    "vector_strategy": (
        "the plain vector strategy: no fusion, no rerank, only the similarity cut",
        QUESTION,
        lambda: {"RETRIEVAL_STRATEGY": "vector"},
    ),
    "vector_cosine_gate_fails": (
        "the same gate in the vector profile: only the vector stage, no final",
        QUESTION,
        lambda: {"RETRIEVAL_STRATEGY": "vector", "RETRIEVAL_MIN_SCORE": 0.95},
    ),
    "vector_metadata_filter": (
        "the vector profile passes the metadata filter to its search",
        QUESTION,
        lambda: {
            "RETRIEVAL_STRATEGY": "vector",
            "metadata_filter": {"source_file": "b.docx"},
        },
    ),
    "vector_similarity_cut": (
        "the vector profile drops candidates below the similarity threshold",
        QUESTION,
        lambda: {
            "RETRIEVAL_STRATEGY": "vector",
            "RETRIEVAL_MIN_SCORE": 0.7,
            "RETRIEVAL_TOP_K": 6,
        },
    ),
    "vector_top_k_above_pool": (
        "a top_k larger than the configured pool still gets a pool of top_k candidates",
        QUESTION,
        lambda: {"RETRIEVAL_STRATEGY": "vector", "RETRIEVAL_TOP_K": 8},
    ),
    "vector_top_k_two": (
        "the vector profile's top_k limits the final context",
        QUESTION,
        lambda: {"RETRIEVAL_STRATEGY": "vector", "RETRIEVAL_TOP_K": 2},
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
            "vector": [1, 3, 5, 7, 2, 4],
            "vector_csls": [1, 3, 5, 7, 2, 4],
            "fulltext": [1, 3, 5],
            "fused": [1, 3, 5, 7, 2, 4],
            "reranked": [1, 3, 5, 7, 2, 4],
            "listwise": [1, 3, 5],
            "final": [1, 3, 5],
        },
        [1, 3, 5],
    ),
    "years": (
        {
            "vector": [1, 3, 5, 7, 2, 4],
            "vector_years": [3, 5, 7, 4, 6, 8],
            "vector_csls": [1, 3, 5, 7, 2, 4, 6, 8],
            "fulltext": [1, 3, 5, 9, 10, 11, 12, 13],
            "fulltext_years": [3, 5, 13, 14],
            "fused": [1, 3, 5, 7, 9, 2, 10, 4, 11, 6, 12, 8, 13, 14],
            "reranked": [1, 3, 5, 7, 9, 2, 10, 4, 11, 6, 12, 8, 13, 14],
            "listwise": [1, 3, 5],
            "final": [3, 5, 1],
        },
        [3, 5, 1],
    ),
    "period_filter_without_years": (
        {
            "vector": [1, 3, 5, 7, 2, 4],
            "vector_csls": [1, 3, 5, 7, 2, 4],
            "fulltext": [1, 3, 5],
            "fused": [1, 3, 5, 7, 2, 4],
            "reranked": [1, 3, 5, 7, 2, 4],
            "listwise": [1, 3, 5],
            "final": [1, 3, 5],
        },
        [1, 3, 5],
    ),
    "years_with_metadata_filter": (
        {
            "vector": [3, 4],
            "vector_years": [3, 4],
            "vector_csls": [3, 4],
            "fulltext": [3],
            "fulltext_years": [3],
            "fused": [3, 4],
            "reranked": [3, 4],
            "listwise": [3],
            "final": [3],
        },
        [3],
    ),
    "another_reranker_has_no_score_gate": (
        {
            "vector": [1, 3, 5, 7, 2, 4],
            "vector_csls": [1, 3, 5, 7, 2, 4],
            "fulltext": [1, 3, 5],
            "fused": [1, 3, 5, 7, 2, 4],
            "reranked": [1, 3, 5, 7, 2, 4],
            "listwise": [1, 3, 5, 7, 2, 4],
            "final": [1, 3, 5, 7],
        },
        [1, 3, 5, 7],
    ),
    "one_identifier": (
        {
            "vector": [1, 3, 5, 7, 2, 4],
            "vector_csls": [1, 3, 5, 7, 2, 4],
            "fulltext": [9, 10, 11, 12, 13, 14],
            "identifier": [9, 10, 11, 12],
            "fused": [1, 9, 3, 10, 5, 11, 7, 12, 2, 13, 4, 14],
            "reranked": [1, 3, 5, 9, 10, 11, 7, 12, 2, 13, 4, 14],
            "listwise": [1, 3, 5, 9, 10, 11, 12],
            "final": [9, 10, 11, 12],
        },
        [9, 10, 11, 12],
    ),
    "two_identifiers_diversified": (
        {
            "vector": [1, 3, 5, 7, 2, 4],
            "vector_csls": [1, 3, 5, 7, 2, 4],
            "fulltext": [9, 10, 11, 12, 13, 14],
            "identifier": [9, 10, 11, 12, 13, 14],
            "fused": [1, 9, 3, 10, 5, 11, 7, 12, 2, 13, 4, 14],
            "reranked": [1, 3, 5, 9, 10, 11, 7, 12, 2, 13, 4, 14],
            "listwise": [1, 3, 5, 9, 10, 11, 12, 13, 14],
            "final": [9, 13, 10, 14],
        },
        [9, 13, 10, 14],
    ),
    "two_identifiers_not_diversified": (
        {
            "vector": [1, 3, 5, 7, 2, 4],
            "vector_csls": [1, 3, 5, 7, 2, 4],
            "fulltext": [9, 10, 11, 12, 13, 14],
            "identifier": [9, 10, 11, 12, 13, 14],
            "fused": [1, 9, 3, 10, 5, 11, 7, 12, 2, 13, 4, 14],
            "reranked": [1, 3, 5, 9, 10, 11, 7, 12, 2, 13, 4, 14],
            "listwise": [1, 3, 5, 9, 10, 11, 12, 13, 14],
            "final": [9, 10, 11, 12],
        },
        [9, 10, 11, 12],
    ),
    "identifier_and_topic": (
        {
            "vector": [1, 3, 5, 7, 2, 4],
            "vector_csls": [1, 3, 5, 7, 2, 4],
            "fulltext": [1, 3, 5, 9, 10, 11],
            "identifier": [9, 10, 11, 12],
            "fused": [12, 1, 3, 5, 7, 9, 2, 10, 4, 11],
            "reranked": [1, 3, 5, 12, 7, 9, 2, 10, 4, 11],
            "listwise": [1, 3, 5, 12, 9, 10, 11],
            "final": [12, 9, 10, 11],
        },
        [12, 9, 10, 11],
    ),
    "cross_encoder_rejects_everything": (
        {
            "vector": [1, 3, 5, 7, 2, 4],
            "vector_csls": [1, 3, 5, 7, 2, 4],
            "fulltext": [1, 3, 5],
            "fused": [1, 3, 5, 7, 2, 4],
            "reranked": [1, 3, 5, 7, 2, 4],
            "final": [],
        },
        [],
    ),
    "cosine_gate_fails": (
        {
            "vector": [1, 3, 5, 7, 2, 4],
        },
        [],
    ),
    "listwise_rerank": (
        {
            "vector": [1, 3, 5, 7, 2, 4],
            "vector_csls": [1, 3, 5, 7, 2, 4],
            "fulltext": [1, 3, 5],
            "fused": [1, 3, 5, 7, 2, 4],
            "reranked": [1, 3, 5, 7, 2, 4],
            "listwise": [5, 3, 1],
            "final": [5, 3, 1],
        },
        [5, 3, 1],
    ),
    "vector_strategy": (
        {
            "vector": [1, 3, 5, 7, 2, 4],
            "final": [1, 3, 5, 7],
        },
        [1, 3, 5, 7],
    ),
    "vector_cosine_gate_fails": (
        {"vector": [1, 3, 5, 7, 2, 4]},
        [],
    ),
    "vector_metadata_filter": (
        {"vector": [3, 4], "final": [3, 4]},
        [3, 4],
    ),
    "vector_similarity_cut": (
        {"vector": [1, 3, 5, 7, 2, 4], "final": [1, 3, 5, 7]},
        [1, 3, 5, 7],
    ),
    "vector_top_k_above_pool": (
        {"vector": [1, 3, 5, 7, 2, 4, 6, 8], "final": [1, 3, 5, 7, 2, 4, 6, 8]},
        [1, 3, 5, 7, 2, 4, 6, 8],
    ),
    "vector_top_k_two": (
        {"vector": [1, 3, 5, 7, 2, 4], "final": [1, 3]},
        [1, 3],
    ),
    "metadata_filter": (
        {
            "vector": [3, 4],
            "vector_csls": [3, 4],
            "fulltext": [3],
            "fused": [3, 4],
            "reranked": [3, 4],
            "listwise": [3],
            "final": [3],
        },
        [3],
    ),
    "restricted_store": (
        {
            "vector": [5, 7, 6, 8],
            "vector_csls": [5, 7, 6, 8],
            "fulltext": [5],
            "fused": [5, 7, 6, 8],
            "reranked": [5, 7, 6, 8],
            "listwise": [5],
            "final": [5],
        },
        [5],
    ),
    "csls_demotes_a_generic_chunk": (
        {
            "vector": [1, 3, 5, 7, 2, 4],
            "vector_csls": [3, 1, 5, 7, 2, 4],
            "fulltext": [1, 3, 5],
            "fused": [3, 1, 5, 7, 2, 4],
            "reranked": [3, 1, 5, 7, 2, 4],
            "listwise": [3, 1, 5],
            "final": [3, 1, 5],
        },
        [3, 1, 5],
    ),
    "top_k_two": (
        {
            "vector": [1, 3, 5, 7, 2, 4],
            "vector_csls": [1, 3, 5, 7, 2, 4],
            "fulltext": [1, 3, 5],
            "fused": [1, 3, 5, 7, 2, 4],
            "reranked": [1, 3, 5, 7, 2, 4],
            "listwise": [1, 3, 5],
            "final": [1, 3],
        },
        [1, 3],
    ),
}


@pytest.mark.parametrize(
    ("engine", "name"), [(e, n) for e in ENGINES for n in SCENARIOS]
)
def test_the_pipeline_still_does_exactly_what_it_did(
    engine, name, monkeypatch, settings_override
):
    """The recorded stages and the final context of every scenario are unchanged."""
    _, question, how = SCENARIOS[name]

    stages, final = ENGINES[engine](monkeypatch, settings_override, question, **how())

    assert (stages, final) == EXPECTED[name]


@pytest.mark.parametrize("engine", ["legacy"])
def test_two_runs_of_a_scenario_give_the_same_result(
    engine, monkeypatch, settings_override
):
    """The pinned results only mean something if the fixtures are deterministic."""
    _, question, how = SCENARIOS["one_identifier"]
    run = ENGINES[engine]

    first = run(monkeypatch, settings_override, question, **how())
    second = run(monkeypatch, settings_override, question, **how())

    assert first == second


@pytest.mark.parametrize("engine", ["legacy"])
def test_the_scenarios_cover_every_stage_name_the_funnel_reads(
    engine, monkeypatch, settings_override
):
    """Stage names are a contract with corpus/commands/funnel.py: if one stops being
    produced, the funnel silently loses a column."""
    seen: set[str] = set()
    for _, question, how in SCENARIOS.values():
        stages, _ = ENGINES[engine](monkeypatch, settings_override, question, **how())
        seen |= set(stages)

    assert seen >= {
        "vector",
        "vector_years",
        "vector_csls",
        "fulltext",
        "fulltext_years",
        "identifier",
        "fused",
        "reranked",
        "listwise",
        "final",
    }
