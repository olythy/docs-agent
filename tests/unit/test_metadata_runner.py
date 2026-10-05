"""Tests for metadata.runner / sources / conversion / evidence, with in-memory fakes (no DB, no LLM)."""

from dataclasses import replace
from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock

from metadata.conversion import to_meta_value
from metadata.evidence import EvidenceSelector
from metadata.runner import MetaExtractionRunner
from metadata.sources import (
    Candidate,
    ChunkMetadataSource,
    LLMMetaSource,
    MetaSource,
    ProposedKey,
    SourceResult,
)
from models import (
    ChunkMetadata,
    Document,
    KeyStatus,
    MetaKey,
    MetaState,
    MetaStatus,
    MetaValue,
    RetrievedChunk,
    ValueType,
)
from models import (
    MetaSource as Kind,
)

HASH = "a" * 64
DOC = Document(HASH, "a.docx")


def _chunk(i, body, **meta):
    return RetrievedChunk(
        id=i,
        content=body,
        metadata=ChunkMetadata(
            source_file="a.docx", page_number=i + 1, chunk_index=i, **meta
        ),
        score=0.0,
    )


def _key(name, vtype=ValueType.TEXT, **kw):
    return MetaKey(
        "court_decision",
        name,
        vtype,
        f"desc of {name}",
        status=KeyStatus.APPROVED,
        **kw,
    )


class FakeDocuments:
    """In-memory DocumentRepository."""

    def __init__(self, keys, documents=(DOC,)):
        self.keys = {k.key: k for k in keys}
        self.documents = list(documents)
        self.statuses: dict[tuple[str, str], MetaStatus] = {}
        self.values: dict[tuple[str, str], list[MetaValue]] = {}

    def list_keys(self, doc_type, status=None):
        return [k for k in self.keys.values() if status is None or k.status == status]

    def documents_needing(self, keys, limit=None, seed=None):
        out = [
            d
            for d in self.documents
            if any(
                (d.content_hash, k.key) not in self.statuses
                or self.statuses[(d.content_hash, k.key)].key_version < k.version
                for k in keys
            )
        ]
        return out[:limit] if limit else out

    def get_statuses(self, content_hash):
        return {k: s for (h, k), s in self.statuses.items() if h == content_hash}

    def replace_values(self, content_hash, key, values):
        self.values[(content_hash, key)] = list(values)

    def set_status(self, status):
        self.statuses[(status.content_hash, status.key)] = status

    def upsert_key(self, key):
        self.keys[key.key] = key


class FakeChunks:
    def __init__(self, chunks):
        self._chunks = chunks

    def get_document_chunks(self, content_hash):
        return list(self._chunks)


class ScriptedSource(MetaSource):
    kind = Kind.LLM
    needs_verification = True

    def __init__(self, result, supports=lambda k: True):
        self.result, self._supports, self.calls = result, supports, []

    def supports(self, key):
        return self._supports(key)

    def extract(self, chunks, keys):
        self.calls.append(([c.content for c in chunks], [k.key for k in keys]))
        return self.result


def _runner(documents, chunks, sources, **kw):
    selector = EvidenceSelector(MagicMock(), per_key=2, max_chunks=6)
    return MetaExtractionRunner(documents, FakeChunks(chunks), selector, sources, **kw)


def _status(docs, key):
    return docs.statuses[(HASH, key)].state


BODY = [
    _chunk(
        0, "A Budapest Környéki Törvényszék ítélete. Budapest, 2024. december 16. bíró"
    )
]


def test_a_value_whose_quote_checks_out_is_stored_and_marked_present():
    docs = FakeDocuments([_key("court"), _key("decision_date", ValueType.DATE)])
    source = ScriptedSource(
        SourceResult(
            candidates=[
                Candidate(
                    "court",
                    "Budapest Környéki Törvényszék",
                    evidence="Budapest Környéki Törvényszék",
                ),
                Candidate(
                    "decision_date",
                    "2024-12-16",
                    evidence="Budapest, 2024. december 16.",
                ),
            ]
        )
    )

    report = _runner(docs, BODY, [source]).run("court_decision")

    assert (report.documents, report.present, report.unverified) == (1, 2, 0)
    assert docs.values[(HASH, "decision_date")][0].value_date == date(2024, 12, 16)
    assert _status(docs, "court") is MetaState.PRESENT


def test_an_invented_quote_is_unverified_and_stores_no_value():
    docs = FakeDocuments([_key("court")])
    source = ScriptedSource(
        SourceResult(
            candidates=[
                Candidate("court", "Egri Törvényszék", evidence="Egri Törvényszék")
            ]
        )
    )

    report = _runner(docs, BODY, [source]).run("court_decision")

    assert (report.present, report.unverified) == (0, 1)
    assert _status(docs, "court") is MetaState.UNVERIFIED
    assert docs.values[(HASH, "court")] == []


def test_a_key_the_source_did_not_return_is_confirmed_absent():
    docs = FakeDocuments([_key("court")])

    report = _runner(docs, BODY, [ScriptedSource(SourceResult())]).run("court_decision")

    assert report.confirmed_absent == 1
    assert _status(docs, "court") is MetaState.CONFIRMED_ABSENT


def test_a_failed_source_leaves_the_key_pending_so_the_next_run_retries_it():
    docs = FakeDocuments([_key("court")])

    report = _runner(docs, BODY, [ScriptedSource(SourceResult(failed=True))]).run(
        "court_decision"
    )

    assert report.failed == 1
    assert (HASH, "court") not in docs.statuses
    assert docs.documents_needing(docs.list_keys("court_decision")) == [DOC]


def test_a_second_run_does_nothing_and_a_bumped_key_version_redoes_that_key():
    docs = FakeDocuments([_key("court")])
    source = ScriptedSource(
        SourceResult(
            candidates=[
                Candidate(
                    "court",
                    "Budapest Környéki Törvényszék",
                    evidence="Budapest Környéki Törvényszék",
                )
            ]
        )
    )
    runner = _runner(docs, BODY, [source])
    runner.run("court_decision")

    again = runner.run("court_decision")
    docs.keys["court"] = replace(docs.keys["court"], version=2)
    redone = runner.run("court_decision")

    assert again.documents == 0 and len(source.calls) == 2
    assert redone.documents == 1 and docs.statuses[(HASH, "court")].key_version == 2


def test_only_pending_keys_are_extracted():
    docs = FakeDocuments([_key("court"), _key("kind")])
    docs.set_status(MetaStatus(HASH, "court", MetaState.PRESENT, key_version=1))
    source = ScriptedSource(SourceResult())

    _runner(docs, BODY, [source]).run("court_decision")

    assert source.calls[0][1] == ["kind"]


def test_each_key_goes_to_the_first_source_that_supports_it():
    docs = FakeDocuments([_key("court"), _key("decision_date", ValueType.DATE)])
    deterministic = ChunkMetadataSource({"decision_date": "document_date"})
    llm = ScriptedSource(SourceResult())
    chunks = [_chunk(0, "text", document_date="2024-12-16")]

    report = _runner(docs, chunks, [deterministic, llm]).run("court_decision")

    assert llm.calls[0][1] == ["court"]  # the LLM was not asked for the date
    assert docs.values[(HASH, "decision_date")][0].source is Kind.DETERMINISTIC
    assert docs.values[(HASH, "decision_date")][0].value_date == date(2024, 12, 16)
    assert report.present == 1 and report.confirmed_absent == 1


def test_a_list_valued_chunk_field_becomes_one_value_per_distinct_element():
    docs = FakeDocuments([_key("document_identifier", multi_valued=True)])
    source = ChunkMetadataSource({"document_identifier": "document_identifiers"})
    chunks = [
        _chunk(
            0,
            "text",
            document_identifiers=(
                "27.P.20.093/2020/13-II",
                "27.P.20.093/2020/13",
                "27.P.20.093/2020/13",
            ),
        )
    ]

    report = _runner(docs, chunks, [source]).run("court_decision")

    stored = docs.values[(HASH, "document_identifier")]
    assert [(v.ordinal, v.value_text) for v in stored] == [
        (0, "27.P.20.093/2020/13-II"),
        (1, "27.P.20.093/2020/13"),
    ]
    assert stored[0].source is Kind.DETERMINISTIC and report.present == 1


def test_a_document_without_identifiers_is_confirmed_absent():
    docs = FakeDocuments([_key("document_identifier", multi_valued=True)])
    source = ChunkMetadataSource({"document_identifier": "document_identifiers"})

    report = _runner(docs, [_chunk(0, "text")], [source]).run("court_decision")

    assert (
        report.confirmed_absent == 1
        and (HASH, "document_identifier") not in docs.values
    )


def test_a_single_valued_key_keeps_only_the_first_candidate_a_multi_valued_one_keeps_all():
    one = FakeDocuments([_key("a")])
    many = FakeDocuments([_key("a", multi_valued=True)])
    both = SourceResult(
        candidates=[
            Candidate("a", "Budapest", evidence="Budapest"),
            Candidate("a", "Törvényszék", evidence="Törvényszék"),
        ]
    )

    _runner(one, BODY, [ScriptedSource(both)]).run("court_decision")
    _runner(many, BODY, [ScriptedSource(both)]).run("court_decision")

    assert len(one.values[(HASH, "a")]) == 1
    assert [v.ordinal for v in many.values[(HASH, "a")]] == [0, 1]


def test_a_proposed_key_is_stored_as_proposed_and_never_overwrites_a_known_one():
    docs = FakeDocuments([_key("court")])
    source = ScriptedSource(
        SourceResult(
            proposed_keys=[
                ProposedKey("court", "dup", "text"),
                ProposedKey("case_topic", "Subject", "text"),
            ]
        )
    )

    report = _runner(docs, BODY, [source]).run("court_decision")

    assert report.proposed_keys == 1
    assert docs.keys["case_topic"].status is KeyStatus.PROPOSED
    assert docs.keys["court"].description == "desc of court"


def test_limit_caps_the_documents_processed():
    docs = FakeDocuments(
        [_key("court")],
        documents=[Document("a" * 64, "a.docx"), Document("b" * 64, "b.docx")],
    )

    report = _runner(docs, BODY, [ScriptedSource(SourceResult())]).run(
        "court_decision", limit=1
    )

    assert report.documents == 1


def test_the_extractor_reads_the_documents_own_text_not_the_embedded_summary():
    docs = FakeDocuments([_key("court")])
    seen = []

    class Spy(ScriptedSource):
        def extract(self, chunks, keys):
            seen.extend(c.content for c in chunks)
            return super().extract(chunks, keys)

    stored = _chunk(
        0,
        "Summary text\n\n4.P.1/2020/1\n\nbody text",
        document_summary="Summary text",
        document_identifiers=("4.P.1/2020/1",),
    )

    _runner(docs, [stored], [Spy(SourceResult())]).run("court_decision")

    assert seen == ["body text"]


# ------------------------------------------------------------------ sources / conversion / selector


def test_the_llm_source_parses_values_drops_unknown_keys_and_locates_the_quote():
    llm = MagicMock()
    llm.run_tool_calling_turn.return_value.content = (
        '{"values": [{"key": "court", "value": "Egri Törvényszék", "evidence": "Egri  Törvényszék"},'
        ' {"key": "ghost", "value": "x", "evidence": "x"}, {"key": "court", "value": ""}],'
        ' "proposed_keys": [{"key": "case_topic", "description": "d", "value_type": "text"}]}'
    )
    chunks = [_chunk(3, "xx Egri Törvényszék yy")]

    result = LLMMetaSource(llm).extract(chunks, [_key("court")])

    assert [(c.key, c.value, c.evidence_chunk_index) for c in result.candidates] == [
        ("court", "Egri Törvényszék", 3)
    ]
    assert [p.key for p in result.proposed_keys] == ["case_topic"]
    assert not result.failed


def test_the_llm_source_reports_failure_when_the_reply_is_not_json():
    llm = MagicMock()
    llm.run_tool_calling_turn.return_value.content = "sorry, I cannot do that"

    assert LLMMetaSource(llm).extract([_chunk(0, "x")], [_key("court")]).failed


def test_the_llm_prompt_lists_allowed_values_and_demands_a_verbatim_quote():
    llm = MagicMock()
    llm.run_tool_calling_turn.return_value.content = "{}"
    key = _key("document_kind", allowed_values=("judgment", "order"))

    LLMMetaSource(llm).extract([_chunk(0, "x")], [key])

    prompt = llm.run_tool_calling_turn.call_args.args[0][0]["content"]
    assert "judgment, order" in prompt and "VERBATIM" in prompt


def test_conversion_types_each_value_and_rejects_text_that_is_not_one():
    def convert(vtype, text):
        return to_meta_value(_key("k", vtype), Candidate("k", text), HASH, Kind.LLM)

    def typed(vtype, text):
        value = convert(vtype, text)
        assert value is not None
        return value

    assert typed(ValueType.NUMBER, "629920").value_number == Decimal(629920)
    assert typed(ValueType.DATE, "2024-12-16").value_date == date(2024, 12, 16)
    assert typed(ValueType.BOOL, "yes").value_bool is True
    assert typed(ValueType.TEXT, " x ").value_text == "x"
    assert convert(ValueType.DATE, "16/12/2024") is None
    assert convert(ValueType.NUMBER, "a lot") is None
    assert convert(ValueType.BOOL, "maybe") is None


def test_the_selector_returns_a_short_document_whole_and_always_includes_the_edges_of_a_long_one():
    chunks = [_chunk(i, f"c{i}") for i in range(10)]
    reranker = MagicMock()
    # the reranker likes the middle of the document best
    reranker.rerank.side_effect = lambda query, cs: sorted(
        cs, key=lambda c: abs(c.metadata.chunk_index - 5)
    )
    selector = EvidenceSelector(reranker, per_key=2, max_chunks=4)

    assert [c.id for c in selector.select(chunks[:4], [_key("court")])] == [
        0,
        1,
        2,
        3,
    ]  # short: whole
    picked = selector.select(chunks, [_key("court"), _key("decision_date")])

    assert [c.metadata.chunk_index for c in picked] == [
        0,
        4,
        5,
        9,
    ]  # edges + the best two, in order
    assert reranker.rerank.call_args_list[0].args[0] == "court: desc of court"


def test_the_edges_can_be_switched_off():
    chunks = [_chunk(i, f"c{i}") for i in range(10)]
    reranker = MagicMock()
    reranker.rerank.side_effect = lambda query, cs: sorted(
        cs, key=lambda c: abs(c.metadata.chunk_index - 5)
    )

    picked = EvidenceSelector(
        reranker, per_key=2, max_chunks=4, always_edges=False
    ).select(chunks, [_key("court")])

    assert [c.metadata.chunk_index for c in picked] == [4, 5]
