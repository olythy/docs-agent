"""Tests for corpus.commands.funnel's pure helpers and models.RetrievalTrace."""

from corpus.commands.funnel import (
    court_of,
    courts_named,
    document_ranks,
    satisfies,
    verdict,
)
from models import ChunkMetadata, RetrievalTrace, RetrievedChunk


def _chunk(chunk_id, source):
    return RetrievedChunk(
        id=chunk_id,
        content="x",
        metadata=ChunkMetadata(source_file=source, page_number=None, chunk_index=0),
        score=0.0,
    )


def test_document_ranks_numbers_distinct_documents_by_first_appearance():
    chunks = [_chunk(1, "a.docx"), _chunk(2, "a.docx"), _chunk(3, "b.docx")]

    assert document_ranks(chunks) == {"a.docx": 1, "b.docx": 2}


def test_court_is_the_file_name_prefix():
    assert court_of("Egri_Torvenyszek__P_20125_2022_53.docx") == "Egri_Torvenyszek"


def test_courts_named_matches_accent_insensitively_against_the_question():
    files = {
        "Egri_Torvenyszek__P_1_2022_1.docx",
        "Balassagyarmati_Torvenyszek__G_1_2020_1.docx",
        "Debreceni_Itelotabla__Pf_1_2021_1.docx",
    }

    found = courts_named(
        "Milyen ügyekben ítélkezett az Egri Törvényszék és a Debreceni Ítélőtábla?",
        files,
    )

    assert found == {"Egri_Torvenyszek", "Debreceni_Itelotabla"}


def test_satisfies_checks_only_the_constraints_the_question_states():
    egri = "Egri_Torvenyszek__P_1_2022_1.docx"

    assert satisfies(egri, "2022-05-01", {"Egri_Torvenyszek"}, [2021, 2022])
    assert not satisfies(egri, "2019-05-01", {"Egri_Torvenyszek"}, [2021, 2022])
    assert not satisfies(egri, "2022-05-01", {"Debreceni_Itelotabla"}, [2022])
    # no court and no year stated: everything qualifies
    assert satisfies(egri, None, set(), [])
    # a year is required but the document has no date: cannot be shown to qualify
    assert not satisfies(egri, None, set(), [2022])


def test_retrieval_trace_records_a_copy_per_stage_in_pipeline_order():
    trace = RetrievalTrace()
    chunks = [_chunk(1, "a.docx")]

    trace.record("vector", chunks)
    chunks.append(_chunk(2, "b.docx"))  # later mutation must not leak in
    trace.record("final", chunks)

    assert list(trace.stages) == ["vector", "final"]
    assert len(trace.stages["vector"]) == 1
    assert len(trace.stages["final"]) == 2


def test_verdict_flags_a_final_context_dominated_by_few_documents():
    text = verdict(
        final_docs=2, candidate_docs=17, golden_reached=1, golden_total=2, meeting=2
    )

    assert "1/2 golden document(s) reached the final context" in text
    assert "2 distinct document(s)" in text
    assert "17 documents were candidates" in text
    assert "dominated by a few documents" in text


def test_verdict_does_not_flag_dominance_when_few_documents_were_candidates():
    text = verdict(
        final_docs=3, candidate_docs=4, golden_reached=1, golden_total=1, meeting=3
    )

    assert "dominated" not in text


def test_verdict_says_so_when_nothing_reached_the_final_context():
    assert verdict(0, 5, 0, 2, 0) == "Nothing reached the final context."
