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


# ---- the per-step view -------------------------------------------------------------


def _step_chunk(chunk_id: int, source: str = "a.docx") -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        content="",
        metadata=ChunkMetadata(source_file=source, page_number=None, chunk_index=0),
        score=1.0,
    )


def _record(step, inputs=None, outputs=None, aux=None, notes=None, declined=None):
    from query.runner import StageRecord

    return StageRecord(
        step=step,
        inputs=inputs or {},
        outputs=outputs or {},
        aux=aux or {},
        notes=notes or {},
        seconds=0.5,
        declined=declined,
    )


class TestStepRows:
    def test_a_search_starts_from_nothing_and_a_filter_shows_what_it_received(self):
        from corpus.commands.funnel import step_rows
        from query.context import Slot

        pool = (
            _step_chunk(1, "a.docx"),
            _step_chunk(2, "b.docx"),
            _step_chunk(3, "b.docx"),
        )
        rows = step_rows(
            [
                _record("dense_search", outputs={Slot.DENSE_POOL: pool}),
                _record(
                    "csls_reorder",
                    inputs={Slot.DENSE_POOL: pool},
                    outputs={Slot.DENSE_POOL: pool[:2]},
                ),
            ],
            golden=[],
        )

        search, reorder = rows
        assert search.chunks_in is None
        assert search.chunks_out is not None and len(search.chunks_out) == 3
        assert search.ranks == {
            "a.docx": 1,
            "b.docx": 2,
        }
        assert reorder.chunks_in is not None and len(reorder.chunks_in) == 3
        assert reorder.chunks_out is not None and len(reorder.chunks_out) == 2

    def test_a_fusion_receives_both_pools_once_each(self):
        from corpus.commands.funnel import step_rows
        from query.context import Slot

        dense = (_step_chunk(1), _step_chunk(2))
        keyword = (_step_chunk(2), _step_chunk(3))  # chunk 2 is in both

        (row,) = step_rows(
            [
                _record(
                    "rrf_fusion",
                    inputs={Slot.DENSE_POOL: dense, Slot.KEYWORD_POOL: keyword},
                    outputs={
                        Slot.RANKED: (_step_chunk(2), _step_chunk(1), _step_chunk(3))
                    },
                )
            ],
            golden=[],
        )

        assert row.chunks_in is not None and len(row.chunks_in) == 3  # not 4

    def test_the_step_that_loses_a_golden_document_is_named(self):
        from corpus.commands.funnel import step_rows
        from query.context import Slot

        ranked = (_step_chunk(1, "gold.docx"), _step_chunk(2, "other.docx"))

        rows = step_rows(
            [
                _record(
                    "rerank",
                    inputs={Slot.RANKED: ranked},
                    outputs={Slot.RANKED: ranked},
                ),
                _record(
                    "rerank_score_gate",
                    inputs={Slot.RANKED: ranked},
                    outputs={Slot.RANKED: ranked[1:]},
                ),
            ],
            golden=["gold.docx"],
        )

        assert rows[0].dropped == ()
        assert rows[1].dropped == ("gold.docx",)

    def test_a_step_that_starts_from_nothing_cannot_drop_anything(self):
        from corpus.commands.funnel import step_rows
        from query.context import Slot

        (row,) = step_rows(
            [_record("dense_search", outputs={Slot.DENSE_POOL: (_step_chunk(1),)})],
            golden=["gold.docx"],
        )

        assert row.dropped == ()

    def test_a_gate_that_passes_hands_on_what_it_received(self):
        from corpus.commands.funnel import step_rows
        from query.context import Slot

        pool = (_step_chunk(1), _step_chunk(2))

        (row,) = step_rows(
            [_record("relevance_gate", inputs={Slot.DENSE_POOL: pool})], golden=[]
        )

        assert row.chunks_out == pool and row.refused is None

    def test_a_refusing_gate_gives_out_nothing_and_drops_every_golden_document(self):
        from corpus.commands.funnel import step_rows
        from query.context import Slot
        from query.outcome import Declined, DeclineReason

        pool = (_step_chunk(1, "gold.docx"),)
        refusal = Declined(DeclineReason.NOT_RELEVANT, stage="relevance_gate")

        (row,) = step_rows(
            [
                _record(
                    "relevance_gate",
                    inputs={Slot.DENSE_POOL: pool},
                    declined=refusal,
                )
            ],
            golden=["gold.docx"],
        )

        assert row.chunks_out == ()
        assert row.refused == "not_relevant at relevance_gate"
        assert row.dropped == ("gold.docx",)

    def test_side_pools_and_notes_are_reported_and_years_are_spelled_out(self):
        from corpus.commands.funnel import step_rows
        from query.context import Slot

        (row,) = step_rows(
            [
                _record(
                    "year_dense_widening",
                    inputs={Slot.DENSE_POOL: (_step_chunk(1),)},
                    outputs={Slot.DENSE_POOL: (_step_chunk(1), _step_chunk(2))},
                    aux={"year_pool": (_step_chunk(2),)},
                    notes={"years": [2021]},
                )
            ],
            golden=[],
        )

        assert "year_pool: 1 chunk(s)" in row.notes
        assert "years read from the question: [2021]" in row.notes

    def test_a_step_that_handles_no_chunks_shows_no_size_and_notes_are_one_line(self):
        from corpus.commands.funnel import step_rows

        (embed,) = step_rows([_record("embed_query")], golden=[])
        (gate,) = step_rows(
            [
                _record(
                    "relevance_gate",
                    notes={"gate_passed": True, "gate_top_score": 0.83367161},
                )
            ],
            golden=[],
        )

        assert embed.chunks_out is None
        assert gate.notes == ("gate_passed=True, gate_top_score=0.834",)
