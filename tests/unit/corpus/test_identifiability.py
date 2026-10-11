"""Telling an identifiable description from an ambiguous one."""

from corpus.commands.mark_identifiable import mark
from corpus.identifiability import document_ranking, gold_position
from models import ChunkMetadata, RetrievedChunk


def chunks(*sources):
    return [
        RetrievedChunk(i, "text", ChunkMetadata(source, None, 0), 0.0)
        for i, source in enumerate(sources)
    ]


def test_a_document_with_several_good_chunks_beats_one_lucky_chunk():
    ranking = document_ranking(chunks("a", "b", "b", "b", "a"))

    assert ranking[0] == "b"


def test_the_golden_documents_position_and_its_absence():
    ranked = chunks("a", "b", "c")

    assert gold_position(ranked, "a") == 0
    assert gold_position(ranked, "c") == 2
    assert gold_position(ranked, "z") is None


def question(qid, persona="practical_procedural", status="verified"):
    return {
        "id": qid,
        "persona_id": persona,
        "verification_status": status,
        "citations": [{"source_file": f"{qid}.docx"}],
    }


def test_a_practical_question_whose_document_is_not_first_becomes_ambiguous():
    questions = [question("first"), question("second"), question("absent")]
    positions = {"first": 0, "second": 1, "absent": None}

    counts = mark(questions, lambda q: positions[q["id"]])

    assert counts == (1, 2)
    first, second, absent = questions
    assert (
        first["verification_status"] == "verified"
        and first["gold_document_position"] == 0
    )
    assert second["verification_status"] == "ambiguous"
    assert "ranks 2. among" in second["verification_notes"]
    assert absent["verification_status"] == "ambiguous"
    assert "not among the candidates" in absent["verification_notes"]


def test_only_unlooked_at_verified_practical_questions_are_touched():
    done = question("done") | {"gold_document_position": 3}
    others = [
        question("ff", persona="fact_finder"),
        question("old", status="needs_review"),
        done,
    ]
    asked = []

    counts = mark(others, lambda q: asked.append(q["id"]) or 0)

    assert counts == (0, 0) and asked == []
    assert done["gold_document_position"] == 3
