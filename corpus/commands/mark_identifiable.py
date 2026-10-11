"""The `mark-identifiable` command: keep practical questions the system can identify.

For every verified practical question that has not been looked at, runs the real retrieval
and records where the golden document stands among the candidates (see
``corpus/identifiability.py``). A question whose document is not the first is set to
``ambiguous``: its description fits other decisions as well, so it is left out of the
evaluation (which reads only ``verified`` questions) but kept in the file.

Usage::

    uv run python corpus/cli.py mark-identifiable [--questions-file F]
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer()

QUESTIONS = Path(__file__).resolve().parent.parent / "data" / "questions_v2.json"

#: Takes a question text and returns the golden document's position among the candidates
#: (0 is first) or ``None`` when it is not among them.
Positioner = Callable[[dict], int | None]


def mark(questions: list[dict], position_of: Positioner) -> tuple[int, int]:
    """Record the golden document's position on each verified practical question.

    Args:
        questions: The golden questions; changed in place.
        position_of: Says where a question's golden document stands.

    Returns:
        ``(identifiable, ambiguous)`` counts of the questions looked at now.
    """
    identifiable = ambiguous = 0
    for question in questions:
        if (
            question["persona_id"] != "practical_procedural"
            or question["verification_status"] != "verified"
            or "gold_document_position" in question
        ):
            continue
        position = position_of(question)
        question["gold_document_position"] = position
        if position == 0:
            identifiable += 1
            continue
        ambiguous += 1
        question["verification_status"] = "ambiguous"
        question["verification_notes"] = (
            "The description fits other decisions at least as well: the golden document "
            + (
                "is not among the candidates."
                if position is None
                else f"ranks {position + 1}. among the candidate documents."
            )
        )
    return identifiable, ambiguous


def retrieval_position(question: dict) -> int | None:
    """The golden document's position among the reranked candidates of the real retrieval."""
    from corpus.commands.eval import _answer
    from corpus.identifiability import gold_position
    from query.retrieval.step import Slot

    result = _answer(question["question"], None)
    retrieval = result.explain.retrieval
    if retrieval is None:
        return None
    records = {record.step: record for record in retrieval.records}
    rerank = records.get("rerank")
    ranked = list(rerank.outputs.get(Slot.RANKED, ())) if rerank else []
    return gold_position(ranked, question["citations"][0]["source_file"])


@app.command("mark-identifiable")
def mark_identifiable(
    questions_file: Annotated[
        Path, typer.Option(help="The questions file (default: questions_v2.json).")
    ] = QUESTIONS,
) -> None:
    """Set practical questions whose document is not the first candidate to ambiguous."""
    questions = json.loads(questions_file.read_text(encoding="utf-8"))
    identifiable, ambiguous = mark(questions, retrieval_position)
    questions_file.write_text(
        json.dumps(questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{identifiable} identifiable, {ambiguous} set to ambiguous.")
