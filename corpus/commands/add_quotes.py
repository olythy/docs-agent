"""The `add-quotes` command: give every golden citation a verified supporting quote.

A golden citation names the document; this adds the word-for-word passage of the decision's
own text that supports the expected answer (see ``corpus/golden_quotes.py``), so the
evaluation can tell whether a system retrieved or cited the right *section*, whatever the
chunking of the moment. The model proposes the passage and the code accepts it only if it
is really in the document; run it again and only citations without a quote are touched.

Usage::

    uv run python corpus/cli.py add-quotes [--persona ID] [-q ID ...]
"""

import json
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer()

QUESTIONS = Path(__file__).resolve().parent.parent / "data" / "questions.json"


@app.command("add-quotes")
def add_quotes_command(
    persona: Annotated[str | None, typer.Option(help="Only this persona.")] = None,
    question_id: Annotated[
        list[str] | None, typer.Option("-q", help="Only this question id (repeatable).")
    ] = None,
    questions_file: Annotated[
        Path,
        typer.Option(help="The questions file (default: corpus/data/questions.json)."),
    ] = QUESTIONS,
) -> None:
    """Add a verified supporting quote to every golden citation that has none."""
    from corpus.golden_quotes import add_quotes, llm_ask
    from corpus.verification import fetch_chunk_bodies

    questions = json.loads(questions_file.read_text(encoding="utf-8"))
    wanted = set(question_id or [])

    def only(question: dict) -> bool:
        return (persona is None or question["persona_id"] == persona) and (
            not wanted or question["id"] in wanted
        )

    report = add_quotes(questions, fetch_chunk_bodies, llm_ask, only)
    questions_file.write_text(
        json.dumps(questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"{len(report.added)} quote(s) added, {report.skipped} already there, "
        f"{len(report.failed)} left without one."
    )
    for qid, source, why in report.failed:
        print(f"  no quote for {qid} / {source}: {why}")
