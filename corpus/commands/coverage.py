"""The `coverage` command: how much of the golden set is answerable right now.

Useful specifically during a growing/partial ingest (see docs/decisions.md):
running a full `eval` against an incomplete corpus produces misleading
low numbers that are really "the target document isn't ingested yet," not
a retrieval or answer-quality problem. This answers a narrower, cheaper
question first -- for each persona, what fraction of its verified
questions have *every* cited source_file already present in
document_chunks -- so it's possible to tell "not enough ingested yet" apart
from "worth running the real eval now" without a full eval pass.
"""

from pathlib import Path

import typer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

app = typer.Typer()


def _ingested_source_files() -> set[str]:
    """Return every distinct source_file currently in document_chunks."""
    import sys

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from store import VectorStore

    return VectorStore().get_all_source_files()


def _is_fully_covered(question: dict, ingested: set[str]) -> bool:
    """Whether every citation in ``question`` is already ingested.

    A question with no citations (e.g. adversarial, which never cites a
    real document on purpose) is vacuously covered -- it doesn't depend on
    corpus size at all.
    """
    citations = question.get("citations", [])
    return all(c["source_file"] in ingested for c in citations)


@app.command(name="coverage")
def coverage() -> None:
    """Report, per persona, what fraction of verified golden questions are fully ingested."""
    from corpus.commands.eval import load_verified_questions

    questions = load_verified_questions()
    if not questions:
        print("No verified questions found. Run `generate-questions` first.")
        raise typer.Exit(code=1)

    ingested = _ingested_source_files()
    print(f"{len(ingested)} distinct source_file(s) currently ingested.\n")

    by_persona: dict[str, list[bool]] = {}
    for q in questions:
        by_persona.setdefault(q["persona_id"], []).append(
            _is_fully_covered(q, ingested)
        )

    header = f"{'persona':<22}{'covered':>9}  {'total':>5}  {'pct':>6}"
    print(header)
    print("-" * len(header))

    total_covered = total_all = 0
    for persona_id in sorted(by_persona):
        flags = by_persona[persona_id]
        covered = sum(flags)
        total = len(flags)
        total_covered += covered
        total_all += total
        pct = f"{covered / total:.0%}" if total else "-"
        print(f"{persona_id:<22}{covered:>9}  {total:>5}  {pct:>6}")

    print("-" * len(header))
    overall_pct = f"{total_covered / total_all:.0%}" if total_all else "-"
    print(f"{'TOTAL':<22}{total_covered:>9}  {total_all:>5}  {overall_pct:>6}")

    print(
        "\ncovered = questions whose every cited document is already ingested "
        "(only those are\nfair to evaluate, see `eval --only-covered`); total = verified "
        "golden questions."
    )
    _print_hub_score_coverage()


def _print_hub_score_coverage() -> None:
    """Print how many chunks have a hub_score, in red if any are missing.

    CSLS re-ranking (see ``query.retrieval.steps.candidates.CslsReorderStep``) needs ``compute-hub-scores``
    to have run over the *current* corpus; an eval run against chunks
    without a score silently measures a pipeline with CSLS (partly) off.
    """
    from store import VectorStore

    scored, total = VectorStore().count_hub_scored_chunks()
    pct = f"{scored / total:.1%}" if total else "-"
    line = f"\nhub_score: {scored}/{total} chunks scored ({pct})"
    if total and scored < total:
        typer.secho(
            f"{line} -- WARNING: run `corpus/cli.py compute-hub-scores` before "
            "trusting eval results (CSLS is partly or fully off).",
            fg=typer.colors.RED,
            bold=True,
        )
    else:
        typer.secho(line, fg=typer.colors.GREEN)
