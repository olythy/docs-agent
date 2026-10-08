"""The `retrieval-snapshot` command: pin what retrieval returns, to prove a refactor changed nothing.

Runs the real retrieval service (at the current settings) for the golden questions and
records, per question, the ids every step left in each chunk list it produced
(``<step>.<list>``, plus a step's side results) and the final context.
``--out`` writes that to a JSON file; ``--compare`` runs again and lists every difference
against such a file (exit status 1 if there is any). The point is a restructuring of the
retrieval code: take a snapshot first, change the code in small steps, compare after every
step, and the difference must stay empty. A *deliberate* behaviour change shows up here as
exactly the questions it was meant to change.

It covers retrieval only (no router, no answer LLM), so it is cheap: an embedding and a rerank
call per question. It records the settings that shape retrieval and refuses to call two
snapshots comparable when those differ.

Corpus specific (it reads this corpus's golden set), hence under ``corpus/``.
"""

import json
from pathlib import Path
from typing import Annotated, Any

import typer

app = typer.Typer()

#: The settings that decide what retrieval returns; a comparison across a change of any
#: of them is not a comparison of code.
_SETTINGS = (
    "RETRIEVAL_STRATEGY",
    "RETRIEVAL_TOP_K",
    "RETRIEVAL_CANDIDATE_POOL_SIZE",
    "RETRIEVAL_MIN_SCORE",
    "RETRIEVAL_PERIOD_FILTER",
    "RETRIEVAL_DIVERSIFY_GUARANTEES",
    "RERANKER_DRIVER",
    "RERANKER_MODEL",
    "RERANKER_MIN_SCORE",
    "EMBEDDING_DRIVER",
    "EMBEDDING_MODEL",
    "LISTWISE_RERANK_ENABLED",
)


def diff_snapshots(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """List every difference between two snapshots, in words (empty when identical).

    Args:
        old: The earlier snapshot (``{"settings": ..., "questions": {id: {...}}}``).
        new: The later one.
    """
    problems: list[str] = []
    for key in sorted(set(old["settings"]) | set(new["settings"])):
        if old["settings"].get(key) != new["settings"].get(key):
            problems.append(
                f"settings differ ({key}: {old['settings'].get(key)!r} -> "
                f"{new['settings'].get(key)!r}): not comparable"
            )
    for qid in sorted(set(old["questions"]) | set(new["questions"])):
        before, after = old["questions"].get(qid), new["questions"].get(qid)
        if before is None or after is None:
            problems.append(
                f"{qid}: only in the {'new' if before is None else 'old'} snapshot"
            )
            continue
        for stage in sorted(set(before["stages"]) | set(after["stages"])):
            if before["stages"].get(stage) != after["stages"].get(stage):
                problems.append(
                    f"{qid} stage {stage!r}: {before['stages'].get(stage)} -> "
                    f"{after['stages'].get(stage)}"
                )
        if before["final"] != after["final"]:
            problems.append(f"{qid} final: {before['final']} -> {after['final']}")
        if before["gate_passed"] != after["gate_passed"]:
            problems.append(
                f"{qid} relevance gate: {before['gate_passed']} -> {after['gate_passed']}"
            )
    return problems


def stage_ids(records) -> dict[str, list[int]]:
    """The chunk ids every step left, by ``<step>.<list>`` (side results by ``<step>.<label>``).

    Args:
        records: The step records of one retrieval (:class:`query.runner.StageRecord`).
    """
    stages: dict[str, list[int]] = {}
    for record in records:
        for slot, chunks in record.outputs.items():
            stages[f"{record.step}.{slot}"] = [c.id for c in chunks]
        for label, chunks in record.aux.items():
            stages[f"{record.step}.{label}"] = [c.id for c in chunks]
    return stages


def take_snapshot(questions: list[dict]) -> dict[str, Any]:
    """Run retrieval for ``questions`` and record what each step held."""
    from config import settings
    from drivers.embedding import get_embedding_driver
    from query.composition import build_retrieval_service
    from query.outcome import Answerable
    from query.service import RetrievalRequest
    from store import VectorStore

    driver = get_embedding_driver()
    service = build_retrieval_service(settings)
    recorded: dict[str, Any] = {}
    for q in questions:
        result = service.retrieve(
            RetrievalRequest(
                q["question"],
                profile=settings.RETRIEVAL_STRATEGY,
                query_vector=driver.embed_query(q["question"]),
            ),
            VectorStore(),
        )
        recorded[q["id"]] = {
            "question": q["question"],
            "stages": stage_ids(result.records),
            "final": (
                [c.id for c in result.outcome.chunks]
                if isinstance(result.outcome, Answerable)
                else []
            ),
            "gate_passed": next(
                (
                    r.notes["gate_passed"]
                    for r in result.records
                    if "gate_passed" in r.notes
                ),
                None,
            ),
        }
    return {
        "settings": {name: getattr(settings, name) for name in _SETTINGS},
        "questions": recorded,
    }


@app.command(name="retrieval-snapshot")
def retrieval_snapshot(
    out: Annotated[
        Path | None, typer.Option("--out", help="Write the snapshot to this file.")
    ] = None,
    compare: Annotated[
        Path | None,
        typer.Option(
            "--compare",
            help="Compare a fresh run with this snapshot (exit 1 on a difference).",
        ),
    ] = None,
    question: Annotated[
        list[str] | None,
        typer.Option(
            "--question", "-q", help="Only these question ids (default: all verified)."
        ),
    ] = None,
) -> None:
    """Snapshot (or compare) what retrieval returns for the golden questions."""
    from corpus.commands.eval import load_verified_questions, select_questions

    questions = select_questions(load_verified_questions(), question)
    fresh = take_snapshot(questions)
    print(f"retrieval run for {len(questions)} question(s).")
    if out is not None:
        out.write_text(
            json.dumps(fresh, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"snapshot written to {out}")
    if compare is not None:
        problems = diff_snapshots(
            json.loads(compare.read_text(encoding="utf-8")), fresh
        )
        if not problems:
            print(f"IDENTICAL to {compare}: nothing changed.")
        else:
            print(f"{len(problems)} DIFFERENCE(S) against {compare}:")
            for line in problems[:60]:
                print(f"  {line}")
            raise typer.Exit(1)
