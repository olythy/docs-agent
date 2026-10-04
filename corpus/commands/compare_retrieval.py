"""The `compare-retrieval` command: retrieval-only A/B of identifier-guarantee diversification.

Runs every (covered) verified golden question through the real retrieval path
twice in the same process -- once with ``diversify_guarantees=False`` (the
current behaviour), once with ``True`` -- and compares which of each
question's cited documents reach the final ``top_k`` context. No answer
generation or grading, so it costs only embedding + rerank calls, not LLM
calls. Both arms share one query embedding per question and run back to
back, so they see the same corpus state even while an ingest is still
growing it.
"""

from collections import defaultdict
from typing import Annotated

import typer

app = typer.Typer()


def _doc_recall(question: dict, retrieved_files: set[str]) -> float:
    """Fraction of the question's cited documents present in the retrieved context."""
    cited = {c["source_file"] for c in question["citations"]}
    return len(cited & retrieved_files) / len(cited)


@app.command(name="compare-retrieval")
def compare_retrieval(
    persona: Annotated[
        str | None,
        typer.Option(help="Only compare this persona_id (default: all)."),
    ] = None,
    only_covered: Annotated[
        bool,
        typer.Option(help="Skip questions whose cited documents aren't ingested yet."),
    ] = True,
) -> None:
    """A/B: identifier-guarantee diversification off vs. on, retrieval only."""
    from corpus.commands.coverage import _ingested_source_files, _is_fully_covered
    from corpus.commands.eval import check_retrieval_hit, load_verified_questions
    from drivers.embedding import get_embedding_driver
    from query.retrieval import HybridRetrievalStrategy, retrieve_chunks

    questions = [
        q
        for q in load_verified_questions(persona_filter=persona)
        if q.get("citations")  # adversarial questions cite nothing to compare
    ]
    if only_covered:
        ingested = _ingested_source_files()
        questions = [q for q in questions if _is_fully_covered(q, ingested)]
    if not questions:
        print("No questions to compare.")
        raise typer.Exit(code=1)

    driver = get_embedding_driver()
    arms = {
        "off": HybridRetrievalStrategy(diversify_guarantees=False),
        "on": HybridRetrievalStrategy(diversify_guarantees=True),
    }
    print(f"Comparing {len(questions)} question(s), retrieval only ...\n")

    hits: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    recalls: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    print(f"{'question':<8}{'persona':<22}{'off':>8}{'on':>8}  recall off -> on")
    for q in questions:
        vector = driver.embed_query(q["question"])
        row = {}
        for name, strategy in arms.items():
            chunks = retrieve_chunks(
                q["question"], strategy=strategy, query_vector=vector
            )
            files = {c.metadata.source_file for c in chunks}
            row[name] = (check_retrieval_hit(q, files), _doc_recall(q, files))
            hits[q["persona_id"]][name].append(row[name][0])
            recalls[q["persona_id"]][name].append(row[name][1])
        marker = "  <-differs" if row["off"][0] != row["on"][0] else ""
        print(
            f"{q['id']:<8}{q['persona_id']:<22}"
            f"{'HIT' if row['off'][0] else 'miss':>8}{'HIT' if row['on'][0] else 'miss':>8}"
            f"  {row['off'][1]:.2f} -> {row['on'][1]:.2f}{marker}"
        )

    print(
        f"\n{'persona':<22}{'n':>4}  {'hit off':>8}{'hit on':>8}  {'recall off':>11}{'recall on':>10}"
    )
    for persona_id in sorted(hits):
        n = len(hits[persona_id]["off"])
        print(
            f"{persona_id:<22}{n:>4}  "
            f"{sum(hits[persona_id]['off']) / n:>8.0%}{sum(hits[persona_id]['on']) / n:>8.0%}  "
            f"{sum(recalls[persona_id]['off']) / n:>11.2f}{sum(recalls[persona_id]['on']) / n:>10.2f}"
        )
