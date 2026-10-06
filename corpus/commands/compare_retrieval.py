"""The `compare-retrieval` command: retrieval-only A/B of retrieval options.

Runs every (covered) verified golden question through the real retrieval path
once per *arm* in the same process and compares which of each question's cited
documents reach the final ``top_k`` context. The arms are:

    off        identifier guarantees as originally written
    on         + RETRIEVAL_DIVERSIFY_GUARANTEES (spread across documents)
    on+period  + RETRIEVAL_PERIOD_FILTER (also search the question's years)

No answer generation or grading, so it costs only embedding + rerank calls,
not LLM calls. All arms share one query embedding per question and run back to
back, so they see the same corpus state.
"""

from collections import defaultdict
from typing import Annotated

import typer

app = typer.Typer()


def _any_cited_hit(question: dict, retrieved_files: set[str]) -> bool:
    """Whether at least one cited document is in the retrieved context.

    A retrieval-only *proxy* for ``independent_fact`` grading, NOT the same
    thing: that strategy judges whichever documents the generated answer
    cites, which needs a real answer. This only shows how much the strict
    "every cited document" rule (``exact_match``) matters.
    """
    return any(c["source_file"] in retrieved_files for c in question["citations"])


def _doc_recall(question: dict, retrieved_files: set[str]) -> float:
    """Fraction of the question's cited documents present in the retrieved context."""
    cited = {c["source_file"] for c in question["citations"]}
    return len(cited & retrieved_files) / len(cited)


def _year_match(chunks: list, years: list[int]) -> float | None:
    """Share of ``chunks`` whose document_date year is one the question names.

    ``None`` when the question names no year (nothing to measure).
    """
    if not years or not chunks:
        return None
    wanted = {str(y) for y in years}
    in_period = sum(1 for c in chunks if (c.metadata.document_date or "")[:4] in wanted)
    return in_period / len(chunks)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.0%}"


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
    top_k: Annotated[
        int | None,
        typer.Option(help="Final chunks per question (default: RETRIEVAL_TOP_K)."),
    ] = None,
) -> None:
    """A/B: retrieval options (see module docstring), retrieval only."""
    from corpus.commands.coverage import _ingested_source_files, _is_fully_covered
    from corpus.commands.eval import check_retrieval_hit, load_verified_questions
    from drivers.embedding import get_embedding_driver
    from query.retrieval import HybridRetrievalStrategy, retrieve_chunks
    from query.time_filter import extract_years

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
        "off": HybridRetrievalStrategy(diversify_guarantees=False, period_filter=False),
        "on": HybridRetrievalStrategy(diversify_guarantees=True, period_filter=False),
        "on+period": HybridRetrievalStrategy(
            diversify_guarantees=True, period_filter=True
        ),
    }
    names = list(arms)
    print(
        f"Comparing {len(questions)} question(s), retrieval only, "
        f"top_k={top_k if top_k is not None else 'default'} ...\n"
    )

    # persona -> arm -> list of per-question values
    all_hit: dict = defaultdict(lambda: defaultdict(list))
    any_hit: dict = defaultdict(lambda: defaultdict(list))
    recall: dict = defaultdict(lambda: defaultdict(list))
    in_period: dict = defaultdict(lambda: defaultdict(list))

    print(f"{'question':<8}{'persona':<22}" + "".join(f"{n:>11}" for n in names))
    for q in questions:
        vector = driver.embed_query(q["question"])
        years = extract_years(q["question"])
        cells = []
        for name, strategy in arms.items():
            chunks = retrieve_chunks(
                q["question"], top_k=top_k, strategy=strategy, query_vector=vector
            )
            files = {c.metadata.source_file for c in chunks}
            hit = check_retrieval_hit(q, files)
            all_hit[q["persona_id"]][name].append(hit)
            any_hit[q["persona_id"]][name].append(_any_cited_hit(q, files))
            recall[q["persona_id"]][name].append(_doc_recall(q, files))
            match = _year_match(chunks, years)
            if match is not None:
                in_period[q["persona_id"]][name].append(match)
            cells.append(
                ("HIT" if hit else "miss")
                + (f"/{match:.0%}" if match is not None else "")
            )
        differs = len(set(cells)) > 1
        print(
            f"{q['id']:<8}{q['persona_id']:<22}"
            + "".join(f"{c:>11}" for c in cells)
            + ("  <-differs" if differs else "")
        )

    for title, data, as_rate in (
        ("all (exact_match yardstick)", all_hit, True),
        ("any (loose independent_fact proxy)", any_hit, True),
        ("document recall", recall, False),
        (
            "in-period (final top_k chunks inside the question's years)",
            in_period,
            False,
        ),
    ):
        print(f"\n{title}")
        print(f"{'persona':<22}{'n':>4}" + "".join(f"{n:>11}" for n in names))
        for persona_id in sorted(data):
            counts = len(data[persona_id][names[0]])
            cells = [
                _pct(_mean([float(v) for v in data[persona_id][n]])) for n in names
            ]
            print(f"{persona_id:<22}{counts:>4}" + "".join(f"{c:>11}" for c in cells))

    print(
        """
How to read this (retrieval only, no LLM answer is generated; arms run back to
back on the same corpus state with the same query embedding):
  off / on / on+period   RETRIEVAL_DIVERSIFY_GUARANTEES off / on, and on plus
                         RETRIEVAL_PERIOD_FILTER.
  per-question cell      HIT = every cited document is in the final top_k chunks;
                         '/NN%' = share of those chunks inside the question's years
                         (only shown when the question names a year).
  all / any / recall     see the table titles; 'any' is only a proxy for
                         independent_fact, which needs a real answer (use `eval`).
  in-period              how well the retrieved chunks respect the years asked about.
  A row marked <-differs changed between arms."""
    )
