"""The `funnel` command: where does a document drop out of retrieval?

For each chosen golden question, runs the real retrieval path once with a
:class:`models.RetrievalTrace` attached and reports, stage by stage (vector,
full-text, fusion, rerank, final ``top_k``), how many chunks and distinct
documents were candidates, where each golden document ranks, and how many of
the candidate documents satisfy the constraints the question states (the
years it names, the court it names).

A golden document is only a *sampled* example for category questions, so the
constraint columns matter as much as the golden-document column: they show
whether the pipeline surfaced *valid* documents even when it missed the
sampled ones. Runs at production settings (the environment), so an A/B is
``RETRIEVAL_PERIOD_FILTER=true uv run python corpus/cli.py funnel -q q0010``.

The court is read from the file-name prefix (``Egri_Torvenyszek__...``) and
matched against the question text -- specific to this corpus, which is fine
for a diagnostic of this corpus's golden set, but not production logic.
"""

import re
import unicodedata
from typing import Annotated

import typer

from models import RetrievedChunk

app = typer.Typer()

#: Pipeline order; a stage a run did not reach (or that is switched off) is skipped.
STAGES = [
    "vector",
    "vector_years",
    "vector_csls",
    "fulltext",
    "fulltext_years",
    "identifier",
    "fused",
    "reranked",
    "listwise",
    "final",
]


#: What each stage is, in plain words (printed under the table).
STAGE_MEANING = {
    "vector": "the pool found by embedding similarity (what the question 'looks like')",
    "vector_years": "a second similarity pool limited to the years the question names",
    "vector_csls": "the similarity pool re-ordered to demote generic, boilerplate-like chunks",
    "fulltext": "the pool found by keyword search (stemmed Hungarian words)",
    "fulltext_years": "a second keyword pool limited to the question's years",
    "identifier": "chunks that contain a case number written in the question",
    "fused": "similarity + keyword pools merged (plus any identifier matches)",
    "reranked": "the merged pool re-scored by the reranker model",
    "listwise": "after the optional listwise LLM re-ordering (same as reranked when it is off)",
    "final": "the chunks the answering LLM actually sees (top_k)",
}

#: Why a stage may be absent from a run.
STAGE_NOT_RUN = {
    "vector_years": "RETRIEVAL_PERIOD_FILTER is off, or the question names no year",
    "fulltext_years": "RETRIEVAL_PERIOD_FILTER is off, or the question names no year",
    "identifier": "the question contains no case number",
}


def verdict(
    final_docs: int,
    candidate_docs: int,
    golden_reached: int,
    golden_total: int,
    meeting: int,
) -> str:
    """One plain sentence on what the funnel shows for a question.

    Args:
        final_docs: Distinct documents in the final context.
        candidate_docs: Distinct documents after the last candidate stage
            (the reranked pool).
        golden_reached: Golden documents present in the final context.
        golden_total: Golden documents the question cites.
        meeting: Final-context documents that satisfy the question's constraints.
    """
    if final_docs == 0:
        return "Nothing reached the final context."
    parts = []
    if golden_total:
        parts.append(
            f"{golden_reached}/{golden_total} golden document(s) reached the final context"
        )
    parts.append(
        f"the final context holds {final_docs} distinct document(s), "
        f"{meeting} of them meeting the question's constraints"
    )
    if candidate_docs >= 3 * final_docs and final_docs <= 3:
        parts.append(
            f"although {candidate_docs} documents were candidates -- the top_k chunks are "
            "dominated by a few documents, so a question about many cases sees only a few"
        )
    return "; ".join(parts) + "."


def print_legend() -> None:
    """Print how to read the table, once, after all questions."""
    print("\n" + "=" * 100)
    print("How to read this")
    print("=" * 100)
    print(
        """
Each row is one step of retrieval, in the order a question goes through them. The
same question is followed through all of them, so you can see WHERE a document that
ought to be found drops out.

  chunks        how many text pieces this step holds. A chunk is one piece of a document
                (~250 words); a document has ~20 of them, so many chunks can be few documents.
  docs          how many DIFFERENT documents those chunks come from.
  golden        the documents the golden question cites, in the order they are listed.
                '#3' = that document is the 3rd distinct document at this step; '-' = it is
                not there. For category questions ("which cases ...") the golden documents are
                only examples that were sampled, so also look at the last column.
  meeting       of the documents at this step, how many satisfy what the question itself asks
                for: the court it names and the years it names (read from the question).

Reading the end of each question:
  'reached the final context'      the golden document is among the chunks the LLM sees.
  'LOST after <step>'              it was a candidate up to <step> and then dropped out: that
                                   step (or the cut after it) is where to look.
  'never a candidate'              no step ever found it: a recall problem upstream.

Steps shown (in order):"""
    )
    for stage in STAGES:
        print(f"  {stage:<15} {STAGE_MEANING[stage]}")
    print(
        """
Settings are the environment's, so an A/B is e.g.
  RETRIEVAL_PERIOD_FILTER=true uv run python corpus/cli.py funnel -q q0010
The court is read from the file-name prefix and matched against the question text: this
is a diagnostic for this corpus's golden set, not part of the production pipeline."""
    )


def document_ranks(chunks: list[RetrievedChunk]) -> dict[str, int]:
    """Map each distinct document to its 1-based rank by first appearance in ``chunks``."""
    ranks: dict[str, int] = {}
    for chunk in chunks:
        ranks.setdefault(chunk.metadata.source_file, len(ranks) + 1)
    return ranks


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower().replace("_", " "))
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", stripped)


def court_of(source_file: str) -> str:
    """The court a document belongs to: the file-name prefix before ``__``."""
    return source_file.split("__")[0]


def courts_named(question: str, source_files: set[str]) -> set[str]:
    """Courts (file-name prefixes) whose name appears in ``question``."""
    normalized = _normalize(question)
    return {
        court
        for court in {court_of(f) for f in source_files}
        if _normalize(court) in normalized
    }


def satisfies(
    source_file: str,
    document_date: str | None,
    courts: set[str],
    years: list[int],
) -> bool:
    """Whether a document meets the question's court and year constraints.

    A constraint the question does not state is not checked, so a question
    with no court and no year is satisfied by every document.
    """
    if courts and court_of(source_file) not in courts:
        return False
    return not years or (document_date or "")[:4] in {str(y) for y in years}


def _fmt_ranks(ranks: dict[str, int], golden: list[str]) -> str:
    return (
        " ".join(f"#{ranks[g]}" if g in ranks else "-" for g in golden)
        or "(no golden docs)"
    )


@app.command(name="funnel")
def funnel(
    question: Annotated[
        list[str],
        typer.Option(
            "--question", "-q", help="Question id, e.g. -q q0010 (repeatable)."
        ),
    ],
    top_k: Annotated[
        int | None,
        typer.Option(help="Final chunks (default: RETRIEVAL_TOP_K)."),
    ] = None,
) -> None:
    """Show, per stage, where a question's golden / valid documents are (retrieval only, no LLM)."""
    from corpus.commands.eval import load_verified_questions, select_questions
    from drivers.embedding import get_embedding_driver
    from models import RetrievalTrace
    from query.retrieval import retrieve_chunks
    from query.time_filter import extract_years
    from store import VectorStore

    questions = select_questions(load_verified_questions(), question)
    driver = get_embedding_driver()
    with VectorStore() as store:
        dates = store.get_document_dates()
    all_files = set(dates)

    for q in questions:
        golden = [c["source_file"] for c in q.get("citations", [])]
        years = extract_years(q["question"])
        courts = courts_named(q["question"], all_files)
        ceiling = sum(satisfies(f, d, courts, years) for f, d in dates.items())

        trace = RetrievalTrace()
        retrieve_chunks(
            q["question"],
            top_k=top_k,
            query_vector=driver.embed_query(q["question"]),
            trace=trace,
        )

        print("\n" + "=" * 100)
        print(f"{q['id']} ({q['persona_id']}): {q['question']}")
        print(f"  golden documents : {', '.join(golden) or '-'}")
        print(
            f"  constraints read : years={years or '-'}, courts={sorted(courts) or '-'}"
        )
        print(
            f"  in the whole corpus: {ceiling} of {len(dates)} documents satisfy them"
        )
        if trace.notes.get("gate_passed") is False:
            print(
                f"  RELEVANCE GATE FAILED (top cosine {trace.notes.get('gate_top_score')}"
                f" < {trace.notes.get('gate_min_score')}): nothing is retrieved at all."
            )

        header = (
            f"  {'stage':<15}{'chunks':>7}{'docs':>6}  "
            f"{'golden (rank by doc)':<22}{'docs meeting constraints':>26}"
        )
        print("\n" + header)
        print("  " + "-" * (len(header) - 2))
        last_seen: dict[str, str] = {}
        for stage in STAGES:
            chunks = trace.stages.get(stage)
            if chunks is None:
                continue
            ranks = document_ranks(chunks)
            meeting = sum(satisfies(f, dates.get(f), courts, years) for f in ranks)
            for g in golden:
                if g in ranks:
                    last_seen[g] = stage
            share = f"{meeting}/{len(ranks)}" if ranks else "-"
            print(
                f"  {stage:<15}{len(chunks):>7}{len(ranks):>6}  "
                f"{_fmt_ranks(ranks, golden):<22}{share:>26}"
            )

        not_run = [
            f"{stage} ({STAGE_NOT_RUN[stage]})"
            for stage in STAGES
            if stage not in trace.stages and stage in STAGE_NOT_RUN
        ]
        if not_run:
            print("\n  not run: " + "; ".join(not_run))

        final_ranks = document_ranks(trace.stages.get("final", []))
        print()
        for g in golden:
            if g in final_ranks:
                print(f"  {g}: reached the final context.")
            elif g in last_seen:
                print(
                    f"  {g}: LOST after '{last_seen[g]}' (present there, absent from the final context)."
                )
            else:
                print(f"  {g}: never a candidate at any stage.")

        reranked = document_ranks(trace.stages.get("reranked", []))
        print(
            "\n  => "
            + verdict(
                final_docs=len(final_ranks),
                candidate_docs=len(reranked),
                golden_reached=sum(g in final_ranks for g in golden),
                golden_total=len(golden),
                meeting=sum(
                    satisfies(f, dates.get(f), courts, years) for f in final_ranks
                ),
            )
        )

    print_legend()
