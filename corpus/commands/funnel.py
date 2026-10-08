"""The `funnel` command: what does each step of retrieval take in and give out?

For each chosen golden question, runs the real retrieval once (the step-based
pipeline, whose steps report what they held going in and coming out) and prints, step
by step: how many chunks and distinct documents the step received, how many it
passed on, where each golden document ranks afterwards, how many of the candidate
documents satisfy the constraints the question states (the years it names, the court
it names), and which step **dropped** a golden document that an earlier step had
found. A step that refuses (a gate) says so.

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
from dataclasses import dataclass
from typing import Annotated

import typer

from models import RetrievedChunk
from query.runner import StageRecord

app = typer.Typer()

#: What each step is, in plain words (printed under the table).
STEP_MEANING = {
    "embed_query": "turns the question into a vector",
    "dense_search": "the pool found by embedding similarity (what the question 'looks like')",
    "relevance_gate": "refuses the whole question when nothing is similar enough (cosine)",
    "year_dense_widening": "adds a second similarity pool limited to the years the question names",
    "csls_reorder": "re-orders the similarity pool to demote generic, boilerplate-like chunks",
    "keyword_search": "the pool found by keyword search (stemmed Hungarian words)",
    "year_keyword_widening": "adds a second keyword pool limited to the question's years",
    "rrf_fusion": "merges the similarity and keyword pools into one ranked list",
    "rerank": "re-scores the merged list with the reranker model",
    "rerank_score_gate": "drops what the reranker scored too low (refuses if nothing is left)",
    "listwise_rerank": "the optional listwise LLM re-ordering",
    "top_k_selection": "the final cut: the chunks the answering LLM actually sees",
    "cosine_cut": "the plain similarity cut of the vector profile",
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


@dataclass(frozen=True)
class StepRow:
    """What one step took in and gave out, for the table.

    Attributes:
        step: The step's name.
        chunks_in: Chunks it received (``None`` for a step that starts from nothing).
        chunks_out: Chunks it passed on (empty if it refused; ``None`` for a step that
            handles no chunks, such as embedding the question).
        ranks: Each document of the output, by rank of first appearance.
        dropped: Golden documents the step received and did not pass on.
        notes: What the step reported about itself (a gate's scores, side pools).
        seconds: How long it took.
        refused: Why it refused the question, if it did.
    """

    step: str
    chunks_in: tuple[RetrievedChunk, ...] | None
    chunks_out: tuple[RetrievedChunk, ...] | None
    ranks: dict[str, int]
    dropped: tuple[str, ...]
    notes: tuple[str, ...]
    seconds: float
    refused: str | None


def _unique(chunks) -> tuple[RetrievedChunk, ...]:
    """The chunks without repeats (by id), in first-seen order."""
    seen: set[int] = set()
    out = []
    for chunk in chunks:
        if chunk.id not in seen:
            seen.add(chunk.id)
            out.append(chunk)
    return tuple(out)


def step_rows(records: list[StageRecord], golden: list[str]) -> list[StepRow]:
    """One :class:`StepRow` per step that ran.

    What a step "received" is the slot it reads *and* writes (a reorder or a filter),
    or, for a step that builds a new list from others (a fusion, a final cut), all the
    lists it reads, once each. A gate that passes hands on what it received.

    Args:
        records: The records of the steps that ran, in order.
        golden: The source files of the golden documents.
    """
    rows = []
    for record in records:
        refused = record.declined is not None
        inputs = {slot: tuple(chunks) for slot, chunks in record.inputs.items()}
        if record.outputs:
            slot = next(iter(record.outputs))
            out = tuple(record.outputs[slot])
            received = (
                inputs[slot]
                if slot in inputs
                else (
                    _unique(c for v in inputs.values() for c in v) if inputs else None
                )
            )
        else:  # a gate: it passes on what it received, or nothing when it refuses
            received = (
                _unique(c for v in inputs.values() for c in v) if inputs else None
            )
            out = () if refused else (received or ())
        docs_in = {c.metadata.source_file for c in (received or ())}
        docs_out = {c.metadata.source_file for c in out}
        handles_chunks = bool(record.outputs or inputs or refused)
        dropped = (
            tuple(g for g in golden if g in docs_in and g not in docs_out)
            if received is not None
            else ()
        )
        notes = [
            f"{label}: {len(chunks)} chunk(s)" for label, chunks in record.aux.items()
        ]
        facts = [
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in record.notes.items()
            if k != "years"
        ]
        if facts:
            notes.append(", ".join(facts))
        if "years" in record.notes:
            notes.append(f"years read from the question: {record.notes['years']}")
        rows.append(
            StepRow(
                step=record.step,
                chunks_in=received,
                chunks_out=out if handles_chunks else None,
                ranks=document_ranks(list(out)),
                dropped=dropped,
                notes=tuple(notes),
                seconds=record.seconds,
                refused=(
                    f"{record.declined.reason} at {record.declined.stage}"
                    if record.declined
                    else None
                ),
            )
        )
    return rows


def _size(chunks) -> str:
    """``chunks/documents``, or ``-`` for a step that received nothing."""
    if chunks is None:
        return "-"
    docs = {c.metadata.source_file for c in chunks}
    return f"{len(chunks)}/{len(docs)}"


def print_legend() -> None:
    """Print how to read the table, once, after all questions."""
    print("\n" + "=" * 100)
    print("How to read this")
    print("=" * 100)
    print(
        """
Each row is one step of retrieval, in the order a question goes through them. The
same question is followed through all of them, so you can see WHICH step drops a
document that ought to be found.

  in, out       chunks/documents the step received and passed on. A chunk is one piece
                of a document (~250 words); a document has ~20 of them, so many chunks
                can be few documents. '-' = the step starts from nothing (a search).
  golden        the documents the golden question cites, in the order they are listed,
                as they stand in the step's OUTPUT: '#3' = the 3rd distinct document;
                '-' = not there. For category questions ("which cases ...") the golden
                documents are only examples that were sampled, so also look at the
                last column.
  meeting       of the documents in the output, how many satisfy what the question
                itself asks for: the court it names and the years it names (read from
                the question).
  seconds       how long the step took.

Below the table: what each step reported about itself (a gate's scores, side pools), and
'DROPPED by <step>' for a golden document the step received and did not pass on.

Steps (in order of a run; a step that is switched off, or has nothing to do, is not run):"""
    )
    for step, meaning in STEP_MEANING.items():
        print(f"  {step:<22} {meaning}")
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
    """Show, per step, what it took in and gave out, and where the golden documents are (retrieval only, no LLM)."""
    from config import settings
    from corpus.commands.eval import load_verified_questions, select_questions
    from drivers.embedding import get_embedding_driver
    from query.composition import build_retrieval_service
    from query.outcome import Answerable
    from query.service import RetrievalRequest
    from query.time_filter import extract_years
    from store import VectorStore

    questions = select_questions(load_verified_questions(), question)
    driver = get_embedding_driver()
    service = build_retrieval_service(settings)
    with VectorStore() as store:
        dates = store.get_document_dates()
    all_files = set(dates)

    for q in questions:
        golden = [c["source_file"] for c in q.get("citations", [])]
        years = extract_years(q["question"])
        courts = courts_named(q["question"], all_files)
        ceiling = sum(satisfies(f, d, courts, years) for f, d in dates.items())

        result = service.retrieve(
            RetrievalRequest(
                q["question"],
                profile=settings.RETRIEVAL_STRATEGY,
                query_vector=driver.embed_query(q["question"]),
                top_k=top_k,
            ),
            VectorStore(),
        )
        rows = step_rows(list(result.records), golden)

        print("\n" + "=" * 100)
        print(f"{q['id']} ({q['persona_id']}): {q['question']}")
        print(f"  golden documents : {', '.join(golden) or '-'}")
        print(
            f"  constraints read : years={years or '-'}, courts={sorted(courts) or '-'}"
        )
        print(
            f"  in the whole corpus: {ceiling} of {len(dates)} documents satisfy them"
        )
        if not isinstance(result.outcome, Answerable):
            d = result.outcome
            print(f"  REFUSED by '{d.stage}' ({d.reason}): nothing reaches the answer.")

        header = (
            f"  {'step':<22}{'in':>9}{'out':>9}  "
            f"{'golden (rank by doc)':<22}{'meeting':>9}{'seconds':>9}"
        )
        print("\n" + header)
        print("  " + "-" * (len(header) - 2))
        for row in rows:
            meeting = sum(satisfies(f, dates.get(f), courts, years) for f in row.ranks)
            share = f"{meeting}/{len(row.ranks)}" if row.ranks else "-"
            flag = "  REFUSED" if row.refused else ""
            print(
                f"  {row.step:<22}{_size(row.chunks_in):>9}{_size(row.chunks_out):>9}  "
                f"{_fmt_ranks(row.ranks, golden):<22}{share:>9}{row.seconds:>9.2f}{flag}"
            )

        for row in rows:
            for note in row.notes:
                print(f"    {row.step}: {note}")
        print()
        dropped_by: dict[str, str] = {}
        for row in rows:
            for g in row.dropped:
                dropped_by.setdefault(g, row.step)
                print(f"  DROPPED by {row.step}: {g}")

        final = (
            list(result.outcome.chunks)
            if isinstance(result.outcome, Answerable)
            else []
        )
        final_ranks = document_ranks(final)
        found = {g for row in rows for g in golden if g in row.ranks}
        for g in golden:
            if g in final_ranks:
                print(f"  {g}: reached the final context.")
            elif g in found:
                print(
                    f"  {g}: LOST at '{dropped_by.get(g, '?')}' (found earlier, absent from the final context)."
                )
            else:
                print(f"  {g}: never a candidate at any step.")

        candidates = next(
            (
                r.chunks_in
                for r in reversed(rows)
                if r.step in ("top_k_selection", "cosine_cut") and r.chunks_in
            ),
            (),
        )
        print(
            "\n  => "
            + verdict(
                final_docs=len(final_ranks),
                candidate_docs=len(document_ranks(list(candidates))),
                golden_reached=sum(g in final_ranks for g in golden),
                golden_total=len(golden),
                meeting=sum(
                    satisfies(f, dates.get(f), courts, years) for f in final_ranks
                ),
            )
        )

    print_legend()
