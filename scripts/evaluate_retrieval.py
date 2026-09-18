"""Retrieval-quality evaluation: pure-vector vs. hybrid+rerank.

Purpose:
    Answers "how do you measure quality?" with our own, repeatable numbers
    instead of an ad-hoc manual check. Loads a small, hand-written question
    set (``tests/data/eval_questions.json``) against the two real, committed
    fixtures (``tests/data/sample.md``/``sample.pdf``), and compares two
    retrieval configurations:

        - "vector-only": ``query.retrieval.VectorRetrievalStrategy`` — the
          pre-hybrid-search behavior.
        - "hybrid+rerank": ``query.retrieval.HybridRetrievalStrategy``
          (vector + keyword search fused with RRF, then whichever
          ``RERANKER_DRIVER`` is configured — ``none`` by default).

    Both run through the exact same ``query.retrieval.retrieve_chunks()``
    entry point production code uses (with an explicit ``strategy=``
    override for each comparison), not a hand-rolled stand-in — so this
    never drifts out of sync with what ``query_knowledge_base()`` actually
    does. Each question is embedded exactly once up front
    (``_precompute_embeddings``) and passed to both strategies via
    ``retrieve_chunks()``'s ``query_vector=`` override — without this,
    every one of the 26 (13 questions x 2 strategies) calls would create
    its own embedding driver and re-trigger its lazy model load, 26 times
    over for what's really just 13 unique embeddings.

    Measures, per configuration:
        - Recall@k: for each answerable question, did a chunk from the
          expected source file appear anywhere in the returned top-k?
        - MRR (Mean Reciprocal Rank): how high up was the first chunk from
          the expected source file?
        - Fallback rate: for deliberately unanswerable questions, did
          retrieval correctly return nothing (the signal
          ``query.retrieval.query_knowledge_base`` uses to return
          ``NO_RESULTS_MESSAGE`` instead of asking the LLM to guess)? This
          only measures the *retrieval-layer* gate (``RETRIEVAL_MIN_SCORE``
          on cosine similarity) — it is NOT the whole safety story. See
          ``print_comparison_table``'s printed note for why a low number
          here doesn't mean the system hallucinates: the LLM prompt has
          its own, separate instruction to admit when the given context
          doesn't actually answer the question. Pass ``--with-llm`` to
          actually measure that second layer too (see below).

    On a small corpus the two configurations can easily land on identical
    aggregate numbers by coincidence, which hides whether they actually
    behave differently per question — ``print_per_question_breakdown``
    shows each question's result side by side specifically to catch that.

    Honesty note: at minimum this corpus is two documents and ~13
    questions — the actual corpus is whatever's already in document_chunks
    plus these two (see "Which database?" below), so Recall@k/MRR will
    vary with it. Either way, these are numbers for comparing our own
    configurations against each other, not a statistically meaningful
    benchmark.

    Which database?
        Deliberately **not** gated on AGENT_ENV=test, and safe to run
        against a populated dev database: earlier versions of this script
        truncated document_chunks first for a clean slate, which is a
        destructive operation — the same class of accident that once ran
        against the real Supabase DATABASE_URL in this project (because
        AGENT_ENV wasn't checked first). This version only *adds* the two
        fixtures if they're not already present
        (``VectorStore.has_chunks_from_source``) and never deletes
        anything, so there's nothing destructive left to gate. Prints
        which database it's about to touch either way, for visibility.
        Run against ``AGENT_ENV=test`` instead when you want a fully
        controlled, repeatable comparison (no other documents mixed in).

    ``--with-llm``:
        Also generates a real answer via ``LLM_DRIVER`` for every
        question, for both configurations — makes real network calls
        (13 questions x 2 configs = 26 LLM calls), skipped by default to
        stay fast and free to run. For the 3 deliberately unanswerable
        questions, checks whether the answer looks like a decline (a
        loose heuristic — see ``_looks_like_a_decline``) and reports a
        decline rate; this is the second defense layer's own measurement,
        complementing Fallback rate above (which only measures the first,
        retrieval-layer one). The 10 answerable questions' answers are
        printed for manual review, not auto-scored — judging real answer
        *correctness* would need an LLM-as-judge or similar, out of scope
        here.

Usage:
    uv run python scripts/evaluate_retrieval.py
    uv run python scripts/evaluate_retrieval.py --with-llm
    AGENT_ENV=test uv run python scripts/evaluate_retrieval.py  # controlled corpus
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from urllib.parse import urlsplit

# ---------------------------------------------------------------------------
# Ensure the project root is importable (needed for running as a script)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from drivers.embedding import get_embedding_driver
from ingestion.ingest import add_document
from query.retrieval import (
    NO_RESULTS_MESSAGE,
    HybridRetrievalStrategy,
    VectorRetrievalStrategy,
    retrieve_chunks,
)
from scripts.format_utils import truncate, wrap
from store import VectorStore

EVAL_QUESTIONS_PATH = PROJECT_ROOT / "tests" / "data" / "eval_questions.json"
FIXTURE_DOCS = [
    PROJECT_ROOT / "tests" / "data" / "sample.md",
    PROJECT_ROOT / "tests" / "data" / "sample.pdf",
]

# All console output in this script is sized to stay under 80 columns —
# the conservative, universally-safe terminal width — even for the widest
# row (the per-question breakdown table's "<- differs" marker included).
QUESTION_COLUMN_WIDTH = 40
STATUS_COLUMN_WIDTH = 10
LLM_QUESTION_PREVIEW_WIDTH = 45
LLM_ANSWER_PREVIEW_WIDTH = 65


def _print_target_database() -> None:
    """Print which database this run will read from (and add fixtures to).

    Never prints credentials — just enough of DATABASE_URL to recognize
    which database this is. This script no longer deletes anything (see
    the module docstring), so this is transparency, not a safety gate.
    """
    parsed = urlsplit(settings.DATABASE_URL)
    print(
        f"[eval] Target database: {parsed.hostname}:{parsed.port}{parsed.path} "
        f"(AGENT_ENV={settings.AGENT_ENV})"
    )


def _ensure_fixtures_seeded() -> None:
    """Add each fixture document only if it isn't already in the database.

    Idempotent and non-destructive: safe to run repeatedly, and safe to
    run against a database that already has real, unrelated documents in
    it — nothing is ever removed or duplicated.
    """
    store = VectorStore()
    for doc_path in FIXTURE_DOCS:
        if store.has_chunks_from_source(doc_path.name):
            print(f"[eval] {doc_path.name} already present — skipping re-ingest.")
        else:
            add_document(doc_path)


def _precompute_embeddings(questions: list[dict]) -> dict[str, list[float]]:
    """Embed every question once, so both strategies can reuse the same vector.

    Without this, ``retrieve_chunks()`` creates its own
    ``EmbeddingDriver`` (and re-triggers its lazy model load) on every one
    of its calls — 13 questions x 2 strategies would otherwise mean 26
    redundant model loads for what's really just 13 unique embeddings.
    Passing the result back into ``retrieve_chunks(..., query_vector=...)``
    skips both the embedding call and the model load on every call after
    this one.
    """
    driver = get_embedding_driver()
    print(f"[eval] Pre-embedding {len(questions)} question(s) ...")
    return {q["question"]: driver.embed_text(q["question"]) for q in questions}


def _rank_of_expected(chunks: list[dict], expected_source_file: str) -> int | None:
    """Return the 1-based rank of the first chunk from ``expected_source_file``, or None."""
    for rank, chunk in enumerate(chunks, start=1):
        if chunk["metadata"].get("source_file") == expected_source_file:
            return rank
    return None


def evaluate(config_name: str, retrieve_fn, questions: list[dict]) -> dict:
    """Run every eval question through ``retrieve_fn`` and compute metrics.

    Args:
        config_name: Label for the results table.
        retrieve_fn: Callable taking a question string and returning a
            list of chunk dicts.
        questions: Parsed ``eval_questions.json`` entries.

    Returns:
        A dict with ``config``, ``recall_at_k``, ``mrr``,
        ``fallback_accuracy``, ``n_answerable``, ``n_unanswerable``, and
        ``details`` — a per-question list of ``{"question",
        "expected_source_file", "rank", "n_chunks", "chunks"}`` dicts,
        used by :func:`print_per_question_breakdown` and (with
        ``--with-llm``) :func:`print_llm_answers`.
    """
    recalls = []
    reciprocal_ranks = []
    fallback_correct = 0
    fallback_total = 0
    details = []

    for q in questions:
        chunks = retrieve_fn(q["question"])
        expected = q["expected_source_file"]
        rank = _rank_of_expected(chunks, expected) if expected else None

        if expected is None:
            fallback_total += 1
            if not chunks:
                fallback_correct += 1
        else:
            recalls.append(rank is not None)
            reciprocal_ranks.append(1 / rank if rank else 0.0)

        details.append(
            {
                "question": q["question"],
                "expected_source_file": expected,
                "rank": rank,
                "n_chunks": len(chunks),
                "chunks": chunks,
            }
        )

    return {
        "config": config_name,
        "recall_at_k": sum(recalls) / len(recalls) if recalls else 0.0,
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "fallback_accuracy": fallback_correct / fallback_total if fallback_total else 0.0,
        "n_answerable": len(recalls),
        "n_unanswerable": fallback_total,
        "details": details,
    }


def print_comparison_table(results: list[dict]) -> None:
    """Print a plain-text comparison table across configurations."""
    print()
    print(f"{'Config':<28} {'Recall@k':>10} {'MRR':>8} {'Fallback rate':>15}")
    print("-" * 63)
    for r in results:
        print(
            f"{r['config']:<28} {r['recall_at_k']:>10.2f} {r['mrr']:>8.2f} "
            f"{r['fallback_accuracy']:>15.2f}"
        )
    print()
    print(
        f"(n_answerable={results[0]['n_answerable']}, "
        f"n_unanswerable={results[0]['n_unanswerable']})"
    )
    print(
        "\n"
        + wrap(
            "Fallback rate measures the RETRIEVAL-layer gate only "
            "(RETRIEVAL_MIN_SCORE on cosine similarity). A low number here can "
            "be a real, expected limitation, not a bug: a question that stays "
            "on-topic (e.g. asks about Player Central) but asks for a fact the "
            "document never states (e.g. a phone number) can still score highly "
            "on pure topical similarity — the embedding can't tell 'about this "
            "topic' apart from 'answers this exact question'. This is why the "
            "system has a second, independent safety net: the LLM prompt "
            "(drivers/llm.py) explicitly instructs the model to admit when the "
            "given context doesn't actually answer the question, rather than "
            "guessing. Pass --with-llm to actually measure that second layer "
            "too, with real (network-calling) answers."
        )
    )


def _status_label(entry: dict) -> str:
    """One-line status for a single question's retrieval result.

    Answerable questions: ``hit@N`` (found at rank N) or ``MISS``.
    Unanswerable questions: ``rejected`` (correctly returned nothing) or
    ``kept (N)`` (N chunks slipped through the relevance gate).
    """
    if entry["expected_source_file"] is None:
        return "rejected" if entry["n_chunks"] == 0 else f"kept ({entry['n_chunks']})"
    return f"hit@{entry['rank']}" if entry["rank"] else "MISS"


def print_per_question_breakdown(vector_result: dict, hybrid_result: dict) -> None:
    """Print one row per question, showing where vector and hybrid agree or differ.

    The aggregate metrics in :func:`print_comparison_table` can land on
    identical numbers by coincidence on a small corpus, even when the two
    strategies actually behave differently per question — this shows that
    directly instead of hiding it behind an average.
    """
    print("\nPer-question breakdown (vector vs. hybrid):\n")
    header = (
        f"  {'Question':<{QUESTION_COLUMN_WIDTH}} "
        f"{'Vector':<{STATUS_COLUMN_WIDTH}} {'Hybrid':<{STATUS_COLUMN_WIDTH}}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    differences = 0
    rows = zip(vector_result["details"], hybrid_result["details"], strict=True)
    for vector_entry, hybrid_entry in rows:
        vector_label = _status_label(vector_entry)
        hybrid_label = _status_label(hybrid_entry)
        differs = vector_label != hybrid_label
        differences += differs
        marker = " <-differs" if differs else ""
        question_text = truncate(vector_entry["question"], QUESTION_COLUMN_WIDTH)
        print(
            f"  {question_text:<{QUESTION_COLUMN_WIDTH}} {vector_label:<{STATUS_COLUMN_WIDTH}} "
            f"{hybrid_label:<{STATUS_COLUMN_WIDTH}}{marker}"
        )

    total = len(vector_result["details"])
    print(f"\n{differences} of {total} question(s) got a different result between strategies.")


def _answer_with_llm(question: str, chunks: list[dict]) -> str:
    """Generate the final answer for ``chunks``, exactly like
    ``query.retrieval.query_knowledge_base()`` does internally — but taking
    already-retrieved ``chunks`` directly, so this works for either
    strategy's results without needing a strategy override on
    ``query_knowledge_base()`` itself.
    """
    if not chunks:
        return NO_RESULTS_MESSAGE

    from drivers.llm import get_answer_driver

    return get_answer_driver().answer(question=question, context_chunks=chunks)


def _looks_like_a_decline(answer: str) -> bool:
    """Heuristic: does ``answer`` look like the model declined to answer?

    A loose, case-insensitive substring check rather than an exact match
    against either NO_RESULTS_MESSAGE or drivers/llm.py's prompt-specified
    refusal sentence — those are two different strings on purpose (one is
    the retrieval-layer fallback, the other is what the prompt *asks* the
    model to say), and a free/weaker model won't always follow "respond
    with exactly ..." to the letter anyway, so a loose match is more
    realistic than an exact one.
    """
    return "could not find" in answer.lower()


def print_llm_answers(vector_result: dict, hybrid_result: dict) -> None:
    """Generate and print a real LLM answer for every question, per strategy.

    Makes one real network call per question per config (26 total for the
    current 13-question set) — only runs when ``--with-llm`` is passed.
    """
    for config_name, result in [
        (vector_result["config"], vector_result),
        (hybrid_result["config"], hybrid_result),
    ]:
        print(f"\n--- {config_name} ---")
        declined_correctly = 0
        unanswerable_total = 0

        for entry in result["details"]:
            answer = _answer_with_llm(entry["question"], entry["chunks"])
            question_preview = truncate(entry["question"], LLM_QUESTION_PREVIEW_WIDTH)
            answer_preview = truncate(answer, LLM_ANSWER_PREVIEW_WIDTH)

            if entry["expected_source_file"] is None:
                unanswerable_total += 1
                declined = _looks_like_a_decline(answer)
                declined_correctly += declined
                tag = "DECLINED" if declined else "DID NOT DECLINE"
            else:
                tag = "answer"
            print(f"  [{tag:<15}] {question_preview}\n      -> {answer_preview}")

        if unanswerable_total:
            print(
                f"\n  LLM decline rate ({config_name}): {declined_correctly}/"
                f"{unanswerable_total} unanswerable question(s) correctly declined."
            )


def main() -> None:
    # retrieve_chunks()/add_document() log their progress via `logging`, not
    # print() — configure a bare, print()-like handler so this script's
    # output stays exactly as before (logging is silent by default).
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Retrieval-quality evaluation.")
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help=(
            "Also generate real LLM answers for every question (both "
            "configs) and report the decline rate for unanswerable "
            "questions. Makes real network calls via LLM_DRIVER — off by "
            "default to stay fast and free to run."
        ),
    )
    args = parser.parse_args()

    _print_target_database()
    _ensure_fixtures_seeded()

    questions = json.loads(EVAL_QUESTIONS_PATH.read_text())
    print(f"[eval] Loaded {len(questions)} eval question(s).")

    embeddings = _precompute_embeddings(questions)

    print("\n[eval] Running vector-only baseline ...")
    vector_only_results = evaluate(
        "vector-only",
        lambda q: retrieve_chunks(
            q, strategy=VectorRetrievalStrategy(), query_vector=embeddings[q]
        ),
        questions,
    )

    print(f"\n[eval] Running hybrid+rerank (RERANKER_DRIVER={settings.RERANKER_DRIVER}) ...")
    hybrid_results = evaluate(
        f"hybrid+rerank ({settings.RERANKER_DRIVER})",
        lambda q: retrieve_chunks(
            q, strategy=HybridRetrievalStrategy(), query_vector=embeddings[q]
        ),
        questions,
    )

    print_comparison_table([vector_only_results, hybrid_results])
    print_per_question_breakdown(vector_only_results, hybrid_results)

    if args.with_llm:
        print(f"\n[eval] --with-llm: generating real answers via LLM_DRIVER={settings.LLM_DRIVER} ...")
        print_llm_answers(vector_only_results, hybrid_results)

    print(
        "\n"
        + wrap(
            "Note: a tiny, hand-written question set (13 questions), and the "
            "corpus is whatever's in document_chunks right now (at minimum the "
            "2 fixtures this script seeds — see the module docstring's "
            "'Which database?' section). These are our own, repeatable numbers "
            "for comparing configurations against each other, not a "
            "statistically significant benchmark."
        )
    )


if __name__ == "__main__":
    main()
