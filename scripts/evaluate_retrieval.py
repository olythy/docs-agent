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
        question, comparing both configurations side-by-side. For
        answerable questions, checks whether the expected gold fact
        is included in the completion. For deliberately unanswerable
        questions, checks whether retrieval filtered them out (0 chunks)
        or the prompt safely declined, versus potential hallucinations.
        Prints a consolidated LLM Benchmark Scorecard at the end with
        gold fact retention %, hallucination resistance %, API call counts,
        and latency. Makes real network calls via LLM_DRIVER (skipped by
        default to stay fast and free to run).

Usage:
    uv run python scripts/evaluate_retrieval.py
    uv run python scripts/evaluate_retrieval.py --with-llm
    AGENT_ENV=test uv run python scripts/evaluate_retrieval.py  # controlled corpus
"""

import argparse
import json
import logging
import sys
import textwrap
import time
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
    PROJECT_ROOT / "tests" / "data" / "sample_hu.md",
]

# All console output in this script is sized to stay under 80 columns —
# the conservative, universally-safe terminal width — even for the widest
# row (the per-question breakdown table's "<- differs" marker included).
QUESTION_COLUMN_WIDTH = 42
STATUS_COLUMN_WIDTH = 11


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
    of its calls — redundant model loads for each strategy.
    """
    driver = get_embedding_driver()
    print(f"[eval] Pre-embedding {len(questions)} question(s) ...")
    return {q["question"]: driver.embed_query(q["question"]) for q in questions}


def _match_ranks(chunks: list[dict], q: dict) -> tuple[int | None, int | None]:
    """Return (file_rank, passage_rank) for a question against retrieved chunks.

    file_rank is the 1-based rank of the first chunk from expected_source_file.
    passage_rank is the 1-based rank of the first chunk containing expected_text_contains.
    """
    expected_file = q.get("expected_source_file")
    expected_text = q.get("expected_text_contains")

    if not expected_file:
        return None, None

    file_rank: int | None = None
    passage_rank: int | None = None

    for rank, chunk in enumerate(chunks, start=1):
        if chunk["metadata"].get("source_file") == expected_file:
            if file_rank is None:
                file_rank = rank
            if expected_text and expected_text.lower() in chunk["content"].lower():
                if passage_rank is None:
                    passage_rank = rank
                    break
            elif not expected_text and passage_rank is None:
                passage_rank = rank
                break

    return file_rank, passage_rank


def evaluate(config_name: str, retrieve_fn, questions: list[dict]) -> dict:
    """Run every eval question through ``retrieve_fn`` and compute metrics.

    Args:
        config_name: Label for the results table.
        retrieve_fn: Callable taking a question string and returning a
            list of chunk dicts.
        questions: Parsed ``eval_questions.json`` entries.

    Returns:
        A dict with comprehensive evaluation metrics, including passage-level
        Hit@1, Recall@k, MRR, Fallback rate, language-specific breakdowns, and
        per-question details.
    """
    passage_hits_at_1 = []
    passage_recalls = []
    passage_mrr_list = []
    fallback_correct = 0
    fallback_total = 0

    lang_stats: dict[str, dict] = {}
    details = []

    for q in questions:
        lang = q.get("language", "en")
        if lang not in lang_stats:
            lang_stats[lang] = {
                "hits_at_1": [],
                "recalls": [],
                "mrr": [],
                "fallback_correct": 0,
                "fallback_total": 0,
            }

        chunks = retrieve_fn(q["question"])
        expected_file = q.get("expected_source_file")
        file_rank, passage_rank = _match_ranks(chunks, q)

        if expected_file is None:
            fallback_total += 1
            lang_stats[lang]["fallback_total"] += 1
            if not chunks:
                fallback_correct += 1
                lang_stats[lang]["fallback_correct"] += 1
        else:
            hit_1 = passage_rank == 1
            recall_k = passage_rank is not None
            mrr_val = 1 / passage_rank if passage_rank else 0.0

            passage_hits_at_1.append(hit_1)
            passage_recalls.append(recall_k)
            passage_mrr_list.append(mrr_val)

            lang_stats[lang]["hits_at_1"].append(hit_1)
            lang_stats[lang]["recalls"].append(recall_k)
            lang_stats[lang]["mrr"].append(mrr_val)

        details.append(
            {
                "question": q["question"],
                "expected_source_file": expected_file,
                "expected_text_contains": q.get("expected_text_contains"),
                "file_rank": file_rank,
                "passage_rank": passage_rank,
                "language": lang,
                "n_chunks": len(chunks),
                "chunks": chunks,
            }
        )

    def _mean(lst: list) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "config": config_name,
        "passage_hit_at_1": _mean(passage_hits_at_1),
        "passage_recall_at_k": _mean(passage_recalls),
        "passage_mrr": _mean(passage_mrr_list),
        "fallback_accuracy": fallback_correct / fallback_total
        if fallback_total
        else 0.0,
        "n_answerable": len(passage_recalls),
        "n_unanswerable": fallback_total,
        "lang_breakdown": {
            lang: {
                "hit@1": _mean(stats["hits_at_1"]),
                "recall@k": _mean(stats["recalls"]),
                "mrr": _mean(stats["mrr"]),
                "fallback": (
                    stats["fallback_correct"] / stats["fallback_total"]
                    if stats["fallback_total"]
                    else 0.0
                ),
                "n_ans": len(stats["hits_at_1"]),
                "n_unans": stats["fallback_total"],
            }
            for lang, stats in lang_stats.items()
        },
        "details": details,
    }


def print_comparison_table(results: list[dict]) -> None:
    """Print a plain-text comparison table across configurations."""
    print()
    print(
        f"{'Config':<30} {'Hit@1':>8} {'Recall@k':>10} {'Passage MRR':>13} {'Fallback':>10}"
    )
    print("-" * 75)
    for r in results:
        print(
            f"{r['config']:<30} {r['passage_hit_at_1']:>8.2f} {r['passage_recall_at_k']:>10.2f} "
            f"{r['passage_mrr']:>13.2f} {r['fallback_accuracy']:>10.2f}"
        )
    print()
    print(
        f"(Total questions: {results[0]['n_answerable']} answerable, "
        f"{results[0]['n_unanswerable']} unanswerable)"
    )

    print("\nLanguage Breakdown (Passage Hit@1 / Recall@k / Fallback):")
    for r in results:
        print(f"  [{r['config']}]")
        for lang, stats in r["lang_breakdown"].items():
            lang_label = "Hungarian (HU)" if lang == "hu" else "English (EN)"
            print(
                f"    {lang_label:<16} Hit@1: {stats['hit@1']:.2f} | "
                f"Recall@k: {stats['recall@k']:.2f} | "
                f"MRR: {stats['mrr']:.2f} | "
                f"Fallback: {stats['fallback']:.2f} "
                f"(ans={stats['n_ans']}, unans={stats['n_unans']})"
            )


def _status_label(entry: dict) -> str:
    """One-line status for a single question's retrieval result."""
    if entry["expected_source_file"] is None:
        return "rejected" if entry["n_chunks"] == 0 else f"kept ({entry['n_chunks']})"
    if entry["passage_rank"] == 1:
        return "HIT@1"
    if entry["passage_rank"] is not None:
        return f"hit@{entry['passage_rank']}"
    if entry["file_rank"] is not None:
        return f"file@{entry['file_rank']}"
    return "MISS"


def print_per_question_breakdown(vector_result: dict, hybrid_result: dict) -> None:
    """Print one row per question, showing where vector and hybrid agree or differ."""
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
    print(
        f"\n{differences} of {total} question(s) got a different result between strategies."
    )


def _format_answer(text: str, indent: str = "  ", width: int = 76) -> str:
    """Format and word-wrap an LLM answer, preserving paragraphs.

    Args:
        text: Raw answer string from the LLM.
        indent: Prefix prepended to every line of output.
        width: Maximum terminal width for wrapped lines.

    Returns:
        Reflowed multi-line string.
    """
    paragraphs = text.split("\n")
    formatted = []
    wrap_width = max(width - len(indent), 20)
    for p in paragraphs:
        stripped = p.strip()
        if not stripped:
            formatted.append("")
        else:
            lines = textwrap.wrap(stripped, width=wrap_width)
            formatted.append("\n".join(f"{indent}{line}" for line in lines))
    return "\n".join(formatted)


def _looks_like_a_decline(answer: str) -> bool:
    """Heuristic: does ``answer`` look like the model declined to answer?

    Checks for common English and Hungarian decline and refusal phrases.
    """
    lower = answer.lower()
    decline_phrases = [
        "could not find",
        "does not mention",
        "cannot find",
        "not mentioned",
        "not provided",
        "no information",
        "unable to find",
        "nem találtam",
        "nem található",
        "nem tartalmaz",
        "nem szerepel",
        "nincs információ",
        "nem tér ki",
        "nem derül ki",
        "nem állapítható meg",
    ]
    return any(p in lower for p in decline_phrases)


def _print_llm_scorecard(v_stats: dict, h_stats: dict) -> None:
    """Print a comparative summary scorecard of LLM generation performance."""
    print("\n" + "=" * 80)
    print("                    LLM GENERATION BENCHMARK SCORECARD")
    print("=" * 80)

    v_name = truncate(v_stats["config"], 17)
    h_name = truncate(h_stats["config"], 17)
    print(f"{'Metric':<42} {v_name:>17} {h_name:>17}")
    print("-" * 80)

    ans_tot = v_stats["ans_total"]
    print(f"Answerable Questions ({ans_tot}):")

    def _fmt_rate(count: int, total: int) -> str:
        if total == 0:
            return "0/0  (  0.0%)"
        pct = (count / total) * 100
        return f"{count:>2}/{total:<2} ({pct:>5.1f}%)"

    v_fact = _fmt_rate(v_stats["gold_matches"], ans_tot)
    h_fact = _fmt_rate(h_stats["gold_matches"], ans_tot)
    print(f"{'  - Gold Fact Inclusion Rate':<42} {v_fact:>17} {h_fact:>17}")

    v_decl = _fmt_rate(v_stats["ans_declined"], ans_tot)
    h_decl = _fmt_rate(h_stats["ans_declined"], ans_tot)
    print(f"{'  - Declined / Unanswered':<42} {v_decl:>17} {h_decl:>17}")

    unans_tot = v_stats["unans_total"]
    print(f"\nUnanswerable / Hallucination Gate ({unans_tot}):")

    v_rej = _fmt_rate(v_stats["unans_retrieval_rejected"], unans_tot)
    h_rej = _fmt_rate(h_stats["unans_retrieval_rejected"], unans_tot)
    print(f"{'  - Filtered at Retrieval (0 chunks)':<42} {v_rej:>17} {h_rej:>17}")

    v_pdecl = _fmt_rate(v_stats["unans_prompt_declined"], unans_tot)
    h_pdecl = _fmt_rate(h_stats["unans_prompt_declined"], unans_tot)
    print(f"{'  - Safely Declined by Prompt':<42} {v_pdecl:>17} {h_pdecl:>17}")

    v_safe_tot = v_stats["unans_retrieval_rejected"] + v_stats["unans_prompt_declined"]
    h_safe_tot = h_stats["unans_retrieval_rejected"] + h_stats["unans_prompt_declined"]
    v_safe = _fmt_rate(v_safe_tot, unans_tot)
    h_safe = _fmt_rate(h_safe_tot, unans_tot)
    print(f"{'  - Total Safe Decline Rate':<42} {v_safe:>17} {h_safe:>17}")

    v_hall = _fmt_rate(v_stats["unans_hallucinations"], unans_tot)
    h_hall = _fmt_rate(h_stats["unans_hallucinations"], unans_tot)
    print(f"{'  - Potential Hallucinations':<42} {v_hall:>17} {h_hall:>17}")

    print("\nEfficiency & Latency:")
    print(
        f"{'  - Real LLM API Calls Made':<42} {v_stats['api_calls']:>17} {h_stats['api_calls']:>17}"
    )

    v_lat = (
        f"{sum(v_stats['latencies']) / len(v_stats['latencies']):.2f}s"
        if v_stats["latencies"]
        else "N/A"
    )
    h_lat = (
        f"{sum(h_stats['latencies']) / len(h_stats['latencies']):.2f}s"
        if h_stats["latencies"]
        else "N/A"
    )
    print(f"{'  - Average LLM Latency':<42} {v_lat:>17} {h_lat:>17}")
    print("=" * 80)


def print_llm_answers(vector_result: dict, hybrid_result: dict) -> None:
    """Generate and compare real LLM answers side-by-side for every question.

    Evaluates gold fact inclusion for answerable questions, and checks
    retrieval filtering / prompt decline behavior for unanswerable questions.
    Prints an LLM Generation Benchmark Scorecard upon completion.
    """
    from drivers.llm import get_answer_driver

    driver = get_answer_driver()

    stats = {
        "vec": {
            "config": vector_result["config"],
            "ans_total": 0,
            "gold_matches": 0,
            "ans_declined": 0,
            "unans_total": 0,
            "unans_retrieval_rejected": 0,
            "unans_prompt_declined": 0,
            "unans_hallucinations": 0,
            "api_calls": 0,
            "latencies": [],
        },
        "hyb": {
            "config": hybrid_result["config"],
            "ans_total": 0,
            "gold_matches": 0,
            "ans_declined": 0,
            "unans_total": 0,
            "unans_retrieval_rejected": 0,
            "unans_prompt_declined": 0,
            "unans_hallucinations": 0,
            "api_calls": 0,
            "latencies": [],
        },
    }

    v_details = vector_result["details"]
    h_details = hybrid_result["details"]
    total = len(v_details)

    for i in range(total):
        v_entry = v_details[i]
        h_entry = h_details[i]

        q_text = v_entry["question"]
        expected_file = v_entry["expected_source_file"]
        expected_fact = v_entry.get("expected_text_contains")
        lang = v_entry.get("language", "en")
        is_unans = expected_file is None

        print("\n" + "=" * 80)
        print(f"[{i + 1}/{total}] ({lang.upper()}) {q_text}")
        if is_unans:
            print("Expected: [DELIBERATELY UNANSWERABLE — EXPECTED TO DECLINE]")
        else:
            fact_str = f"'{expected_fact}'" if expected_fact else "(none specified)"
            print(f"Expected: {fact_str} in {expected_file}")
        print("-" * 80)

        for key, entry in [("vec", v_entry), ("hyb", h_entry)]:
            cfg_name = stats[key]["config"]
            s = stats[key]
            chunks = entry["chunks"]

            if not chunks:
                answer = NO_RESULTS_MESSAGE
                latency = 0.0
                if is_unans:
                    s["unans_total"] += 1
                    s["unans_retrieval_rejected"] += 1
                    tag = "🛡️  RETRIEVAL REJECTED"
                else:
                    s["ans_total"] += 1
                    s["ans_declined"] += 1
                    tag = "⚠️  NO CHUNKS RETRIEVED"
            else:
                t0 = time.perf_counter()
                answer = driver.answer(question=q_text, context_chunks=chunks)
                latency = time.perf_counter() - t0
                s["api_calls"] += 1
                s["latencies"].append(latency)

                is_decline = _looks_like_a_decline(answer)
                if is_unans:
                    s["unans_total"] += 1
                    if is_decline:
                        s["unans_prompt_declined"] += 1
                        tag = "🛡️  PROMPT DECLINED"
                    else:
                        s["unans_hallucinations"] += 1
                        tag = "🚨 POTENTIAL HALLUCINATION"
                else:
                    s["ans_total"] += 1
                    has_fact = bool(
                        expected_fact and expected_fact.lower() in answer.lower()
                    )
                    if has_fact:
                        s["gold_matches"] += 1
                        tag = "✅ GOLD FACT MATCH"
                    elif is_decline:
                        s["ans_declined"] += 1
                        tag = "⚠️  DECLINED / REFUSED"
                    else:
                        tag = "ℹ️  ANSWERED (FACT NOT FOUND)"

            print(f"[{cfg_name}] ({len(chunks)} chunks, {latency:.2f}s) [{tag}]")
            print(_format_answer(answer, indent="  ", width=76))
            print()

    _print_llm_scorecard(stats["vec"], stats["hyb"])


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

    print(
        f"\n[eval] Running hybrid+rerank (RERANKER_DRIVER={settings.RERANKER_DRIVER}) ..."
    )
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
        print(
            f"\n[eval] --with-llm: generating real answers via LLM_DRIVER={settings.LLM_DRIVER} ..."
        )
        print_llm_answers(vector_only_results, hybrid_results)

    print(
        "\n"
        + wrap(
            f"Note: a curated question set ({len(questions)} questions), and the "
            "corpus is whatever's in document_chunks right now (at minimum the "
            "fixtures this script seeds — see the module docstring's "
            "'Which database?' section). These are our own, repeatable numbers "
            "for comparing configurations against each other, not a "
            "statistically significant benchmark."
        )
    )


if __name__ == "__main__":
    main()
