"""Evaluation and diagnostics CLI for docs-agent.

Consolidates all quality evaluation, chunk inspection, and document extraction diagnostics:
- Automated RAG quality evaluation (recall, MRR, fallback rate, and optional LLM scorecard)
- Multi-strategy chunking diagnostic matrix against embedding model token limits
- Document extraction sanity check (raw text and page/section preview)

Usage:
    uv run python scripts/eval_cli.py [command] [args]

Commands:
    eval, benchmark        Run the 25-question retrieval quality evaluation suite (default).
                           Options:
                             --with-llm         Generate real answers via LLM and measure hallucinations.
                             --with-rerank      Run hybrid retrieval with cross_encoder reranking.
                             --reranker <name>  Explicitly specify RERANKER_DRIVER (e.g. cross_encoder).
    inspect [path]         Compare chunking strategies and token overflows for a document.
                           (Defaults to TEST_DOC_PATH from .env if omitted).
    extract [path]         Preview raw text extraction grouped by page or markdown section.
                           (Defaults to TEST_DOC_PATH from .env if omitted).
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlsplit

# Ensure project root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from drivers.embedding import get_embedding_driver
from drivers.llm import get_answer_driver
from ingestion.chunker import (
    SplitOverflowStrategy,
)
from ingestion.extractors import get_extractor
from ingestion.ingest import add_document
from models import Chunk, RetrievedChunk
from query.retrieval import (
    NO_RESULTS_MESSAGE,
    HybridRetrievalStrategy,
    VectorRetrievalStrategy,
    retrieve_chunks,
)
from scripts.utils import (
    format_paragraphs,
    resolve_doc_path,
    truncate,
    wrap,
)
from store import VectorStore

BAR_WIDTH = 30
PREVIEW_CHARS = 500
OVERFLOW_STRATEGIES = ["warn", "split"]

EVAL_DATA_DIR = PROJECT_ROOT / "tests" / "data"
EVAL_QUESTIONS_PATH = EVAL_DATA_DIR / "eval_questions.json"


def discover_eval_fixtures(data_dir: Path = EVAL_DATA_DIR) -> list[Path]:
    """Discover all document fixtures in tests/data (excluding JSON files and hidden files)."""
    return sorted(
        p
        for p in data_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".md", ".pdf", ".txt"}
        and not p.name.startswith(".")
    )


QUESTION_COLUMN_WIDTH = 42
STATUS_COLUMN_WIDTH = 11


# --- Extraction Diagnostics ---


def print_extraction_report(
    doc_path: Path, full_text: str, word_page_map: list[int]
) -> None:
    """Print human-readable extraction report grouped by page/section."""
    words = full_text.split()

    print("=" * 60)
    print(f"File          : {doc_path.name}")
    print(f"Pages/sections: {len(set(word_page_map))}")
    print(f"Total words   : {len(words)}")
    print(f"Total chars   : {len(full_text)}")
    print("=" * 60)

    if not words:
        print("\n⚠️  WARNING: No text found.")
        print("   For a PDF, this usually means it is scanned (image-based) —")
        print("   OCR would be required to extract text.")
        return

    section_order: list[int] = []
    section_words: dict[int, list[str]] = {}
    for word, page in zip(words, word_page_map, strict=True):
        if page not in section_words:
            section_words[page] = []
            section_order.append(page)
        section_words[page].append(word)

    for page in section_order:
        text = " ".join(section_words[page])
        print(f"\n--- Page/section {page} ({len(text)} chars) ---")
        print(truncate(text, PREVIEW_CHARS))
        if len(text) > PREVIEW_CHARS:
            print(f"  ... [{len(text) - PREVIEW_CHARS} more characters]")


def cmd_extract(argv: list[str]) -> int:
    """Run text extraction diagnostic."""
    path_arg = argv[0] if argv else None
    doc_path = resolve_doc_path(path_arg)

    print(f"\n📄 Extracting text from: {doc_path}\n")
    extractor = get_extractor(doc_path)
    full_text, word_page_map = extractor.extract(doc_path)
    print_extraction_report(doc_path, full_text, word_page_map)
    return 0


# --- Chunk Inspection Diagnostics ---


def _combinations_for(doc_path: Path) -> list[tuple[str, str]]:
    if doc_path.suffix.lower() == ".pdf":
        return [("flat", "word"), ("flat", "langchain"), ("blocks", "langchain")]
    return [("native", "word"), ("native", "langchain")]


def render_bar(tokens: int, max_seq_length: int, width: int = BAR_WIDTH) -> str:
    """Render a text bar representing token length against max_sequence_length."""
    if max_seq_length <= 0:
        return ""
    fill_len = min(width, round((tokens / max_seq_length) * width))
    bar = "█" * fill_len + "░" * (width - fill_len)
    pct = (tokens / max_seq_length) * 100
    return f"[{bar}] {pct:>5.1f}%"


def _chunks_for(
    doc_path: Path, extraction_mode: str, strategy: str, driver
) -> list[Chunk]:
    import ingestion.chunker as chunker_module
    from ingestion.chunker import chunk_document
    from ingestion.extractors import get_extractor

    original_settings = chunker_module.settings
    try:
        chunker_module.settings = replace(original_settings, CHUNKING_STRATEGY=strategy)
        extractor = get_extractor(doc_path)
        full_text, word_page_map, word_header_map = extractor.extract_with_headers(
            doc_path, mode=extraction_mode
        )
        return chunk_document(
            full_text,
            word_page_map,
            source_file=doc_path.name,
            driver=driver,
            word_header_map=word_header_map,
        )
    finally:
        chunker_module.settings = original_settings


def print_comparison_matrix(doc_path: Path, driver, max_seq_length: int) -> None:
    """Print one row per (extraction, chunking, overflow) combination, each with a bar."""
    driver.count_tokens("warm-up")
    import langchain_text_splitters  # noqa: F401

    print(
        "\nConfiguration comparison — every extraction x chunking x "
        "CHUNK_OVERFLOW_STRATEGY combination for this file:\n"
    )
    header = (
        f"  {'extraction':<11}{'strategy':<11}{'overflow':<9}"
        f"{'chunks':>7}{'avg':>6}{'max':>6}{'ms':>8}  bar (worst chunk vs. limit)"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for extraction_mode, strategy in _combinations_for(doc_path):
        start = time.perf_counter()
        raw_chunks = _chunks_for(doc_path, extraction_mode, strategy, driver)
        extract_and_chunk_seconds = time.perf_counter() - start

        start = time.perf_counter()
        corrected_chunks = SplitOverflowStrategy().apply(raw_chunks, driver)
        correction_seconds = time.perf_counter() - start

        raw_tokens = [driver.count_tokens(c.content) for c in raw_chunks]
        corrected_tokens = [driver.count_tokens(c.content) for c in corrected_chunks]

        for overflow_strategy, chunks, tokens, elapsed_seconds in [
            ("warn", raw_chunks, raw_tokens, extract_and_chunk_seconds),
            (
                "split",
                corrected_chunks,
                corrected_tokens,
                extract_and_chunk_seconds + correction_seconds,
            ),
        ]:
            avg_tok = sum(tokens) / len(tokens) if tokens else 0
            worst = max(tokens, default=0)
            flag = "  OVERFLOW" if worst > max_seq_length else ""
            print(
                f"  {extraction_mode:<11}{strategy:<11}{overflow_strategy:<9}"
                f"{len(chunks):>7}{avg_tok:>6.0f}{worst:>6}{elapsed_seconds * 1000:>8.1f}  "
                f"{render_bar(worst, max_seq_length)}{flag}"
            )

    print(
        "\n"
        + wrap(
            "Note: 'word' relies on CHUNK_OVERFLOW_STRATEGY=split to prevent truncation; "
            "'langchain' sizes chunks in tokens from the start. "
            "Timing is indicative of local processing overhead."
        )
    )


def cmd_inspect(argv: list[str]) -> int:
    """Run chunking strategy diagnostic matrix."""
    path_arg = argv[0] if argv else None
    doc_path = resolve_doc_path(path_arg)

    driver = get_embedding_driver()
    max_seq_length = driver.max_sequence_length()
    if max_seq_length is None:
        print(
            f"EMBEDDING_DRIVER={settings.EMBEDDING_DRIVER} has no max_sequence_length "
            "to check against (e.g. OpenAI driver) — nothing to visualize."
        )
        return 0

    print("=" * 60)
    print(f"File      : {doc_path.name}")
    print(
        f"CHUNK_SIZE={settings.CHUNK_SIZE} words, CHUNK_OVERLAP={settings.CHUNK_OVERLAP} words"
    )
    print(
        f"EMBEDDING_DRIVER={settings.EMBEDDING_DRIVER}, max_sequence_length={max_seq_length}"
    )
    print("=" * 60)

    print_comparison_matrix(doc_path, driver, max_seq_length)
    return 0


# --- Retrieval Quality Evaluation ---


def _print_target_database() -> None:
    parsed = urlsplit(settings.DATABASE_URL)
    print(
        f"[eval] Target database: {parsed.hostname}:{parsed.port}{parsed.path} "
        f"(AGENT_ENV={settings.AGENT_ENV})"
    )


def _ensure_fixtures_seeded() -> None:
    store = VectorStore()
    for doc_path in discover_eval_fixtures():
        if store.has_chunks_from_source(doc_path.name):
            print(f"[eval] {doc_path.name} already present — skipping re-ingest.")
        else:
            add_document(doc_path)


def _precompute_embeddings(questions: list[dict]) -> dict[str, list[float]]:
    driver = get_embedding_driver()
    print(f"[eval] Pre-embedding {len(questions)} question(s) ...")
    return {q["question"]: driver.embed_query(q["question"]) for q in questions}


def _match_ranks(
    chunks: list[RetrievedChunk], q: dict
) -> tuple[int | None, int | None]:
    expected_file = q.get("expected_source_file")
    expected_text = q.get("expected_text_contains")

    if not expected_file:
        return None, None

    file_rank: int | None = None
    passage_rank: int | None = None

    for rank, chunk in enumerate(chunks, start=1):
        if chunk.metadata.source_file == expected_file:
            if file_rank is None:
                file_rank = rank
            if expected_text and expected_text.lower() in chunk.content.lower():
                if passage_rank is None:
                    passage_rank = rank
                    break
            elif not expected_text and passage_rank is None:
                passage_rank = rank
                break

    return file_rank, passage_rank


def evaluate(config_name: str, retrieve_fn, questions: list[dict]) -> dict:
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


def _looks_like_a_decline(answer: str) -> bool:
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

    v_dec = _fmt_rate(v_stats["ans_declined"], ans_tot)
    h_dec = _fmt_rate(h_stats["ans_declined"], ans_tot)
    print(f"{'  - Declined / Unanswered':<42} {v_dec:>17} {h_dec:>17}")

    unans_tot = v_stats["unans_total"]
    print(f"\nUnanswerable / Hallucination Gate ({unans_tot}):")

    v_ret_rej = _fmt_rate(v_stats["unans_retrieval_rejected"], unans_tot)
    h_ret_rej = _fmt_rate(h_stats["unans_retrieval_rejected"], unans_tot)
    print(
        f"{'  - Filtered at Retrieval (0 chunks)':<42} {v_ret_rej:>17} {h_ret_rej:>17}"
    )

    v_prompt_dec = _fmt_rate(v_stats["unans_prompt_declined"], unans_tot)
    h_prompt_dec = _fmt_rate(h_stats["unans_prompt_declined"], unans_tot)
    print(
        f"{'  - Safely Declined by Prompt':<42} {v_prompt_dec:>17} {h_prompt_dec:>17}"
    )

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
                    if is_decline:
                        s["ans_declined"] += 1
                        tag = "⚠️  DECLINED BY PROMPT"
                    elif expected_fact and expected_fact.lower() in answer.lower():
                        s["gold_matches"] += 1
                        tag = "✅ GOLD FACT MATCH"
                    else:
                        tag = "ℹ️  ANSWERED (FACT NOT FOUND)"

            print(f"[{cfg_name}] ({len(chunks)} chunks, {latency:.2f}s) [{tag}]")
            print(format_paragraphs(answer, indent="  ", width=76))
            print()

    _print_llm_scorecard(stats["vec"], stats["hyb"])


def cmd_eval(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="eval_cli.py eval",
        description="Retrieval-quality evaluation and LLM generation benchmark.",
    )
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Generate real LLM answers and benchmark hallucination rejection rate.",
    )
    parser.add_argument(
        "--with-rerank",
        action="store_true",
        help="Run hybrid retrieval with cross_encoder reranking enabled.",
    )
    parser.add_argument(
        "--reranker",
        type=str,
        default=None,
        help="Explicitly override RERANKER_DRIVER (e.g. 'cross_encoder').",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.with_rerank and not args.reranker:
        settings.RERANKER_DRIVER = "cross_encoder"
    elif args.reranker:
        settings.RERANKER_DRIVER = args.reranker

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
    return 0


# --- CLI Dispatcher ---


def print_help() -> None:
    print((__doc__ or "").strip())


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        return cmd_eval([])

    command = argv[0]
    sub_args = argv[1:]

    if command in {"--help", "-h", "help"}:
        print_help()
        return 0

    if command in {"eval", "benchmark"}:
        return cmd_eval(sub_args)

    if command in {"--with-llm", "--with-rerank", "--reranker"}:
        return cmd_eval(argv)

    if command == "inspect":
        return cmd_inspect(sub_args)

    if command == "extract":
        return cmd_extract(sub_args)

    print(f"Unknown command: '{command}'")
    print("Available commands: eval (default), inspect, extract")
    return 1


if __name__ == "__main__":
    sys.exit(main())
