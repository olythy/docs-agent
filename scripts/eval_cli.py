"""Diagnostics CLI for docs-agent: how a document is extracted and chunked.

- Multi-strategy chunking diagnostic matrix against the embedding model's token limits
- Document extraction sanity check (raw text and page/section preview)

The retrieval/answer quality evaluation lives in the golden-set sub-app
(``corpus/cli.py eval``); the older 25-question benchmark that compared retrieval
variants was retired with the ``vector`` profile (see docs/decisions.md, 2026-10-08).

Usage:
    uv run python scripts/eval_cli.py [command] [args]

Commands:
    inspect [path]         Compare chunking strategies and token overflows for a document.
                           (Defaults to TEST_DOC_PATH from .env if omitted).
    extract [path]         Preview raw text extraction grouped by page or markdown section.
                           (Defaults to TEST_DOC_PATH from .env if omitted).
"""

import sys
import time
from dataclasses import replace
from pathlib import Path

# Ensure project root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from drivers.embedding import get_embedding_driver
from ingestion.chunker import (
    SplitOverflowStrategy,
)
from ingestion.extractors import get_extractor
from models import Chunk
from scripts.utils import (
    resolve_doc_path,
    truncate,
    wrap,
)

BAR_WIDTH = 30
PREVIEW_CHARS = 500
OVERFLOW_STRATEGIES = ["warn", "split"]


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


def print_help() -> None:
    print((__doc__ or "").strip())


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if not argv or argv[0] in {"--help", "-h", "help"}:
        print_help()
        return 0

    command, sub_args = argv[0], argv[1:]

    if command == "inspect":
        return cmd_inspect(sub_args)

    if command == "extract":
        return cmd_extract(sub_args)

    print(f"Unknown command: '{command}'")
    print("Available commands: inspect, extract")
    return 1


if __name__ == "__main__":
    sys.exit(main())
