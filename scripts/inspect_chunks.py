"""Chunking diagnostic script.

Purpose:
    A hand-runnable tool that shows every meaningful combination of
    PDF_EXTRACTION_MODE x CHUNKING_STRATEGY x CHUNK_OVERFLOW_STRATEGY
    together, in one table, each row with its own bar — so "which config
    is actually best for this document?" is a glance, not six separate
    reports to hold in your head at once.

    The actual chunking/strategy logic lives in ``ingestion/chunker.py``
    and ``ingestion/pdf_loader.py`` — this script only adds CLI handling
    and a human-readable console report, using the exact same functions
    ``ingestion/ingest.py`` calls (``extract_document_text`` +
    ``chunk_document``), so what this script shows matches what ingestion
    actually does.

Usage:
    # Pass the PDF path as a CLI argument:
    python scripts/inspect_chunks.py /path/to/document.pdf

    # Or set TEST_PDF_PATH in .env and run without arguments:
    python scripts/inspect_chunks.py

    # Uses CHUNK_SIZE/CHUNK_OVERLAP/EMBEDDING_DRIVER from .env. Every
    # PDF_EXTRACTION_MODE/CHUNKING_STRATEGY/CHUNK_OVERFLOW_STRATEGY
    # combination is shown regardless of what's set in .env — no DB
    # connection needed, nothing is stored.
"""

import sys
from dataclasses import replace
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is importable (needed for running as a script)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BAR_WIDTH = 30  # Character width representing max_sequence_length

#: (PDF_EXTRACTION_MODE, CHUNKING_STRATEGY) combinations worth comparing.
#: "word" + "blocks" is deliberately omitted: WordChunkingStrategy works on
#: full_text.split(), which treats the "\n\n" paragraph markers blocks-mode
#: inserts as plain whitespace — so word+blocks always produces identical
#: chunks to word+flat. Only a strategy that actually looks for "\n\n"
#: (langchain) benefits from blocks-mode extraction.
CHUNKING_COMBINATIONS = [
    ("flat", "word"),
    ("flat", "langchain"),
    ("blocks", "langchain"),
]

#: Every CHUNK_OVERFLOW_STRATEGY, applied on top of each chunking combination
#: above — so the full matrix is CHUNKING_COMBINATIONS x OVERFLOW_STRATEGIES.
OVERFLOW_STRATEGIES = ["warn", "split"]


def render_bar(tokens: int, max_seq_length: int, width: int = BAR_WIDTH) -> str:
    """Render an ASCII bar scaled so ``max_seq_length`` lands at ``width`` chars.

    Tokens within budget render as solid blocks (``█``); tokens beyond the
    limit render as a distinct overflow fill (``▓``), capped so one extreme
    outlier can't blow out the whole table's alignment.
    """
    scale = width / max_seq_length
    filled = round(tokens * scale)
    within = min(filled, width)
    overflow = min(filled - width, width // 2)  # cap the visual overshoot
    bar = "█" * within
    if overflow > 0:
        bar += "▓" * overflow
    return f"[{bar:<{width}}]"


def _chunks_for(pdf_path: Path, extraction_mode: str, strategy: str, driver) -> list[dict]:
    """Run the real ingest.py pipeline (extract_document_text + chunk_document)
    for one (PDF_EXTRACTION_MODE, CHUNKING_STRATEGY) combination.

    Temporarily overrides ``ingestion.chunker``'s ``settings`` (restored in a
    ``finally``) — the same technique the test suite uses — since
    ``chunk_document`` reads ``CHUNKING_STRATEGY`` from there rather than
    taking it as a parameter.
    """
    import ingestion.chunker as chunker_module
    from ingestion.chunker import chunk_document
    from ingestion.pdf_loader import extract_document_text

    original_settings = chunker_module.settings
    try:
        chunker_module.settings = replace(original_settings, CHUNKING_STRATEGY=strategy)
        full_text, word_page_map = extract_document_text(pdf_path, mode=extraction_mode)
        return chunk_document(full_text, word_page_map, source_file=pdf_path.name, driver=driver)
    finally:
        chunker_module.settings = original_settings


def print_comparison_matrix(pdf_path: Path, driver, max_seq_length: int) -> None:
    """Print one row per (extraction, chunking, overflow) combination, each with a bar.

    The bar visualizes the *worst* chunk in that row (max real token count)
    against ``max_seq_length`` — that's the number that decides whether
    anything would actually get silently truncated, which matters more for
    judging a config than the average does.
    """
    from ingestion.chunker import SplitOverflowStrategy

    print("\nConfiguration comparison — every PDF_EXTRACTION_MODE x CHUNKING_STRATEGY x "
          "CHUNK_OVERFLOW_STRATEGY combination:\n")
    header = (
        f"  {'extraction':<11}{'strategy':<11}{'overflow':<9}"
        f"{'chunks':>7}{'avg':>6}{'max':>6}  bar (worst chunk vs. limit)"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for extraction_mode, strategy in CHUNKING_COMBINATIONS:
        raw_chunks = _chunks_for(pdf_path, extraction_mode, strategy, driver)
        raw_tokens = [driver.count_tokens(c["content"]) for c in raw_chunks]
        corrected_chunks = SplitOverflowStrategy().apply(raw_chunks, driver)
        corrected_tokens = [driver.count_tokens(c["content"]) for c in corrected_chunks]

        for overflow_strategy, chunks, tokens in [
            ("warn", raw_chunks, raw_tokens),
            ("split", corrected_chunks, corrected_tokens),
        ]:
            avg_tok = sum(tokens) / len(tokens) if tokens else 0
            worst = max(tokens, default=0)
            flag = "  OVERFLOW" if worst > max_seq_length else ""
            print(
                f"  {extraction_mode:<11}{strategy:<11}{overflow_strategy:<9}"
                f"{len(chunks):>7}{avg_tok:>6.0f}{worst:>6}  "
                f"{render_bar(worst, max_seq_length)}{flag}"
            )

    print(
        "\nRead this top to bottom: 'word' relies entirely on CHUNK_OVERFLOW_STRATEGY=split "
        "to avoid truncation (its 'warn' row is often OVERFLOW); 'langchain' sizes chunks in "
        "tokens from the start, so its 'warn' and 'split' rows are usually identical — split "
        "has nothing left to correct. 'blocks' vs 'flat' shows whether paragraph-aware "
        "extraction changed anything for this document."
    )


def main() -> None:
    """Entry point: resolve the target PDF and print the comparison matrix.

    Resolution order for the target file:
    1. CLI argument: ``python scripts/inspect_chunks.py /path/to/file.pdf``
    2. ``TEST_PDF_PATH`` environment variable (set in .env)
    3. Exits with a helpful error if neither is provided.
    """
    from config import settings
    from drivers.embedding import get_embedding_driver

    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
    elif settings.TEST_PDF_PATH:
        pdf_path = Path(settings.TEST_PDF_PATH)
    else:
        print("Error: No PDF path provided.")
        print("  Usage: python scripts/inspect_chunks.py /path/to/document.pdf")
        print("  Or set TEST_PDF_PATH in your .env file.")
        sys.exit(1)

    driver = get_embedding_driver()
    max_seq_length = driver.max_sequence_length()
    if max_seq_length is None:
        print(
            f"EMBEDDING_DRIVER={settings.EMBEDDING_DRIVER} has no max_sequence_length "
            "to check against (e.g. the OpenAI driver) — nothing to visualize."
        )
        sys.exit(0)

    print("=" * 60)
    print(f"File      : {pdf_path.name}")
    print(f"CHUNK_SIZE={settings.CHUNK_SIZE} words, CHUNK_OVERLAP={settings.CHUNK_OVERLAP} words")
    print(f"EMBEDDING_DRIVER={settings.EMBEDDING_DRIVER}, max_sequence_length={max_seq_length}")
    print("=" * 60)

    print_comparison_matrix(pdf_path, driver, max_seq_length)


if __name__ == "__main__":
    main()
