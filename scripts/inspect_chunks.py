"""Chunking diagnostic script.

Purpose:
    A hand-runnable tool that shows every meaningful combination of
    extraction mode x chunking strategy x CHUNK_OVERFLOW_STRATEGY together,
    in one table, each row with its own bar — so "which config is actually
    best for this document?" is a glance, not several separate reports to
    hold in your head at once. Works for PDF and Markdown alike.

    The actual extraction/chunking/strategy logic lives in
    ``ingestion/extractors.py`` and ``ingestion/chunker.py`` — this script
    only adds CLI handling and a human-readable console report, using the
    exact same functions ``ingestion/ingest.py`` calls (``get_extractor``
    + ``chunk_document``), so what this script shows matches what
    ingestion actually does.

Usage:
    # Pass the document path as a CLI argument:
    python scripts/inspect_chunks.py /path/to/document.pdf
    python scripts/inspect_chunks.py /path/to/notes.md

    # Or set TEST_DOC_PATH in .env and run without arguments:
    python scripts/inspect_chunks.py

    # Uses CHUNK_SIZE/CHUNK_OVERLAP/EMBEDDING_DRIVER from .env. Every
    # extraction-mode/CHUNKING_STRATEGY/CHUNK_OVERFLOW_STRATEGY combination
    # is shown regardless of what's set in .env — no DB connection needed,
    # nothing is stored.
"""

import sys
import time
from dataclasses import replace
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is importable (needed for running as a script)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BAR_WIDTH = 30  # Character width representing max_sequence_length

#: Every CHUNK_OVERFLOW_STRATEGY, applied on top of each chunking combination
#: below — so the full matrix is combinations-for-this-file x these.
OVERFLOW_STRATEGIES = ["warn", "split"]


def _combinations_for(doc_path: Path) -> list[tuple[str, str]]:
    """(extraction_mode, CHUNKING_STRATEGY) combinations worth comparing for this file.

    PDFs have two extraction modes worth comparing: "word" + "blocks" is
    skipped even so, since WordChunkingStrategy works on full_text.split(),
    which treats the "\\n\\n" paragraph markers blocks-mode inserts as plain
    whitespace — word+blocks always matches word+flat exactly. Only
    "langchain" actually looks for "\\n\\n", so only it benefits from
    blocks-mode extraction.

    Markdown has no extraction-mode concept at all — MarkdownExtractor
    ignores the mode argument entirely, since its paragraph structure is
    already native to the file — so only the chunking strategy varies.
    """
    if doc_path.suffix.lower() == ".pdf":
        return [("flat", "word"), ("flat", "langchain"), ("blocks", "langchain")]
    return [("native", "word"), ("native", "langchain")]


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


def _chunks_for(doc_path: Path, extraction_mode: str, strategy: str, driver) -> list[dict]:
    """Run the real ingest.py pipeline (get_extractor + chunk_document) for
    one (extraction_mode, CHUNKING_STRATEGY) combination.

    Temporarily overrides ``ingestion.chunker``'s ``settings`` (restored in a
    ``finally``) — the same technique the test suite uses — since
    ``chunk_document`` reads ``CHUNKING_STRATEGY`` from there rather than
    taking it as a parameter.
    """
    import ingestion.chunker as chunker_module
    from ingestion.chunker import chunk_document
    from ingestion.extractors import get_extractor

    original_settings = chunker_module.settings
    try:
        chunker_module.settings = replace(original_settings, CHUNKING_STRATEGY=strategy)
        extractor = get_extractor(doc_path)
        full_text, word_page_map = extractor.extract(doc_path, mode=extraction_mode)
        return chunk_document(full_text, word_page_map, source_file=doc_path.name, driver=driver)
    finally:
        chunker_module.settings = original_settings


def print_comparison_matrix(doc_path: Path, driver, max_seq_length: int) -> None:
    """Print one row per (extraction, chunking, overflow) combination, each with a bar.

    The bar visualizes the *worst* chunk in that row (max real token count)
    against ``max_seq_length`` — that's the number that decides whether
    anything would actually get silently truncated, which matters more for
    judging a config than the average does.

    The "ms" column times only the real pipeline work each row would
    actually cost in ``add_document()`` — extraction + chunking for
    "warn", plus ``SplitOverflowStrategy``'s correction pass for "split".
    It deliberately excludes the ``count_tokens()`` calls this function
    makes for the avg/max columns themselves: those exist only for this
    report, ``WarnOverflowStrategy`` never tokenizes every chunk in a real
    run, so counting them here would make "warn" look more expensive than
    it actually is relative to "split".
    """
    from ingestion.chunker import SplitOverflowStrategy

    combinations = _combinations_for(doc_path)

    # Force every lazily-loaded dependency to load *before* any timing
    # starts. Without this, whichever combination happens to run first
    # would unfairly absorb a one-time load cost into its measurement,
    # making it look like the slowest option by pure luck of being first —
    # these loads only ever happen once per process, and aren't part of
    # what this table is trying to compare.
    #   - the local model itself (~8s on this machine — mostly torch).
    driver.count_tokens("warm-up")
    #   - langchain_text_splitters (~0.5s on top of the above, confirmed
    #     empirically — its own import, separate from the model's).
    import langchain_text_splitters  # noqa: F401

    print("\nConfiguration comparison — every extraction x chunking x "
          "CHUNK_OVERFLOW_STRATEGY combination for this file:\n")
    header = (
        f"  {'extraction':<11}{'strategy':<11}{'overflow':<9}"
        f"{'chunks':>7}{'avg':>6}{'max':>6}{'ms':>8}  bar (worst chunk vs. limit)"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for extraction_mode, strategy in combinations:
        start = time.perf_counter()
        raw_chunks = _chunks_for(doc_path, extraction_mode, strategy, driver)
        extract_and_chunk_seconds = time.perf_counter() - start

        start = time.perf_counter()
        corrected_chunks = SplitOverflowStrategy().apply(raw_chunks, driver)
        correction_seconds = time.perf_counter() - start

        # Tokenizing every chunk for the avg/max columns is diagnostic-only
        # (see docstring) — timed separately from, and excluded from, the
        # pipeline times above.
        raw_tokens = [driver.count_tokens(c["content"]) for c in raw_chunks]
        corrected_tokens = [driver.count_tokens(c["content"]) for c in corrected_chunks]

        for overflow_strategy, chunks, tokens, elapsed_seconds in [
            ("warn", raw_chunks, raw_tokens, extract_and_chunk_seconds),
            ("split", corrected_chunks, corrected_tokens, extract_and_chunk_seconds + correction_seconds),
        ]:
            avg_tok = sum(tokens) / len(tokens) if tokens else 0
            worst = max(tokens, default=0)
            flag = "  OVERFLOW" if worst > max_seq_length else ""
            print(
                f"  {extraction_mode:<11}{strategy:<11}{overflow_strategy:<9}"
                f"{len(chunks):>7}{avg_tok:>6.0f}{worst:>6}{elapsed_seconds * 1000:>7.1f}  "
                f"{render_bar(worst, max_seq_length)}{flag}"
            )

    print(
        "\nRead this top to bottom: 'word' relies entirely on CHUNK_OVERFLOW_STRATEGY=split "
        "to avoid truncation (its 'warn' row is often OVERFLOW); 'langchain' sizes chunks in "
        "tokens from the start, so its 'warn' and 'split' rows are usually identical — split "
        "has nothing left to correct. For a PDF, 'blocks' vs 'flat' shows whether "
        "paragraph-aware extraction changed anything; Markdown has no extraction mode at all "
        "('native' — it's already structured), so only the strategy varies. 'ms' is real "
        "pipeline time (model load excluded) — 'split' costs more than 'warn' by design, the "
        "question is whether that cost is worth paying for this document. Single-run timing, "
        "not a rigorous benchmark: for a PDF, the first row to open the file pays a bit of "
        "one-time OS file-cache warm-up too, so treat 'ms' as indicative, not exact."
    )


def main() -> None:
    """Entry point: resolve the target document and print the comparison matrix.

    Resolution order for the target file:
    1. CLI argument: ``python scripts/inspect_chunks.py /path/to/file.pdf``
    2. ``TEST_DOC_PATH`` environment variable (set in .env)
    3. Exits with a helpful error if neither is provided.
    """
    from config import settings
    from drivers.embedding import get_embedding_driver

    if len(sys.argv) > 1:
        doc_path = Path(sys.argv[1])
    elif settings.TEST_DOC_PATH:
        doc_path = Path(settings.TEST_DOC_PATH)
    else:
        print("Error: No document path provided.")
        print("  Usage: python scripts/inspect_chunks.py /path/to/document.pdf")
        print("  Or set TEST_DOC_PATH in your .env file.")
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
    print(f"File      : {doc_path.name}")
    print(f"CHUNK_SIZE={settings.CHUNK_SIZE} words, CHUNK_OVERLAP={settings.CHUNK_OVERLAP} words")
    print(f"EMBEDDING_DRIVER={settings.EMBEDDING_DRIVER}, max_sequence_length={max_seq_length}")
    print("=" * 60)

    print_comparison_matrix(doc_path, driver, max_seq_length)


if __name__ == "__main__":
    main()
