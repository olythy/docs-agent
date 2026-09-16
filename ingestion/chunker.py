"""Text chunking module.

Splits extracted page-level text into smaller, overlapping chunks suitable
for embedding and vector storage. Uses a simple word-count approximation
instead of a full tokenizer to avoid extra dependencies in a learning project.

A rough rule of thumb: 1 token ≈ 0.75 words (English/Hungarian average).
At 350–400 words per chunk we stay comfortably inside the typical 512-token
limit of sentence-transformer models.

Key exports:
    chunk_pages  -- Convert a list of page dicts into a flat list of chunk dicts.
"""

from config import settings


def _split_words_into_chunks(
    words: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[list[str]]:
    """Slide a window of ``chunk_size`` words over ``words`` with ``chunk_overlap`` overlap.

    Args:
        words: The full list of words to split.
        chunk_size: Target number of words per chunk.
        chunk_overlap: Number of words shared between adjacent chunks.

    Returns:
        A list of word-lists, each representing one chunk.
    """
    if not words:
        return []

    step = chunk_size - chunk_overlap
    chunks = []
    start = 0
    while start < len(words):
        chunk_words = words[start : start + chunk_size]
        chunks.append(chunk_words)
        start += step

    return chunks


def chunk_pages(
    pages: list[dict],
    source_file: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict]:
    """Split page-level text into overlapping word-based chunks.

    Each chunk dict is ready to be inserted into the ``document_chunks`` table:
    it carries the raw text content and a ``metadata`` dict that records where
    in the source document the chunk came from.

    Args:
        pages: Page dicts as returned by :func:`ingestion.pdf_loader.extract_pages`.
            Each dict must have ``page_number`` (int) and ``text`` (str).
        source_file: The basename of the source PDF (e.g. ``"invoice.pdf"``).
            Stored in metadata so answers can cite their source.
        chunk_size: Override for ``settings.CHUNK_SIZE`` (words per chunk).
        chunk_overlap: Override for ``settings.CHUNK_OVERLAP`` (overlap in words).

    Returns:
        A flat list of chunk dicts, each containing:
            - ``content`` (str): The chunk text, ready for embedding.
            - ``metadata`` (dict): ``source_file``, ``page_number``,
              ``chunk_index`` (0-based, global across the whole document).
    """
    size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
    overlap = chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP

    chunks: list[dict] = []
    global_chunk_index = 0

    for page in pages:
        text = page["text"].strip()
        if not text:
            # Skip blank/scanned pages — no content to chunk
            continue

        words = text.split()
        word_groups = _split_words_into_chunks(words, size, overlap)

        for word_group in word_groups:
            content = " ".join(word_group)
            chunks.append(
                {
                    "content": content,
                    "metadata": {
                        "source_file": source_file,
                        "page_number": page["page_number"],
                        "chunk_index": global_chunk_index,
                    },
                }
            )
            global_chunk_index += 1

    return chunks
