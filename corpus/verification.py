"""Shared citation-verification primitives: does a citation exist, and
what is its real content.

Used by both `corpus/commands/generate_questions.py` (verifying a freshly
drafted golden question before trusting it) and
`corpus/commands/eval.py`'s `IndependentFactGradingStrategy` (verifying
what a RAG system's own generated answer claims, independent of a golden
question's originally-sampled citations). Each caller writes its own
LLM-verification prompt on top of these -- the two tasks are framed
differently enough (verifying a drafted question vs. grading a live
answer) that sharing prompt text would blur two genuinely different jobs;
only the pure, non-prompt pieces live here.

Key exports:
    extract_json           -- Parse the first JSON object out of an LLM response.
    verify_citation_exists -- Tier 1: does this source_file exist in document_chunks?
    fetch_chunk_bodies     -- Every chunk of one source_file as (chunk_index, own text).
    fetch_full_content     -- The document's own text, chunk by chunk, in order.
"""

from llm_json import extract_json  # noqa: F401  (re-exported: the existing import path)


def verify_citation_exists(citation: dict) -> bool:
    """Tier 1: does this citation's source_file match a real ingested document?

    Deterministic DB lookup -- no LLM involved. Matches purely on
    ``source_file``, the one field guaranteed to be exact (court/case_number
    are reconstructed/free text for prompt readability, not authoritative).

    Args:
        citation: A ``{"court", "case_number", "source_file"}`` dict.

    Returns:
        True if a chunk with this source_file exists in document_chunks.
    """
    from db import get_connection

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM document_chunks WHERE metadata->>'source_file' = %s LIMIT 1;",
                (citation["source_file"],),
            )
            return cur.fetchone() is not None
    finally:
        conn.close()


def fetch_chunk_bodies(source_file: str) -> list[tuple[int, str]]:
    """Every chunk of one source_file as ``(chunk_index, own text)``, in chunk order.

    The own text is the stored content without the document summary, the identifiers and
    the heading that open every stored chunk (``RetrievedChunk.body``): a passage of those
    is not a passage of the document.
    """
    from db import get_connection
    from models import ChunkMetadata, RetrievedChunk

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, content, metadata FROM document_chunks
                WHERE metadata->>'source_file' = %s
                ORDER BY (metadata->>'chunk_index')::int;
                """,
                (source_file,),
            )
            chunks = [
                RetrievedChunk(
                    id=row[0],
                    content=row[1],
                    metadata=ChunkMetadata.from_dict(row[2]),
                    score=0.0,
                )
                for row in cur.fetchall()
            ]
    finally:
        conn.close()
    return [(c.metadata.chunk_index, c.body) for c in chunks]


def fetch_full_content(source_file: str) -> str:
    """The document's own text: every chunk's own text, in chunk order.

    Without the summary, identifiers and heading that open each stored chunk: they repeat
    in all of the document's chunks (about 25), and a reader that checks a claim against
    this text should neither be shown the generated summary as if it were the document nor
    spend a sixth of its character budget on it. Neighbouring chunks overlap by a few words.
    """
    return "\n\n".join(body for _, body in fetch_chunk_bodies(source_file))
