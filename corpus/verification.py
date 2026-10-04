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
    fetch_full_content     -- Concatenate every chunk for one source_file, in order.
"""

import json


def extract_json(text: str) -> dict:
    """Parse the first JSON object found in an LLM response.

    Models sometimes wrap JSON in ```` ```json ... ``` ```` fences despite
    being asked not to -- strip those before parsing rather than failing.
    Also tolerates raw control characters (e.g. a literal newline) inside
    JSON strings, which ``json.loads``' default strict mode rejects: confirmed
    live, one such grader reply aborted a whole 33-question eval run.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped
        stripped = stripped.rsplit("```", 1)[0]
    return json.loads(stripped, strict=False)


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


def fetch_full_content(source_file: str) -> str:
    """Concatenate every chunk for one source_file, in chunk_index order."""
    from db import get_connection

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT content FROM document_chunks
                WHERE metadata->>'source_file' = %s
                ORDER BY (metadata->>'chunk_index')::int;
                """,
                (source_file,),
            )
            return "\n\n".join(row[0] for row in cur.fetchall())
    finally:
        conn.close()
