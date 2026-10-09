"""A golden citation names the passage that supports the answer, not a chunk.

A chunk's index and id change when the corpus is ingested again with other settings; the
sentence of the decision that states the fact does not. So a golden citation carries a
``quote``: a word-for-word passage of the document's own text. At evaluation time the quote
is *located* in the chunks as they are now, and whether a system found or cited the right
section is whether the chunk it retrieved or cited contains that quote.

Key exports:
    locate         -- Which chunks contain a quote.
    section_hits   -- Whether a set of chunks holds every golden passage.
    add_quotes     -- Gives every golden citation without one a verified quote.
    QuoteReport    -- What happened.
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from llm_json import extract_json
from metadata.verification import EvidenceVerifier

#: The longest quote asked for, in characters.
MAX_QUOTE_CHARS = 250

Chunks = Sequence[tuple[int, str]]
#: Takes (question, expected answer, citation, numbered chunk texts) and returns the
#: model's JSON reply as a dict, or ``None``.
Ask = Callable[[dict, str, dict, Chunks, str], dict | None]


def locate(quote: str, chunks: Chunks) -> list[int]:
    """The ``chunk_index`` of every chunk whose own text contains ``quote``.

    Whitespace and case are ignored, then punctuation and accents; an ellipsis in the quote
    marks left-out text (see :class:`metadata.verification.EvidenceVerifier`).
    """
    return [index for index, body in chunks if EvidenceVerifier(body).contains(quote)]


def section_hits(
    citations: Sequence[dict], chunks: Sequence[tuple[str, str]]
) -> bool | None:
    """Whether every golden citation's supporting passage is in one of ``chunks``.

    Args:
        citations: The golden citations of a question (``source_file`` and ``quote``).
        chunks: ``(source_file, own text)`` of the chunks to look in.

    Returns:
        ``True`` if for every citation some chunk of the same document contains its quote;
        ``None`` if the question has no citation, or any citation has no quote yet (nothing
        can be said at section level).
    """
    if not citations or any(not c.get("quote") for c in citations):
        return None
    return all(
        any(
            source == citation["source_file"]
            and EvidenceVerifier(body).contains(citation["quote"])
            for source, body in chunks
        )
        for citation in citations
    )


@dataclass
class QuoteReport:
    """What :func:`add_quotes` did.

    Attributes:
        added: ``(question id, source_file, quote)`` for every quote verified and stored.
        failed: ``(question id, source_file, why)`` for every citation left without one.
        skipped: How many citations already had a quote.
    """

    added: list[tuple[str, str, str]] = field(default_factory=list)
    failed: list[tuple[str, str, str]] = field(default_factory=list)
    skipped: int = 0


def _shorten(quote: str) -> str:
    """``quote`` cut at a word boundary to ``MAX_QUOTE_CHARS``; the start of a verified
    passage is still a passage of the same chunk."""
    if len(quote) <= MAX_QUOTE_CHARS:
        return quote
    return quote[:MAX_QUOTE_CHARS].rsplit(None, 1)[0]


def add_quotes(
    questions: list[dict],
    fetch: Callable[[str], Chunks],
    ask: Ask,
    only: Callable[[dict], bool] = lambda question: True,
    attempts: int = 3,
) -> QuoteReport:
    """Give each golden citation without a ``quote`` one the code has verified.

    The model proposes a passage; the code accepts it only if it is in one of the
    document's chunks (:func:`locate`). A proposal longer than the limit is cut at a word boundary; one that is not in
    the document is tried again with the reason; a citation that still has none is reported
    and left without a ``quote``.

    Args:
        questions: The golden questions; changed in place.
        fetch: ``source_file`` to the document's ``(chunk_index, own text)`` list.
        ask: Asks the model.
        only: Which questions to touch.
        attempts: How many times to ask for one citation.
    """
    report = QuoteReport()
    for question in questions:
        if not only(question):
            continue
        for citation in question.get("citations", []):
            if citation.get("quote"):
                report.skipped += 1
                continue
            chunks = fetch(citation["source_file"])
            feedback = ""
            for _ in range(attempts):
                reply = ask(
                    question, question["expected_answer"], citation, chunks, feedback
                )
                quote = reply.get("quote") if isinstance(reply, dict) else None
                if not isinstance(quote, str) or not quote.strip():
                    feedback = "Your reply had no quote."
                elif not locate(quote, chunks):
                    feedback = "That quote is not word for word in any excerpt."
                else:
                    quote = _shorten(quote.strip())
                    citation["quote"] = quote
                    report.added.append(
                        (question["id"], citation["source_file"], quote)
                    )
                    break
            else:
                report.failed.append(
                    (question["id"], citation["source_file"], feedback)
                )
    return report


def llm_ask(
    question: dict, expected: str, citation: dict, chunks: Chunks, feedback: str
) -> dict | None:
    """Ask the answer driver for the passage that supports the expected answer."""
    from drivers.llm import get_answer_driver

    excerpts = "\n\n".join(f"[{index}] {body}" for index, body in chunks)
    prompt = (
        "A question about one court decision has an expected answer. Pick the ONE passage "
        "of the decision itself that states the key fact(s) of the expected answer about "
        f"this case, copied word for word from one excerpt, at most {MAX_QUOTE_CHARS} "
        "characters. For the ruling prefer the sentence that states the decision; for a "
        "subject or a party, the sentence that names it. Never use text that is not in the "
        "excerpts.\n\n"
        f"Question: {question['question']}\n"
        f"Expected answer: {expected}\n"
        f"The decision: {citation.get('court', '')} {citation.get('case_number', '')}\n"
        + (f"Note on your previous try: {feedback}\n" if feedback else "")
        + f"\nEXCERPTS\n{excerpts}\n\n"
        'Output ONLY this JSON: {"excerpt": <number>, "quote": "<the passage>"}'
    )
    reply = (
        get_answer_driver()
        .run_tool_calling_turn([{"role": "user", "content": prompt}], temperature=0.0)
        .content
    )
    try:
        data = extract_json(reply or "")
    except (ValueError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None
