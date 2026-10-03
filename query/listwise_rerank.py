"""Listwise LLM reranking: a final disambiguation pass over near-duplicate candidates.

Key export:
    listwise_rerank -- Re-orders candidate chunks so the document an LLM
        judges as the actual answer (reading all candidates' document
        summaries at once, explicitly told to ignore shared/formulaic
        framing) comes first.

Confirmed live (see docs/decisions.md) as the single most effective fix for
a known-hard near-duplicate case: 19 real competing documents all read as
superficially similar ("the court annulled a decision and ordered a new
procedure"), but only one's document_summary was actually about the
specific matter the question asked. A per-pair cross-encoder reranker
can't make this distinction (it scores one candidate against the question
at a time, with no notion that several candidates share the same
formulaic framing) -- an LLM reading every candidate's summary side by
side, explicitly instructed to ignore what they have in common, can.
"""

import re

from drivers.llm import AnswerDriver
from models import RetrievedChunk

_PROMPT_TEMPLATE = """Egy kérdésre keresünk választ egy dokumentum-korpuszban. Az alábbiakban {n} jelölt dokumentum rövid összefoglalója szerepel -- ezek felszínesen nagyon hasonlóak lehetnek egymáshoz (azonos típusú/formájú dokumentumok, hasonló megfogalmazással).

KÉRDÉS: {question}

JELÖLTEK:
{candidates}

FELADAT: Hagyd figyelmen kívül a jelöltek közös, formulaszerű jellemzőit. Csak azt nézd, hogy melyik jelölt TARTALMA (a benne leírt konkrét, egyedi tények) felel meg valójában a kérdésnek.

Add vissza KIZÁRÓLAG a legjobban megfelelő jelölt sorszámát egyetlen számként, semmi mást (ha egyik sem felel meg jól, add vissza a legközelebbit)."""


def _build_candidates_block(chunks: list[RetrievedChunk]) -> list[tuple[str, int]]:
    """Group ``chunks`` by document, one representative per document.

    Args:
        chunks: Candidate chunks, already ranked (highest-scoring first).

    Returns:
        A list of ``(source_file, representative_chunk_index_in_chunks)``
        pairs, one per distinct document, in the order each document's
        best-scoring chunk first appears in ``chunks``.
    """
    seen: dict[str, int] = {}
    for i, chunk in enumerate(chunks):
        seen.setdefault(chunk.metadata.source_file, i)
    return list(seen.items())


def _parse_choice(response_text: str, num_candidates: int) -> int | None:
    """Extract the chosen 1-based candidate number from the LLM's reply.

    Args:
        response_text: The raw model response, expected to be just a number.
        num_candidates: How many candidates were offered -- a parsed number
            outside ``[1, num_candidates]`` is treated as unparseable.

    Returns:
        The chosen candidate's 0-based index into the document list, or
        ``None`` if no valid number could be parsed.
    """
    match = re.search(r"\d+", response_text)
    if not match:
        return None
    choice = int(match.group())
    if not (1 <= choice <= num_candidates):
        return None
    return choice - 1


def listwise_rerank(
    question: str,
    chunks: list[RetrievedChunk],
    answer_driver: AnswerDriver,
    max_candidates: int = 20,
) -> list[RetrievedChunk]:
    """Re-order ``chunks`` so the LLM's chosen document's chunks come first.

    Groups ``chunks`` by document (one representative per document, its
    highest-scoring chunk), asks the LLM to pick which document's
    ``document_summary`` actually answers ``question`` -- explicitly
    instructed to ignore shared/formulaic framing (see module docstring).
    Only the *order* changes: every input chunk is still present in the
    output, just with the chosen document's chunks moved to the front
    (preserving their relative order), everything else unchanged after
    that.

    Falls back to returning ``chunks`` unchanged if there are fewer than 2
    distinct documents (nothing to disambiguate), if no candidate has a
    ``document_summary`` yet (nothing for the LLM to compare), or if the
    LLM's reply can't be parsed into a valid choice -- this is a
    best-effort disambiguation step, not a required one, and must never
    make retrieval worse by dropping a chunk or crashing the query.

    Args:
        question: The user's natural-language question.
        chunks: Candidate chunks, already ranked (highest-scoring first).
        answer_driver: Any :class:`drivers.llm.AnswerDriver`.
        max_candidates: Only the first this many distinct documents (by
            current rank) are offered to the LLM -- keeps the prompt a
            bounded size regardless of how wide the candidate pool is.

    Returns:
        ``chunks``, re-ordered so the chosen document's chunks come first,
        or unchanged if disambiguation wasn't possible/needed.
    """
    documents = _build_candidates_block(chunks)[:max_candidates]
    if len(documents) < 2:
        return chunks

    summaries = {
        source_file: chunks[i].metadata.document_summary
        for source_file, i in documents
    }
    if not any(summaries.values()):
        return chunks

    candidates_text = "\n\n".join(
        f"[{idx + 1}] {summaries[source_file] or '(nincs összefoglaló)'}"
        for idx, (source_file, _) in enumerate(documents)
    )
    prompt = _PROMPT_TEMPLATE.format(
        n=len(documents), question=question, candidates=candidates_text
    )

    result = answer_driver.run_tool_calling_turn(
        messages=[{"role": "user", "content": prompt}]
    )
    chosen_index = _parse_choice(result.content or "", len(documents))
    if chosen_index is None:
        return chunks

    chosen_source_file = documents[chosen_index][0]
    chosen_chunks = [c for c in chunks if c.metadata.source_file == chosen_source_file]
    rest = [c for c in chunks if c.metadata.source_file != chosen_source_file]
    return chosen_chunks + rest
