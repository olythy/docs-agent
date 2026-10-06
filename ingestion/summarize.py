"""Document-level, fact-focused summary generation for embedding enrichment.

Key export:
    generate_document_summary -- One LLM call per document (not per chunk),
        used by ingestion.ingest.add_document to populate
        models.ChunkMetadata.document_summary.
"""

from drivers.llm import AnswerDriver

#: How many leading characters of a document to send to the LLM. Mirrors
#: the budget confirmed live during the A/B test that validated this
#: approach (see docs/decisions.md) -- long enough to reach the operative
#: part of a typical court decision, short enough to keep the call cheap.
DEFAULT_MAX_CHARS = 6000

_SUMMARY_PROMPT_TEMPLATE = (
    "Ez egy dokumentum szövege. Készíts egy szigorúan tényfókuszú, "
    "2 mondatos összefoglalót, amely kiemeli az ÜGYET/DOKUMENTUMOT EGYEDIVÉ "
    "tévő konkrét tényeket -- NE általános kereteződést vagy sablonos "
    "fordulatokat írj.\n"
    "1. mondat (Tárgy): a dokumentum konkrét, egyedi tárgya -- ami ezt a "
    "dokumentumot megkülönbözteti a hasonló típusú dokumentumoktól (pl. "
    "konkrét hely/azonosító, érintett felek/szervezet neve, pontos téma).\n"
    "2. mondat (Eredmény/lényeg): a dokumentum TÉNYLEGES, KONKRÉT "
    "eredménye vagy fő állítása -- ne csak egy általános szót írj (pl. "
    '"megsemmisítette" vagy "elutasította" önmagában -- ezek a szavak '
    "majdnem minden hasonló dokumentumban előfordulnak a sablonos "
    "kereteződésben), írd le PONTOSAN mi történt vagy mi a konklúzió.\n"
    "Csak a két mondatot add vissza, semmi mást.\n\n"
    "Dokumentum szövege:\n{text}"
)


def generate_document_summary(
    full_text: str,
    answer_driver: AnswerDriver,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    """Generate a short, fact-focused summary of a whole document via an LLM.

    One call per document (not per chunk) -- the summary is identical for
    every chunk of the same document, so it's computed once and reused
    across all of them (see ingestion.chunker.chunk_document).

    Confirmed live (see docs/decisions.md): a *generic* document summary
    ("this is an administrative review of an expropriation decision")
    makes near-duplicate documents' embeddings *more* alike, not less --
    the prompt here explicitly asks for the document's own distinguishing
    facts (specific entities, concrete outcome) instead of a topic-level
    restatement, which measurably improved retrieval rank on a known-hard
    case (though not enough alone -- paired with CSLS re-ranking and/or a
    listwise comparative rerank for the final disambiguation).

    Args:
        full_text: The whole document's extracted text.
        answer_driver: Any :class:`drivers.llm.AnswerDriver` -- uses
            :meth:`run_tool_calling_turn`, the same generic single-prompt
            completion mechanism every concrete driver already implements
            for agent.py's tool-calling loop, with no tools offered.
        max_chars: How many leading characters of ``full_text`` to send.

    Returns:
        A short (typically 2-sentence) summary string, or an empty string
        if the model returned no content.
    """
    prompt = _SUMMARY_PROMPT_TEMPLATE.format(text=full_text[:max_chars])
    result = answer_driver.run_tool_calling_turn(
        messages=[{"role": "user", "content": prompt}]
    )
    return (result.content or "").strip()
