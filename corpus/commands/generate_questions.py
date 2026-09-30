"""The `generate-questions` command: draft + two-tier-verify golden questions.

Reads skills/generate-golden-questions/SKILL.md (same instructions the
interactive Skill follows) and sends them to the project's own LLM_DRIVER
(drivers.llm.get_answer_driver()) to draft a question, then runs it through
the two-tier verification described there:
    1. Deterministic citation-existence check (verify_citation_exists()).
    2. A *separate* LLM call judging whether the cited content actually
       supports the drafted answer (verify_content_support()) -- never the
       same call that drafted the question, so it isn't grading its own work.

Appends the result to corpus/data/questions.json with verification_status
set to "verified" (both checks passed), "needs_review" (citation missing,
or content support NOT_SUPPORTED/UNCLEAR), or, for the adversarial persona,
"verified" whenever the citation is correctly empty (an adversarial
question has no real citation to check).

This unattended path is expected to be noisier than drafting the question
interactively inside Claude Code (see SKILL.md) -- that's exactly why
nothing here is trusted without the verification step.
"""

import json
import re
import sys
from pathlib import Path
from typing import Annotated

import typer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PERSONAS_PATH = DATA_DIR / "personas.json"
QUESTIONS_PATH = DATA_DIR / "questions.json"
SKILL_DIR = PROJECT_ROOT / "skills" / "generate-golden-questions"

app = typer.Typer()


def load_personas() -> dict[str, dict]:
    """Load corpus/data/personas.json, keyed by persona id."""
    personas = json.loads(PERSONAS_PATH.read_text())
    return {p["id"]: p for p in personas}


def load_questions() -> list[dict]:
    """Load corpus/data/questions.json (empty list if missing)."""
    if not QUESTIONS_PATH.exists():
        return []
    return json.loads(QUESTIONS_PATH.read_text())


def save_questions(questions: list[dict]) -> None:
    """Write corpus/data/questions.json, pretty-printed."""
    QUESTIONS_PATH.write_text(
        json.dumps(questions, indent=2, ensure_ascii=False) + "\n"
    )


def next_question_id(questions: list[dict]) -> str:
    """Return the next sequential id (e.g. 'q0004') given existing questions."""
    existing = [int(q["id"][1:]) for q in questions if re.fullmatch(r"q\d+", q["id"])]
    return f"q{(max(existing) + 1) if existing else 1:04d}"


def _case_number_from_filename(source_file: str) -> str:
    """Reconstruct a human-readable case number from a corpus filename.

    Filenames are `<Court>__<CaseNumber-with-underscores>.docx`
    (see corpus/download_court_decisions.py) -- e.g.
    'Bekesi_Jarasbirosag__P_20099_2020_92.docx' -> 'P.20099.2020.92'.
    Best-effort only: used for prompt context, not for the citation-existence
    check itself, which matches against the real source_file directly.
    """
    stem = Path(source_file).stem
    _, _, case_part = stem.partition("__")
    return case_part.replace("_", ".") if case_part else stem


def sample_chunks_for_persona(persona: dict, count: int = 1) -> list[dict]:
    """Pull real chunk_index=0 content for `count` random documents.

    A document's first chunk reliably contains the case header (court,
    case number, parties, subject) -- see the real examples used to seed
    corpus/data/questions.json's q0001/q0002. Good enough for
    single-document personas; multi-document personas (precedent_seeker,
    synthesizer) get `count` independent documents to compare/connect, not
    one document with more chunks.

    Args:
        persona: A persona dict from personas.json.
        count: Number of distinct documents to sample.

    Returns:
        A list of ``{"source_file", "court", "case_number", "content"}`` dicts.
    """
    from db import get_connection

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT metadata->>'source_file', content
                FROM document_chunks
                WHERE metadata->>'chunk_index' = '0'
                ORDER BY random()
                LIMIT %s;
                """,
                (count,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    samples = []
    for source_file, content in rows:
        court = source_file.split("__")[0].replace("_", " ") if source_file else ""
        samples.append(
            {
                "source_file": source_file,
                "court": court,
                "case_number": _case_number_from_filename(source_file),
                "content": content,
            }
        )
    return samples


def _extract_json(text: str) -> dict:
    """Parse the first JSON object found in an LLM response.

    Models sometimes wrap JSON in ```` ```json ... ``` ```` fences despite
    being asked not to -- strip those before parsing rather than failing.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped
        stripped = stripped.rsplit("```", 1)[0]
    return json.loads(stripped)


def draft_question(persona: dict, samples: list[dict]) -> dict:
    """Ask LLM_DRIVER to draft one question, following SKILL.md Part 1.

    Args:
        persona: The persona dict driving question style.
        samples: Real chunks to ground the question in (empty for
            ``adversarial``, which fabricates a nonexistent reference instead).

    Returns:
        The drafted question dict (question/expected_answer/citations/
        difficulty/notes), not yet verified.
    """
    from drivers.llm import get_answer_driver

    skill_text = (SKILL_DIR / "SKILL.md").read_text()
    part1 = skill_text.split("## Part 2", 1)[0]

    prompt = (
        f"{part1}\n\n"
        f"Persona:\n{json.dumps(persona, ensure_ascii=False, indent=2)}\n\n"
        f"Real content to ground the question in "
        f"(empty on purpose for the adversarial persona):\n"
        f"{json.dumps(samples, ensure_ascii=False, indent=2)}\n"
    )

    driver = get_answer_driver()
    response = driver.run_tool_calling_turn([{"role": "user", "content": prompt}])
    return _extract_json(response.content or "")


def verify_citation_exists(citation: dict) -> bool:
    """Tier 1: does this (court, case_number, source_file) match a real ingested document?

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


def _fetch_full_content(source_file: str) -> str:
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


def verify_content_support(drafted: dict) -> tuple[str, str]:
    """Tier 2: does each citation's real content actually support the drafted answer?

    A *separate* LLM_DRIVER call from draft_question() -- never lets a
    generation pass grade its own work (see SKILL.md Part 2).

    Args:
        drafted: The dict returned by :func:`draft_question`.

    Returns:
        A ``(verdict, reason)`` tuple, verdict one of
        "SUPPORTED"/"NOT_SUPPORTED"/"UNCLEAR".
    """
    from drivers.llm import get_answer_driver

    if not drafted.get("citations"):
        # Adversarial persona: no real content to check against -- the
        # citation-existence check (tier 1) already confirmed there's
        # nothing to find, which is the whole point.
        return "SUPPORTED", "Adversarial question: no citation to verify by design."

    full_contents = {
        c["source_file"]: _fetch_full_content(c["source_file"])
        for c in drafted["citations"]
    }

    skill_text = (SKILL_DIR / "SKILL.md").read_text()
    part2 = skill_text.split("## Part 2", 1)[1]

    prompt = (
        f"## Part 2{part2}\n\n"
        f"Drafted question:\n{json.dumps(drafted, ensure_ascii=False, indent=2)}\n\n"
        f"Real full content of each cited source_file:\n"
        f"{json.dumps(full_contents, ensure_ascii=False, indent=2)}\n"
    )

    driver = get_answer_driver()
    response = driver.run_tool_calling_turn([{"role": "user", "content": prompt}])
    result = _extract_json(response.content or "")
    return result["verdict"], result["reason"]


def generate_one(persona_id: str, personas: dict[str, dict]) -> dict:
    """Draft and verify one golden question for the given persona.

    Args:
        persona_id: A key in personas.json.
        personas: Loaded personas, keyed by id.

    Returns:
        A questions.json-shaped entry, with verification_status/
        verification_notes filled in.
    """
    persona = personas[persona_id]
    samples = (
        []
        if persona_id == "adversarial"
        else sample_chunks_for_persona(
            persona, count=2 if persona["document_scope"] == "multi" else 1
        )
    )

    drafted = draft_question(persona, samples)

    missing = [c for c in drafted.get("citations", []) if not verify_citation_exists(c)]
    if missing:
        verdict, reason = (
            "NOT_SUPPORTED",
            f"Citation(s) not found in document_chunks: {[c['source_file'] for c in missing]}",
        )
    else:
        verdict, reason = verify_content_support(drafted)

    status = "verified" if verdict == "SUPPORTED" else "needs_review"

    return {
        "id": None,  # filled in by the caller once appended
        "persona_id": persona_id,
        "question": drafted["question"],
        "expected_answer": drafted["expected_answer"],
        "citations": drafted.get("citations", []),
        "difficulty": drafted.get("difficulty", "medium"),
        "is_answerable": persona_id != "adversarial",
        "reviewed": False,
        "verification_status": status,
        "verification_notes": reason,
        "notes": drafted.get("notes", ""),
    }


@app.command("generate-questions")
def generate_questions(
    persona_id: Annotated[
        str, typer.Argument(help="A persona id from corpus/data/personas.json.")
    ],
    count: Annotated[
        int, typer.Option(help="How many questions to draft and verify.")
    ] = 1,
) -> None:
    """Draft + two-tier-verify N golden questions for one persona.

    Sends skills/generate-golden-questions/SKILL.md's instructions to the
    project's own LLM_DRIVER, then verifies each draft (citation existence +
    a separate content-support check) before appending it to
    corpus/data/questions.json. See SKILL.md for the full design.
    """
    personas = load_personas()
    if persona_id not in personas:
        print(f"Unknown persona_id: '{persona_id}'. Options: {list(personas)}")
        raise typer.Exit(code=1)

    questions = load_questions()
    for _ in range(count):
        entry = generate_one(persona_id, personas)
        entry["id"] = next_question_id(questions)
        questions.append(entry)
        save_questions(questions)
        print(
            f"[{entry['id']}] {entry['verification_status']}: {entry['question'][:80]}"
        )
