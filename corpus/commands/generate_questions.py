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

A verified question's citations also get a word-for-word supporting ``quote``
(corpus/golden_quotes.py); one that cannot be quoted is "needs_review".

This unattended path is expected to be noisier than drafting the question
interactively inside Claude Code (see SKILL.md) -- that's exactly why
nothing here is trusted without the verification step.
"""

import json
import random
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from corpus.verification import (
    case_number_in_corpus,
    case_numbers_in,
    extract_json,
    fetch_chunk_bodies,
    fetch_full_content,
    other_document_with_number,
    verify_citation_exists,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PERSONAS_PATH = DATA_DIR / "personas.json"
QUESTIONS_PATH = DATA_DIR / "questions.json"
SKILL_DIR = PROJECT_ROOT / "skills" / "generate-golden-questions"

#: The procedural topics a practical_procedural question is about; one is drawn per question
#: so the questions do not all collapse onto the costs of the proceedings.
PROCEDURAL_TOPICS = (
    "which evidence the court accepted or rejected, and why",
    "a deadline, a time limit or a limitation period the court applied",
    "why the court had (or did not have) jurisdiction or competence",
    "who bears the costs of the proceedings and what the court based that on",
    "whether and why the court appointed an expert, and what the expert established",
    "an objection of a party the court dealt with on procedural grounds",
    "what the court said about the burden of proof",
)

#: The longest expected answer a generated question may have (key facts only).
MAX_ANSWER_CHARS = 320

#: Plausible traps for the adversarial persona; the right behaviour is to decline.
TRAP_KINDS = ("invented_case_number", "missing_fact")

app = typer.Typer()


def load_personas() -> dict[str, dict]:
    """Load corpus/data/personas.json, keyed by persona id."""
    personas = json.loads(PERSONAS_PATH.read_text())
    return {p["id"]: p for p in personas}


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    """Load the questions file (default corpus/data/questions.json; empty list if missing)."""
    if not path.exists():
        return []
    return json.loads(path.read_text())


def save_questions(questions: list[dict], path: Path = QUESTIONS_PATH) -> None:
    """Write the questions file (default corpus/data/questions.json), pretty-printed."""
    path.write_text(json.dumps(questions, indent=2, ensure_ascii=False) + "\n")


def append_question(entry: dict, path: Path = QUESTIONS_PATH) -> list[dict]:
    """Give ``entry`` the next id and append it, under a lock on the file.

    The file is read again inside the lock, so two generator runs writing the same file
    (e.g. two personas in parallel) add to each other's questions instead of overwriting.

    Returns:
        All questions in the file after the append.
    """
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(path.suffix + ".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        questions = load_questions(path)
        entry["id"] = next_question_id(questions)
        questions.append(entry)
        save_questions(questions, path)
    return questions


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


def sample_identifiable_documents(count: int = 1) -> list[dict]:
    """Random documents that court, year and subject identify uniquely.

    A question that names a court, a year and what the lawsuit is about, but not the case
    number, points at exactly one document only if no other document shares those three
    values. The metadata layer (``issuing_body``, ``decision_date``, ``case_subject_raw``)
    says which documents qualify; about half of the corpus does.

    Returns:
        ``{"source_file", "court", "year", "subject", "content"}`` dicts, ``content`` being
        the document's whole own text.
    """
    from db import get_connection

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH facts AS (
                    SELECT d.source_file,
                           max(m.value_text) FILTER (WHERE m.key = 'issuing_body') AS court,
                           max(extract(year FROM m.value_date)::int)
                               FILTER (WHERE m.key = 'decision_date') AS year,
                           max(m.value_text) FILTER (WHERE m.key = 'case_subject_raw') AS subject
                    FROM documents d JOIN document_meta m ON m.document_id = d.id
                    GROUP BY d.id, d.source_file
                ), counted AS (
                    SELECT *, count(*) OVER (PARTITION BY court, year, subject) AS same
                    FROM facts
                    WHERE court IS NOT NULL AND year IS NOT NULL AND subject IS NOT NULL
                )
                SELECT source_file, court, year, subject FROM counted
                WHERE same = 1 ORDER BY random() LIMIT %s;
                """,
                (count,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return [
        {
            "source_file": source_file,
            "court": court,
            "year": year,
            "subject": subject,
            "content": fetch_full_content(source_file),
        }
        for source_file, court, year, subject in rows
    ]


def condense_answer(
    question: str, answer: str, ask: Callable[[str], str] | None = None
) -> str | None:
    """The expected answer shortened to the key facts the question asks for, or ``None``.

    The drafting model keeps writing background and commentary past the length it is
    given (28 of 85 practical drafts), and a rejected draft is a wasted draft. The
    shortened answer is checked like any other afterwards (grounded, answers the question).

    Args:
        question: The drafted question.
        answer: The drafted, too long expected answer.
        ask: Takes a prompt and returns the model's reply (default: the answer driver).

    Returns:
        The shortened answer, or ``None`` when the reply is empty or still too long.
    """

    def driver_ask(prompt: str) -> str:
        from drivers.llm import get_answer_driver

        reply = get_answer_driver().run_tool_calling_turn(
            [{"role": "user", "content": prompt}], temperature=0.0
        )
        return reply.content or ""

    short = (
        (ask or driver_ask)(
            "Shorten the EXPECTED ANSWER to the key facts the QUESTION asks for: one or two "
            f"sentences, at most {MAX_ANSWER_CHARS - 40} characters, in the same language. Use "
            "only facts already in the expected answer; add nothing; drop background and "
            f"commentary.\n\nQUESTION: {question}\n\nEXPECTED ANSWER: {answer}\n\n"
            "Output only the shortened answer."
        )
        .strip()
        .strip('"')
    )
    if not short or answer_too_long(short):
        return None
    return short


def answer_too_long(expected_answer: str) -> str | None:
    """Why an expected answer is not "key facts only", or ``None`` if it is short enough."""
    if len(expected_answer) <= MAX_ANSWER_CHARS:
        return None
    return (
        f"The expected answer has {len(expected_answer)} characters; key facts only, "
        f"at most {MAX_ANSWER_CHARS}."
    )


def names_another_document(
    question: str, citations: list[dict], find=other_document_with_number
) -> str | None:
    """A case number in ``question`` that belongs to a document that is not cited, or ``None``.

    A decision names the decision it reviews; a question that asks about the appeal "in the
    9.G.40.230/2020/32 case" using the first-instance number is about the other document,
    and the drafter had no way to tell (the cited text does contain that number).
    """
    cited = {c["source_file"] for c in citations}
    for number in case_numbers_in(question):
        owner = find(number, "")
        if owner and owner not in cited:
            return f"The case number {number} in the question belongs to {owner}."
    return None


def identifies_by_content(
    question: str, sample: dict, citations: list[dict]
) -> str | None:
    """Why ``question`` does not point at ``sample`` by court, year and subject, or ``None``.

    The question must name the court and the year and must not contain a case number: the
    point of this persona is a document found by what it is about.
    """
    text = _alnum(question).casefold()
    if _alnum(sample["court"]).casefold() not in text:
        return f"The question does not name the court ({sample['court']})."
    if str(sample["year"]) not in question:
        return f"The question does not name the year ({sample['year']})."
    for number in [c.get("case_number", "") for c in citations]:
        if number and _alnum(number).casefold() in text:
            return f"The question contains the case number {number}."
    return None


def draft_question(
    persona: dict,
    samples: list[dict],
    trap_kind: str | None = None,
    topic: str | None = None,
) -> dict:
    """Ask LLM_DRIVER to draft one question, following SKILL.md Part 1.

    Args:
        persona: The persona dict driving question style.
        samples: Real chunks to ground the question in.
        trap_kind: For ``adversarial`` only: one of :data:`TRAP_KINDS`.
        topic: For ``practical_procedural`` only: one of :data:`PROCEDURAL_TOPICS`.

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
        + (f"trap_kind: {trap_kind}\n\n" if trap_kind else "")
        + (f"procedural topic: {topic}\n\n" if topic else "")
        + "Real content to ground the question in:\n"
        f"{json.dumps(samples, ensure_ascii=False, indent=2)}\n"
    )

    driver = get_answer_driver()
    response = driver.run_tool_calling_turn([{"role": "user", "content": prompt}])
    return extract_json(response.content or "")


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
        c["source_file"]: fetch_full_content(c["source_file"])
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
    result = extract_json(response.content or "")
    return result["verdict"], result["reason"]


def _alnum(text: str) -> str:
    """``text`` reduced to its letters and digits, for punctuation-blind comparison."""
    return "".join(ch for ch in text if ch.isalnum())


def judge_answer_fits(question: str, answer: str, content: str) -> tuple[str, str]:
    """Strict check of a drafted pair against the whole document, in two parts.

    The first content check ("is the answer supported in substance?") let through
    questions whose answer is true of the document but does not answer what was asked
    (a question about the burden of proof answered with the definition of an interest), and
    one whose answer was not in the document at all. This call asks the two things apart.

    Returns:
        ``("SUPPORTED", reason)`` or ``("NOT_SUPPORTED", reason)``.
    """
    from drivers.llm import get_answer_driver

    prompt = (
        "Check a question and its expected answer against the court decision below. "
        "Answer two things separately.\n"
        "1. ANSWERS: does the expected answer actually answer what the question asks "
        "(not just something related to the topic)?\n"
        "2. GROUNDED: is EVERY statement of the expected answer stated in the decision "
        "itself? One statement that is not in the decision makes it false.\n"
        "4. NOT_ABSENCE: is the expected answer more than the statement that something "
        "did not happen or is not mentioned (e.g. that no expert was appointed)? A text that "
        "is silent cannot show that, so such a question cannot be graded.\n"
        "3. PREMISES: does the question presuppose anything the decision does not say "
        "(e.g. that the court appointed an expert, ordered a measure or ruled on a point "
        "the decision never mentions)? A question built on a false premise is wrong even "
        "if the answer repeats it.\n\n"
        f"QUESTION\n{question}\n\nEXPECTED ANSWER\n{answer}\n\nDECISION\n{content}\n\n"
        'Output ONLY this JSON: {"answers": true | false, "grounded": true | false, '
        '"premises_hold": true | false, "not_absence": true | false, '
        '"reason": "<one sentence naming what fails, if anything>"}'
    )
    response = get_answer_driver().run_tool_calling_turn(
        [{"role": "user", "content": prompt}], temperature=0.0
    )
    result = extract_json(response.content or "")
    if (
        result.get("answers") is True
        and result.get("grounded") is True
        and result.get("premises_hold") is True
        and result.get("not_absence") is True
    ):
        return "SUPPORTED", str(result.get("reason", ""))
    return "NOT_SUPPORTED", f"answer fit: {result.get('reason', '')}"


def candidate_documents(sample: dict) -> list[dict]:
    """The documents of the sample's court and year, with what the metadata says of each.

    These are the documents a person could mean by "that court, that year, that kind of
    lawsuit", i.e. what the question has to tell apart.

    Returns:
        ``{"source_file", "subject", "summary"}`` dicts, the sample itself included.
    """
    from db import get_connection

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT d.source_file, d.summary,
                       max(m.value_text) FILTER (WHERE m.key = 'case_subject_raw')
                FROM documents d JOIN document_meta m ON m.document_id = d.id
                GROUP BY d.id, d.source_file, d.summary
                HAVING max(m.value_text) FILTER (WHERE m.key = 'issuing_body') = %s
                   AND max(extract(year FROM m.value_date)::int)
                       FILTER (WHERE m.key = 'decision_date') = %s
                """,
                (sample["court"], sample["year"]),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return [
        {"source_file": sf, "subject": subject, "summary": (summary or "")[:400]}
        for sf, summary, subject in rows
    ]


def judge_which_documents_fit(question: str, candidates: list[dict]) -> list[str]:
    """Ask which candidate documents the question's description could be about.

    Returns:
        The ``source_file`` of every candidate the description fits.
    """
    from drivers.llm import get_answer_driver

    items = [
        {"id": i, "subject": c["subject"], "summary": c["summary"]}
        for i, c in enumerate(candidates, start=1)
    ]
    prompt = (
        "A person asks the question below about ONE court decision, without knowing its "
        "case number. Here are the decisions of that court and year. List the ids of ALL "
        "decisions the question's description could be about (its kind of lawsuit and "
        "the details it states); a decision the description clearly does not fit is "
        "not listed.\n\n"
        f"QUESTION\n{question}\n\nDECISIONS\n{json.dumps(items, ensure_ascii=False)}\n\n"
        'Output ONLY this JSON: {"fits": [<id>, ...]}'
    )
    response = get_answer_driver().run_tool_calling_turn(
        [{"role": "user", "content": prompt}], temperature=0.0
    )
    fits = extract_json(response.content or "").get("fits", [])
    return [
        candidates[i - 1]["source_file"]
        for i in fits
        if isinstance(i, int) and 1 <= i <= len(candidates)
    ]


def ambiguous_description(
    question: str,
    sample: dict,
    candidates=candidate_documents,
    judge=judge_which_documents_fit,
) -> str | None:
    """Why the question's description fits another document too, or ``None``.

    Court, year and subject identify the sampled document in the metadata, but the
    question paraphrases them, and a vague paraphrase ("a 2022 administrative review")
    fits several decisions of that court and year.
    """
    fitting = judge(question, candidates(sample))
    others = [f for f in fitting if f != sample["source_file"]]
    if sample["source_file"] not in fitting:
        return "The question's description does not fit its own document."
    if others:
        return f"The description also fits {len(others)} other document(s), e.g. {others[0]}."
    return None


def judge_fact_absent(drafted: dict, content: str) -> tuple[str, str]:
    """Ask a separate call whether the detail a ``missing_fact`` trap asks for is absent.

    Returns:
        ``(verdict, reason)``, verdict one of "ABSENT"/"PRESENT"/"UNCLEAR".
    """
    from drivers.llm import get_answer_driver

    prompt = (
        "A trap question asks about a detail a court decision does NOT state. Read the "
        "WHOLE decision and say whether the question could be answered from it: PRESENT "
        "if the detail is stated in any wording or a closely related detail that a "
        "reader would take as the answer is (a date of the judgment for a question "
        "about when it was decided, an amount for a question about costs); ABSENT only "
        "if nothing in the decision answers it.\n\n"
        f"Question: {drafted['question']}\n\nDECISION\n{content}\n\n"
        'Output ONLY this JSON: {"verdict": "ABSENT" | "PRESENT" | "UNCLEAR", '
        '"reason": "<one sentence>"}'
    )
    response = get_answer_driver().run_tool_calling_turn(
        [{"role": "user", "content": prompt}], temperature=0.0
    )
    result = extract_json(response.content or "")
    return result["verdict"], result["reason"]


def verify_trap(
    drafted: dict,
    trap_kind: str,
    sample: dict,
    exists=case_number_in_corpus,
    judge=judge_fact_absent,
) -> tuple[str, str]:
    """Is an adversarial draft really a trap of its kind?

    Args:
        drafted: The drafted question (``invented_case_number`` needs ``trap_case_number``).
        trap_kind: One of :data:`TRAP_KINDS`.
        sample: The real document the draft was grounded in (``content``, ``case_number``).
        exists: Whether a case number occurs in the corpus (injectable for tests).
        judge: Whether the asked detail is absent from a document (injectable for tests).

    Returns:
        ``("SUPPORTED", reason)`` when the draft is a valid trap, otherwise
        ``("NOT_SUPPORTED", reason)``.
    """
    if drafted.get("citations"):
        return "NOT_SUPPORTED", "An adversarial question must cite nothing."
    if trap_kind == "invented_case_number":
        number = str(drafted.get("trap_case_number") or "").strip()
        if not number or _alnum(number) not in _alnum(drafted.get("question", "")):
            return "NOT_SUPPORTED", "The invented case number is not in the question."
        if exists(number):
            return "NOT_SUPPORTED", f"The case number {number} exists in the corpus."
        return "SUPPORTED", f"The case number {number} occurs in no stored document."
    verdict, reason = judge(drafted, sample["content"])
    if verdict == "ABSENT":
        return "SUPPORTED", reason
    return "NOT_SUPPORTED", f"{verdict}: {reason}"


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
    topic = (
        random.choice(PROCEDURAL_TOPICS)
        if persona_id == "practical_procedural"
        else None
    )
    samples = (
        sample_identifiable_documents()
        if topic
        else sample_chunks_for_persona(
            persona, count=2 if persona["document_scope"] == "multi" else 1
        )
    )
    trap_kind = random.choice(TRAP_KINDS) if persona_id == "adversarial" else None
    if trap_kind == "missing_fact":
        # The judge needs the whole decision, not just its first chunk.
        samples[0]["content"] = fetch_full_content(samples[0]["source_file"])

    drafted = draft_question(persona, samples, trap_kind, topic)

    if persona_id != "adversarial" and answer_too_long(drafted["expected_answer"]):
        shortened = condense_answer(drafted["question"], drafted["expected_answer"])
        if shortened:
            drafted["expected_answer_long"] = drafted["expected_answer"]
            drafted["expected_answer"] = shortened

    missing = [c for c in drafted.get("citations", []) if not verify_citation_exists(c)]
    if trap_kind:
        verdict, reason = verify_trap(drafted, trap_kind, samples[0])
    elif missing:
        verdict, reason = (
            "NOT_SUPPORTED",
            f"Citation(s) not found in document_chunks: {[c['source_file'] for c in missing]}",
        )
    elif persona_id != "adversarial" and (
        too_long := answer_too_long(drafted["expected_answer"])
    ):
        verdict, reason = "NOT_SUPPORTED", too_long
    elif persona_id != "adversarial" and (
        other := names_another_document(
            drafted["question"], drafted.get("citations", [])
        )
    ):
        verdict, reason = "NOT_SUPPORTED", other
    elif topic and (
        problem := identifies_by_content(
            drafted["question"], samples[0], drafted.get("citations", [])
        )
        or ambiguous_description(drafted["question"], samples[0])
    ):
        verdict, reason = "NOT_SUPPORTED", problem
    else:
        verdict, reason = verify_content_support(drafted)
        if verdict == "SUPPORTED":
            contents = "\n\n".join(
                f"[{c['source_file']}]\n{fetch_full_content(c['source_file'])}"
                for c in drafted.get("citations", [])
            )
            verdict, reason = judge_answer_fits(
                drafted["question"], drafted["expected_answer"], contents
            )

    status = "verified" if verdict == "SUPPORTED" else "needs_review"

    entry = {
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
    if trap_kind:
        entry["trap_kind"] = trap_kind
        if trap_kind == "invented_case_number":
            entry["trap_case_number"] = drafted.get("trap_case_number")
    if status == "verified":
        quote_citations(entry)
    return entry


def quote_citations(entry: dict, fetch=fetch_chunk_bodies, ask=None) -> None:
    """Give a verified question's citations a code-verified supporting ``quote``.

    A citation that gets none (the model could not quote the document word for word)
    makes the question ``needs_review``: without a quote nothing can be said about it at
    section level.

    Args:
        entry: A questions.json-shaped entry; changed in place.
        fetch: ``source_file`` to the document's ``(chunk_index, own text)`` list.
        ask: Asks the model for a passage (default: the answer driver).
    """
    from corpus.golden_quotes import add_quotes, llm_ask

    report = add_quotes([entry], fetch, ask or llm_ask)
    if report.failed:
        entry["verification_status"] = "needs_review"
        entry["verification_notes"] = (
            "No word-for-word supporting quote found for: "
            + ", ".join(source for _, source, _ in report.failed)
        )


@app.command("generate-questions")
def generate_questions(
    persona_id: Annotated[
        str, typer.Argument(help="A persona id from corpus/data/personas.json.")
    ],
    count: Annotated[
        int, typer.Option(help="How many questions to draft and verify.")
    ] = 1,
    questions_file: Annotated[
        Path,
        typer.Option(
            help="The questions file to append to (default: corpus/data/questions.json)."
        ),
    ] = QUESTIONS_PATH,
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

    for _ in range(count):
        entry = generate_one(persona_id, personas)
        append_question(entry, questions_file)
        print(
            f"[{entry['id']}] {entry['verification_status']}: {entry['question'][:80]}"
        )
