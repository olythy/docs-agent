"""The `eval` command: golden-set evaluation (persona-bucketed accuracy + citation correctness).

Measures the two pipeline stages separately, since a single pass/fail number
can't tell you which one to tune:
    - retrieval_hit_rate: did the correct source document even make it into
      the retrieved/reranked context? If not, no answer-generation fix can
      help -- this is what to check before touching RETRIEVAL_MIN_SCORE,
      RERANKER_MIN_SCORE, chunking, etc.
    - answer_accuracy: given that context, did the generated answer actually
      convey the expected facts? This is the metric the >95%-per-persona
      target is measured against.
    - citation_accuracy: does the answer's stated citation actually point to
      the *correct* source (not just *a* real one)? Tracked separately from
      answer_accuracy on purpose -- a fluent, wrong-citation answer is a
      worse failure mode in a legal domain than an honestly incomplete one.
    - decline_correct_rate: for the adversarial persona only -- did the
      system honestly decline instead of fabricating an answer?

Grading (answer_accuracy/citation_accuracy) is a *separate* LLM_DRIVER call
from whatever generated the answer -- same "never grade your own work"
principle as corpus/commands/generate_questions.py's tier-2 verification.
Only questions with verification_status="verified" and reviewed=true are
evaluated -- an unverified or needs-review golden question isn't a
trustworthy yardstick yet.
"""

import json
import sys
from pathlib import Path
from typing import Annotated

import typer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
QUESTIONS_PATH = DATA_DIR / "questions.json"

app = typer.Typer()


def load_verified_questions(persona_filter: str | None = None) -> list[dict]:
    """Load corpus/data/questions.json, keeping only trustworthy entries.

    Args:
        persona_filter: If given, only questions for this persona_id.

    Returns:
        Questions with verification_status="verified" and reviewed=true --
        anything still "needs_review"/"unverified" isn't a trustworthy
        yardstick yet.
    """
    questions = json.loads(QUESTIONS_PATH.read_text())
    questions = [
        q
        for q in questions
        if q.get("verification_status") == "verified" and q.get("reviewed") is True
    ]
    if persona_filter:
        questions = [q for q in questions if q["persona_id"] == persona_filter]
    return questions


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


def check_retrieval_hit(question: dict, retrieved_source_files: set[str]) -> bool:
    """Did every expected citation's source_file make it into the retrieved context?

    Args:
        question: A questions.json entry (with real citations -- not adversarial).
        retrieved_source_files: The source_file values of the chunks
            actually retrieved for this question.

    Returns:
        True if every cited source_file was retrieved.
    """
    expected = {c["source_file"] for c in question["citations"]}
    return expected.issubset(retrieved_source_files)


def grade_answer(question: dict, generated_answer: str) -> dict:
    """Judge a generated answer against the golden question -- a separate
    LLM_DRIVER call from whichever call produced the answer.

    Args:
        question: The golden questions.json entry (question/expected_answer/citations).
        generated_answer: What query_knowledge_base() actually returned.

    Returns:
        A dict with ``answer_correct``/``citation_correct`` (bool) and
        ``reason`` (str).
    """
    from drivers.llm import get_answer_driver

    prompt = (
        "You are grading a RAG system's answer against a gold-standard "
        "reference, for a Hungarian real-estate-law question-answering "
        "evaluation. Judge two things independently:\n\n"
        "1. answer_correct: does the system's answer convey the same key "
        "facts as the expected answer? Paraphrasing is fine, this is not a "
        "string-match check.\n"
        "2. citation_correct: does the system's answer's stated citation(s) "
        "(court/case number mentioned in its text) correctly match the "
        "expected citations below? A missing citation where one was "
        "expected, or a citation to the wrong case, is NOT correct.\n\n"
        f"Question: {question['question']}\n\n"
        f"Expected answer: {question['expected_answer']}\n\n"
        f"Expected citations: {json.dumps(question['citations'], ensure_ascii=False)}\n\n"
        f"System's actual generated answer:\n{generated_answer}\n\n"
        "Output exactly this JSON shape, no surrounding prose: "
        '{"answer_correct": true/false, "citation_correct": true/false, "reason": "..."}'
    )

    driver = get_answer_driver()
    response = driver.run_tool_calling_turn([{"role": "user", "content": prompt}])
    return _extract_json(response.content or "")


def evaluate_one(question: dict, strategy_name: str) -> dict:
    """Run one golden question through the real pipeline and grade the result.

    Args:
        question: A verified questions.json entry.
        strategy_name: "vector" or "hybrid" -- which RetrievalStrategy to use.

    Returns:
        A dict with this question's raw results (retrieval_hit,
        answer_correct, citation_correct, is_decline -- whichever apply).
    """
    from query.decline_detection import looks_like_a_decline
    from query.retrieval import (
        HybridRetrievalStrategy,
        VectorRetrievalStrategy,
        query_knowledge_base,
        retrieve_chunks,
    )

    strategy = (
        HybridRetrievalStrategy()
        if strategy_name == "hybrid"
        else VectorRetrievalStrategy()
    )

    retrieved = retrieve_chunks(question["question"], strategy=strategy)
    retrieved_source_files = {c.metadata.source_file for c in retrieved}

    answer = query_knowledge_base(question["question"], strategy=strategy)

    result: dict = {"persona_id": question["persona_id"]}

    if question["persona_id"] == "adversarial":
        result["is_decline"] = looks_like_a_decline(answer)
        return result

    result["retrieval_hit"] = check_retrieval_hit(question, retrieved_source_files)
    grade = grade_answer(question, answer)
    result["answer_correct"] = bool(grade.get("answer_correct"))
    result["citation_correct"] = bool(grade.get("citation_correct"))
    return result


def _rate(results: list[dict], key: str) -> float | None:
    """Fraction of results where results[key] is True, or None if the key never applies."""
    applicable = [r[key] for r in results if key in r]
    if not applicable:
        return None
    return sum(applicable) / len(applicable)


def print_report(results: list[dict]) -> None:
    """Print a persona-bucketed accuracy/citation-correctness report."""
    personas = sorted({r["persona_id"] for r in results})

    header = f"{'persona':<22}{'n':>4}  {'retrieval':>10}  {'answer':>8}  {'citation':>9}  {'decline':>8}"
    print(header)
    print("-" * len(header))

    for persona_id in personas:
        bucket = [r for r in results if r["persona_id"] == persona_id]
        retrieval = _rate(bucket, "retrieval_hit")
        answer = _rate(bucket, "answer_correct")
        citation = _rate(bucket, "citation_correct")
        decline = _rate(bucket, "is_decline")

        def fmt(rate: float | None) -> str:
            return "-" if rate is None else f"{rate:.0%}"

        flag = ""
        if answer is not None and answer < 0.95:
            flag = "  <-- below 95% target"

        print(
            f"{persona_id:<22}{len(bucket):>4}  {fmt(retrieval):>10}  {fmt(answer):>8}  "
            f"{fmt(citation):>9}  {fmt(decline):>8}{flag}"
        )


@app.command()
def eval(
    persona: Annotated[
        str | None,
        typer.Option(help="Only evaluate this persona_id (default: all)."),
    ] = None,
    strategy: Annotated[
        str,
        typer.Option(help="Retrieval strategy to evaluate: 'vector' or 'hybrid'."),
    ] = "hybrid",
) -> None:
    """Run the golden-set evaluation: persona-bucketed accuracy + citation correctness.

    Only evaluates questions with verification_status="verified" and
    reviewed=true (see corpus/data/questions.json) -- run
    `generate-questions` first if there aren't enough yet.
    """
    if strategy not in ("vector", "hybrid"):
        raise typer.BadParameter("--strategy must be 'vector' or 'hybrid'")

    questions = load_verified_questions(persona_filter=persona)
    if not questions:
        print(
            "No verified+reviewed questions to evaluate. Run `generate-questions` first."
        )
        raise typer.Exit(code=1)

    print(f"Evaluating {len(questions)} question(s) with strategy={strategy} ...\n")
    results = [evaluate_one(q, strategy) for q in questions]
    print_report(results)
