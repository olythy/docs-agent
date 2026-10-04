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

Grading is a *separate* LLM_DRIVER call from whatever generated the answer
-- same "never grade your own work" principle as
corpus/commands/generate_questions.py's tier-2 verification. Only questions
with verification_status="verified" are evaluated -- an unverified or
needs-review golden question isn't a trustworthy yardstick yet.

*How* a question is graded is itself a Strategy choice, not a single fixed
rule -- see GradingStrategy's docstring and corpus/data/personas.json's
grading_strategies field for why: a question like "which cases involve X
type of ruling" accepts ANY real, correctly-matching document, not just
the one a human happened to sample when drafting it, which the original
fixed source_file-matching logic (ExactMatchGradingStrategy) couldn't
account for -- confirmed live, see docs/decisions.md's 2026-10-03 entry.
"""

import json
import re
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Annotated

import typer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from corpus.verification import (
    extract_json,
    fetch_full_content,
    verify_citation_exists,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PERSONAS_PATH = DATA_DIR / "personas.json"
QUESTIONS_PATH = DATA_DIR / "questions.json"

app = typer.Typer()


def load_personas() -> dict[str, dict]:
    """Load corpus/data/personas.json, keyed by persona id."""
    personas = json.loads(PERSONAS_PATH.read_text())
    return {p["id"]: p for p in personas}


def load_verified_questions(persona_filter: str | None = None) -> list[dict]:
    """Load corpus/data/questions.json, keeping only trustworthy entries.

    Only requires verification_status="verified" -- the two-tier pipeline
    (citation existence + a separate content-support/citation-verbatim
    check) is trusted on its own now that Part 2's verbatim-citation check
    exists (see docs/decisions.md for the live-confirmed bug it was added
    to catch). ``reviewed`` stays in the schema as an optional, stronger
    signal for anyone who does look a question over by hand, but isn't
    required to include a question here anymore.

    Args:
        persona_filter: If given, only questions for this persona_id.

    Returns:
        Questions with verification_status="verified" -- anything still
        "needs_review"/"unverified" isn't a trustworthy yardstick yet.
    """
    questions = json.loads(QUESTIONS_PATH.read_text())
    questions = [q for q in questions if q.get("verification_status") == "verified"]
    if persona_filter:
        questions = [q for q in questions if q["persona_id"] == persona_filter]
    return questions


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
    return extract_json(response.content or "")


_FILE_NAME_PATTERN = re.compile(r"[\w.\-]+\.(?:docx|pdf|md|rtf|txt)", re.IGNORECASE)


def _file_names_cited(generated_answer: str) -> list[str]:
    """File names written in ``generated_answer`` that are real ingested documents.

    Args:
        generated_answer: What query_knowledge_base() actually returned.

    Returns:
        Distinct source_file values, in the order first written. Empty if the
        answer names no ingested file.
    """
    from store import VectorStore

    written = list(dict.fromkeys(_FILE_NAME_PATTERN.findall(generated_answer)))
    if not written:
        return []
    known = VectorStore().get_all_source_files()
    return [name for name in written if name in known]


def _resolve_cited_source_files(generated_answer: str) -> list[str]:
    """Find which real source_files the generated answer actually cites.

    Extracts identifier-like tokens (case numbers, ...) directly from the
    answer's own text and looks each one up against real document
    content -- deterministic, no LLM involved. The same mechanism
    query.retrieval's identifier-rescue already uses at query time
    (extract_identifier_tokens/search_by_identifier), reused here to
    verify what the answer itself claims instead of what a question asks.

    Args:
        generated_answer: What query_knowledge_base() actually returned.

    Returns:
        Distinct source_file values the answer's stated identifiers
        actually resolve to, in the order first encountered. Empty if the
        answer states no identifier, or none of them exist.
    """
    from store import VectorStore, extract_identifier_tokens

    # An answer that names its sources by file name ("Source: X.docx, page 3")
    # has told us exactly what it cites -- use that, exactly. Resolving by
    # identifier-like tokens alone was confirmed noisy: a tolerated fraction
    # ("15/100-ad"), a statute fragment ("(1)-(2)") or an amount in the answer
    # text were matched literally against document *content*, "citing"
    # unrelated courts' documents the answer never mentioned, which the
    # grader then penalised (see docs/decisions.md).
    named = _file_names_cited(generated_answer)
    if named:
        return named

    tokens = extract_identifier_tokens(generated_answer)
    if not tokens:
        return []

    # per_token: with one shared LIMIT, a long document citing one identifier
    # in every chunk could fill all rows and hide the answer's other
    # citation from the grader (same bug as in retrieval, see
    # docs/decisions.md).
    matches = VectorStore().search_by_identifier(tokens, top_k=10, per_token=True)
    seen: list[str] = []
    for m in matches:
        if m.metadata.source_file not in seen:
            seen.append(m.metadata.source_file)
    return seen


def _verify_answer_claim_support(
    question_text: str, generated_answer: str, cited_source_files: list[str]
) -> tuple[str, str]:
    """Does the real content of what the answer cites actually satisfy
    the question's own criteria, and does the answer describe it accurately?

    A *separate* LLM_DRIVER call from whichever call produced the answer
    -- same "never grade your own work" principle as grade_answer() and
    corpus/commands/generate_questions.py's tier-2 verification. Unlike
    grade_answer(), this never compares against a golden expected_answer/
    citations -- the question accepts any real, correctly-matching
    document (see GradingStrategy's docstring), so there's nothing fixed
    to compare against; only the question's own stated criteria matter.

    Args:
        question_text: The golden question's own question text (its
            stated criteria -- topic, outcome type, legal principle --
            is what the cited content must actually satisfy).
        generated_answer: What query_knowledge_base() actually returned.
        cited_source_files: What :func:`_resolve_cited_source_files` found.

    Returns:
        A ``(verdict, reason)`` tuple, verdict one of
        "SUPPORTED"/"NOT_SUPPORTED"/"UNCLEAR".
    """
    from drivers.llm import get_answer_driver

    full_contents = {f: fetch_full_content(f) for f in cited_source_files}

    prompt = (
        "You are verifying a RAG system's answer against real source "
        "documents, for a Hungarian real-estate-law question-answering "
        "evaluation. This question asks about a *category or pattern* of "
        'case (e.g. "which cases involve X type of ruling"), so the '
        "system is allowed to cite ANY real, correctly-matching document "
        "-- not necessarily one specific document a human happened to "
        "sample when a golden reference answer for this question was "
        "originally drafted. Judge two things:\n\n"
        "1. Does the cited document's real content genuinely satisfy the "
        "question's own stated criteria (topic, outcome type, legal "
        "principle)?\n"
        "2. Does the system's answer accurately describe that content -- "
        "no fabricated facts, no contradiction with the real document?\n\n"
        f"Question: {question_text}\n\n"
        f"System's answer:\n{generated_answer}\n\n"
        f"Real content of the document(s) the answer cites:\n"
        f"{json.dumps(full_contents, ensure_ascii=False)}\n\n"
        "Output exactly this JSON shape, no surrounding prose: "
        '{"verdict": "SUPPORTED | NOT_SUPPORTED | UNCLEAR", "reason": "..."}'
    )

    driver = get_answer_driver()
    response = driver.run_tool_calling_turn([{"role": "user", "content": prompt}])
    result = extract_json(response.content or "")
    return result["verdict"], result["reason"]


class GradingStrategy(ABC):
    """Strategy for judging whether a generated answer is correct, given a
    golden question -- selected per persona via corpus/data/personas.json's
    ``grading_strategies`` list (see :func:`get_grading_strategy`).

    Different personas make fundamentally different claims about what
    counts as "correct":
        - ``"exact_match"`` (:class:`ExactMatchGradingStrategy`): the
          question has exactly one correct answer, pinned to the specific
          document(s) sampled when it was drafted (e.g. ``fact_finder``).
          Compares the generated answer and its stated citation(s)
          directly against ``expected_answer``/``citations``.
        - ``"independent_fact"`` (:class:`IndependentFactGradingStrategy`):
          the question accepts ANY real document that satisfies the
          question's own stated criteria, not just the one originally
          sampled (e.g. ``precedent_seeker``, whose own
          ``citation_expectation`` already says "one or more... based on
          content/legal pattern, not region"). Verifies whatever the
          system's own answer actually claims/cites, independent of the
          golden ``citations`` -- confirmed live (see docs/decisions.md's
          2026-10-03 entry) that comparing against one originally-sampled
          document produces false negatives: a retrieved-and-cited
          *different*, equally real and on-topic document scores as
          "wrong" under ``exact_match`` alone.

    A persona can configure more than one strategy at once (run and
    reported separately per strategy) -- useful during a transition
    between two strategies, to compare them directly without re-running
    anything twice.
    """

    @abstractmethod
    def grade(
        self, question: dict, generated_answer: str, retrieved_source_files: set[str]
    ) -> dict:
        """Grade one already-generated answer.

        Args:
            question: The golden questions.json entry.
            generated_answer: What query_knowledge_base() actually returned.
            retrieved_source_files: The source_file values of the chunks
                actually retrieved for this question.

        Returns:
            A dict with ``retrieval_hit``/``answer_correct``/``citation_correct``
            (bool) and ``reason`` (str) -- same shape regardless of strategy,
            so callers (print_report()) don't need to know which one ran.
        """


class ExactMatchGradingStrategy(GradingStrategy):
    """Current/original behavior -- see GradingStrategy's docstring."""

    def grade(
        self, question: dict, generated_answer: str, retrieved_source_files: set[str]
    ) -> dict:
        retrieval_hit = check_retrieval_hit(question, retrieved_source_files)
        grade = grade_answer(question, generated_answer)
        return {
            "retrieval_hit": retrieval_hit,
            "answer_correct": bool(grade.get("answer_correct")),
            "citation_correct": bool(grade.get("citation_correct")),
            "reason": grade.get("reason", ""),
        }


class IndependentFactGradingStrategy(GradingStrategy):
    """For category/pattern questions -- see GradingStrategy's docstring."""

    def grade(
        self, question: dict, generated_answer: str, retrieved_source_files: set[str]
    ) -> dict:
        cited_source_files = _resolve_cited_source_files(generated_answer)
        if not cited_source_files:
            return {
                "retrieval_hit": False,
                "answer_correct": False,
                "citation_correct": False,
                "reason": "No real, existing citation found in the generated answer.",
            }

        missing = [
            f
            for f in cited_source_files
            if not verify_citation_exists({"source_file": f})
        ]
        if missing:
            # Shouldn't happen in practice -- search_by_identifier() only
            # returns rows that already exist -- but checked explicitly
            # rather than assumed, since this is the one thing standing
            # between "the answer cites something real" and trusting it.
            return {
                "retrieval_hit": False,
                "answer_correct": False,
                "citation_correct": False,
                "reason": f"Resolved citation(s) not found in document_chunks: {missing}",
            }

        verdict, reason = _verify_answer_claim_support(
            question["question"], generated_answer, cited_source_files
        )
        correct = verdict == "SUPPORTED"
        retrieval_hit = any(f in retrieved_source_files for f in cited_source_files)
        return {
            "retrieval_hit": retrieval_hit,
            "answer_correct": correct,
            "citation_correct": correct,
            "reason": reason,
        }


def get_grading_strategy(name: str) -> GradingStrategy:
    """Factory function: return the named grading strategy.

    Args:
        name: One of "exact_match"/"independent_fact" (see
            corpus/data/personas.json's grading_strategies field).

    Returns:
        A :class:`GradingStrategy` instance ready to call.

    Raises:
        ValueError: If ``name`` is unknown.
    """
    if name == "exact_match":
        return ExactMatchGradingStrategy()
    if name == "independent_fact":
        return IndependentFactGradingStrategy()
    raise ValueError(
        f"Unknown grading strategy: '{name}'. "
        "Valid options are: 'exact_match', 'independent_fact'."
    )


#: How many documents deep to look when diagnosing a retrieval miss --
#: wide enough to distinguish "just outside the production top_k" (a
#: ranking problem) from "nowhere near" (a real recall gap), without
#: being so wide every miss looks reachable.
DIAGNOSTIC_POOL_SIZE = 50


def _citation_ranks(
    question_text: str, citations: list[dict], strategy
) -> dict[str, int | None]:
    """Find each citation's 1-based document rank in a wide candidate pool.

    Confirmed valuable live (see docs/decisions.md's q0030 investigation):
    a binary retrieval_hit/miss alone meant re-deriving this by hand, one
    question at a time, to tell "ranked 9th, a tuning problem" apart from
    "not in the corpus/pool at all, a different problem." Reuses the real
    production retrieval path (:func:`query.retrieval.retrieve_chunks`)
    at a wider ``top_k`` than production uses, purely for this diagnostic
    -- not a separate, hand-rolled ranking.

    Args:
        question_text: The golden question's text.
        citations: The golden question's ``citations`` list.
        strategy: The same ``RetrievalStrategy`` instance ``evaluate_one()``
            used for the real (production-top_k) retrieval call.

    Returns:
        ``{source_file: rank}`` for each cited source_file, 1-based by
        first distinct-document occurrence, or ``None`` if it doesn't
        appear even within ``DIAGNOSTIC_POOL_SIZE`` documents.
    """
    from query.retrieval import retrieve_chunks

    wide_pool = retrieve_chunks(
        question_text, strategy=strategy, top_k=DIAGNOSTIC_POOL_SIZE
    )
    doc_rank: dict[str, int] = {}
    for chunk in wide_pool:
        doc_rank.setdefault(chunk.metadata.source_file, len(doc_rank) + 1)

    return {c["source_file"]: doc_rank.get(c["source_file"]) for c in citations}


#: How many times a whole question is retried after an API rate limit/transient
#: failure that the drivers' own per-call retries already gave up on, and how
#: long to wait first. Confirmed live: one Vertex 429 on question 25 of 33
#: aborted an entire ~10 minute (paid) eval run.
_QUESTION_RETRIES = 3
_QUESTION_RETRY_WAIT_SECONDS = 45


def _evaluate_one_with_retry(
    question: dict, strategy_name: str, personas: dict[str, dict]
) -> dict:
    """Run :func:`evaluate_one`, waiting out transient API failures between tries.

    Raises:
        TransientAPIError: If the question still fails after every retry.
    """
    import time

    from retry_policy import TransientAPIError

    for attempt in range(1, _QUESTION_RETRIES + 1):
        try:
            return evaluate_one(question, strategy_name, personas)
        except (TransientAPIError, json.JSONDecodeError) as exc:
            if attempt == _QUESTION_RETRIES:
                raise
            # A grader reply that is not JSON even after extract_json's repairs
            # is a one-off of a non-deterministic LLM: asking again is enough, no
            # need to wait. An API rate limit does need a pause.
            wait = (
                _QUESTION_RETRY_WAIT_SECONDS
                if isinstance(exc, TransientAPIError)
                else 0
            )
            print(
                f"  [{question['id']}] {type(exc).__name__} ({exc}); retry "
                f"{attempt + 1}/{_QUESTION_RETRIES}"
                + (f" after {wait}s ..." if wait else " ...")
            )
            time.sleep(wait)
    raise AssertionError("unreachable")


def evaluate_one(question: dict, strategy_name: str, personas: dict[str, dict]) -> dict:
    """Run one golden question through the real pipeline and grade it under
    every grading strategy its persona configures.

    Args:
        question: A verified questions.json entry.
        strategy_name: "vector" or "hybrid" -- which RetrievalStrategy to use.
        personas: Loaded personas.json, keyed by id (see load_personas()).

    Returns:
        A dict with ``persona_id`` and either ``is_decline`` (adversarial
        only -- no citation to grade by any strategy, it's a pure
        decline/no-decline check) or a ``grades`` sub-dict of
        ``{strategy_name: grade_dict}``, one entry per strategy the
        persona configures in ``grading_strategies``.
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

    result: dict = {
        "persona_id": question["persona_id"],
        "question_id": question["id"],
        "question": question["question"],
        "answer": answer,
        "retrieved_source_files": sorted(retrieved_source_files),
        "expected_source_files": [
            c["source_file"] for c in question.get("citations", [])
        ],
    }

    if question["persona_id"] == "adversarial":
        result["is_decline"] = looks_like_a_decline(answer)
        return result

    persona = personas[question["persona_id"]]
    grading_strategy_names = persona.get("grading_strategies", ["exact_match"])
    result["grades"] = {
        name: get_grading_strategy(name).grade(question, answer, retrieved_source_files)
        for name in grading_strategy_names
    }

    # Diagnostic only, not part of grading: find out *where* (if anywhere)
    # a missed citation actually landed, so a miss doesn't require manual
    # re-investigation to tell "ranking problem" apart from "recall gap."
    if any(not g["retrieval_hit"] for g in result["grades"].values()):
        result["citation_ranks"] = _citation_ranks(
            question["question"], question.get("citations", []), strategy
        )
    return result


def _rate(results: list[dict], key: str) -> float | None:
    """Fraction of results where results[key] is True, or None if the key never applies."""
    applicable = [r[key] for r in results if key in r]
    if not applicable:
        return None
    return sum(applicable) / len(applicable)


def print_report(results: list[dict]) -> None:
    """Print a persona+strategy-bucketed accuracy/citation-correctness report.

    One row per ``(persona_id, grading_strategy)`` pair found in
    ``results`` -- a persona configured with more than one
    ``grading_strategies`` entry (see personas.json) gets one row per
    strategy, so comparing two strategies against the same run is
    directly visible without re-running anything. Adversarial has no
    grading strategy at all (it's a pure decline/no-decline check, not a
    citation-correctness one) and gets its own single row instead.
    """

    def fmt(rate: float | None) -> str:
        return "-" if rate is None else f"{rate:.0%}"

    header = (
        f"{'persona':<22}{'strategy':<17}{'n':>4}  {'retrieval':>10}  "
        f"{'answer':>8}  {'citation':>9}  {'decline':>8}"
    )
    print(header)
    print("-" * len(header))

    adversarial_bucket = [r for r in results if "is_decline" in r]
    if adversarial_bucket:
        decline = _rate(adversarial_bucket, "is_decline")
        print(
            f"{'adversarial':<22}{'-':<17}{len(adversarial_bucket):>4}  {'-':>10}  "
            f"{'-':>8}  {'-':>9}  {fmt(decline):>8}"
        )

    graded = [r for r in results if "grades" in r]
    persona_strategy_pairs = sorted(
        {(r["persona_id"], name) for r in graded for name in r["grades"]}
    )
    for persona_id, strategy_name in persona_strategy_pairs:
        bucket = [
            r["grades"][strategy_name]
            for r in graded
            if r["persona_id"] == persona_id and strategy_name in r["grades"]
        ]
        retrieval = _rate(bucket, "retrieval_hit")
        answer = _rate(bucket, "answer_correct")
        citation = _rate(bucket, "citation_correct")

        flag = ""
        if answer is not None and answer < 0.95:
            flag = "  <-- below 95% target"

        print(
            f"{persona_id:<22}{strategy_name:<17}{len(bucket):>4}  {fmt(retrieval):>10}  "
            f"{fmt(answer):>8}  {fmt(citation):>9}  {'-':>8}{flag}"
        )

    _print_report_legend()
    _print_retrieval_miss_diagnostics(graded)


def print_verbose_cases(results: list[dict]) -> None:
    """Print, per question, what the system answered and why it was graded as it was.

    The aggregate table says *how many* questions failed; this says *why*,
    so the failure modes can be told apart: the system declined, it cited
    nothing real, it cited a document that doesn't support the claim, or
    retrieval never surfaced the right document in the first place.
    """
    from query.decline_detection import looks_like_a_decline

    print("\n" + "=" * 90 + "\nPer-question detail (--verbose)\n" + "=" * 90)
    for r in results:
        print(f"\n[{r['question_id']}] ({r['persona_id']}) {r['question']}")
        print(f"  expected docs : {', '.join(r['expected_source_files']) or '-'}")
        print(f"  retrieved docs: {', '.join(r['retrieved_source_files']) or '-'}")
        answer = r["answer"]
        print(f"  declined?     : {'YES' if looks_like_a_decline(answer) else 'no'}")
        cited = _resolve_cited_source_files(answer)
        print(f"  answer cites  : {', '.join(cited) or 'no real, resolvable document'}")
        print(f"  answer        : {answer.strip()[:700]}")
        if "is_decline" in r:
            print(
                f"  graded        : adversarial, declined correctly = {r['is_decline']}"
            )
            continue
        for name, grade in r["grades"].items():
            print(
                f"  graded [{name}]: retrieval={grade['retrieval_hit']} "
                f"answer={grade['answer_correct']} citation={grade['citation_correct']}"
            )
            print(f"      reason: {grade['reason']}")


def _print_report_legend() -> None:
    """Print what each report column and grading strategy means.

    Printed under every report so a number is still interpretable weeks
    later, without having to re-read this module.
    """
    print(
        """
How to read this table:
  n           questions in the row.
  retrieval   did the right document(s) reach the final top_k chunks the LLM sees?
  answer      did the generated answer convey the expected facts?
  citation    does the answer cite the right source?
  decline     adversarial only: did the system honestly refuse instead of inventing?
  <-- below 95% target: answer accuracy under the project's per-persona goal.

Grading strategies (a persona can have two; each gets its own row):
  exact_match       one right answer, pinned to the document(s) cited in the golden
                    question. retrieval = EVERY cited document is in the top_k chunks.
                    answer/citation = an LLM grader compares against the golden answer.
  independent_fact  any real document that satisfies the question counts, not just the
                    one sampled when the question was written. retrieval = at least one
                    document the ANSWER cites was retrieved; answer = citation = a separate
                    LLM confirms the cited document(s) support the answer's claim.
  A low exact_match next to a higher independent_fact usually means the system found
  other valid documents, not that retrieval failed outright.
"""
    )


def _print_retrieval_miss_diagnostics(graded: list[dict]) -> None:
    """Print each retrieval miss's actual document rank, if diagnosed.

    Only questions ``evaluate_one()`` flagged as a miss under at least one
    grading strategy carry a ``citation_ranks`` entry at all -- see its
    docstring for why this is a diagnostic, run at
    ``DIAGNOSTIC_POOL_SIZE``, not the production retrieval call itself.
    Tells apart, at a glance:
        - a rank number: the correct document IS in a wide pool, just not
          reaching the production top_k -- a ranking/tuning problem.
        - "not found": the correct document doesn't surface even at
          DIAGNOSTIC_POOL_SIZE -- a deeper recall problem (or it's simply
          not in the corpus yet, e.g. during a partial/in-progress ingest).
    """
    misses = [r for r in graded if "citation_ranks" in r]
    if not misses:
        return

    print(f"\nRetrieval misses (diagnostic, top {DIAGNOSTIC_POOL_SIZE} CHUNKS):")
    print(
        "  One line per document the missed question cites. 'rank N' = Nth distinct\n"
        "  document within those chunks; 'not found' = not in them at all (or not\n"
        "  ingested). A 'rank 1' line only means THIS cited document was found -- the\n"
        "  question is a miss because another cited document was not."
    )
    for r in misses:
        for source_file, rank in r["citation_ranks"].items():
            where = f"rank {rank}" if rank is not None else "not found"
            print(f"  {r['question_id']} ({r['persona_id']}): {source_file} -- {where}")


def select_questions(questions: list[dict], ids: list[str] | None) -> list[dict]:
    """Keep only the questions whose ``id`` is in ``ids`` (all of them if ``ids`` is empty).

    Args:
        questions: Verified questions.json entries.
        ids: Question ids such as ``["q0010", "q0012"]``.

    Returns:
        The selected questions, in the order ``ids`` lists them.

    Raises:
        typer.BadParameter: If an id is not among ``questions`` -- a typo, or a
            question that isn't verified (or was filtered out by --persona).
    """
    if not ids:
        return questions
    by_id = {q["id"]: q for q in questions}
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        raise typer.BadParameter(
            f"unknown question id(s) {unknown}; evaluable ids: {sorted(by_id)}"
        )
    return [by_id[i] for i in dict.fromkeys(ids)]


def print_repeat_summary(results: list[dict], target: float = 0.95) -> None:
    """Print, per question, in how many of its runs each check passed.

    A single run of one question is either 0% or 100%, so "does this question
    reach 95%?" only means something over repeated runs (the answer LLM and
    the grader are not deterministic). Shown per grading strategy because a
    persona can have two.

    Args:
        results: Every run's :func:`evaluate_one` result (a question appears
            once per repeat).
        target: Pass rate a question must reach to not be flagged.
    """
    from query.decline_detection import looks_like_a_decline

    by_question: dict[str, list[dict]] = {}
    for r in results:
        by_question.setdefault(r["question_id"], []).append(r)

    print("\n" + "=" * 90)
    print(f"Per-question pass rate over repeated runs (target {target:.0%})")
    print("=" * 90)
    for qid, runs in by_question.items():
        n = len(runs)
        refused = sum(looks_like_a_decline(r["answer"]) for r in runs)
        print(f"\n{qid} ({runs[0]['persona_id']}), {n} run(s); refused {refused}/{n}")
        if "is_decline" in runs[0]:
            passed = sum(r["is_decline"] for r in runs)
            flag = "" if passed / n >= target else "   <-- below target"
            print(f"  declined correctly: {passed}/{n}{flag}")
            continue
        for name in runs[0]["grades"]:
            answer_ok = sum(r["grades"][name]["answer_correct"] for r in runs)
            retrieval_ok = sum(r["grades"][name]["retrieval_hit"] for r in runs)
            flag = "" if answer_ok / n >= target else "   <-- below target"
            print(
                f"  [{name}] answer correct {answer_ok}/{n} | "
                f"retrieval hit {retrieval_ok}/{n}{flag}"
            )


@app.command()
def eval(
    persona: Annotated[
        str | None,
        typer.Option(help="Only evaluate this persona_id (default: all)."),
    ] = None,
    question: Annotated[
        list[str] | None,
        typer.Option(
            "--question",
            "-q",
            help=(
                "Only evaluate this question id, e.g. -q q0010; repeat the option "
                "for several (default: all)."
            ),
        ),
    ] = None,
    repeat: Annotated[
        int,
        typer.Option(
            min=1,
            help=(
                "Run every selected question this many times and print a "
                "per-question pass rate -- one run of one question is 0% or "
                "100%, so a rate needs repetition (the LLM is not deterministic)."
            ),
        ),
    ] = 1,
    strategy: Annotated[
        str,
        typer.Option(help="Retrieval strategy to evaluate: 'vector' or 'hybrid'."),
    ] = "hybrid",
    only_covered: Annotated[
        bool,
        typer.Option(
            help=(
                "Skip questions whose cited source_file(s) aren't all ingested "
                "yet (see `coverage`) -- during a partial ingest, those would "
                "show up as retrieval misses that are really just missing data."
            ),
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            help=(
                "After the table, print every question with the system's answer, "
                "the documents it retrieved/cited, and the grader's reason -- to "
                "tell failure modes apart (declined / unsupported / wrong document)."
            ),
        ),
    ] = False,
) -> None:
    """Run the golden-set evaluation: persona-bucketed accuracy + citation correctness.

    Only evaluates questions with verification_status="verified" (see
    corpus/data/questions.json) -- run `generate-questions` first if there
    aren't enough yet. Grading strategy per persona comes from
    corpus/data/personas.json's grading_strategies field.
    """
    if strategy not in ("vector", "hybrid"):
        raise typer.BadParameter("--strategy must be 'vector' or 'hybrid'")

    questions = load_verified_questions(persona_filter=persona)
    if not questions:
        print("No verified questions to evaluate. Run `generate-questions` first.")
        raise typer.Exit(code=1)
    questions = select_questions(questions, question)

    if only_covered:
        from corpus.commands.coverage import (
            _ingested_source_files,
            _is_fully_covered,
        )

        ingested = _ingested_source_files()
        total = len(questions)
        questions = [q for q in questions if _is_fully_covered(q, ingested)]
        print(
            f"--only-covered: kept {len(questions)} of {total} question(s) "
            f"({total - len(questions)} skipped, cited document(s) not ingested yet).\n"
        )
        if not questions:
            print("No fully-covered questions to evaluate yet.")
            raise typer.Exit(code=1)

    personas = load_personas()
    runs = f" x {repeat} run(s)" if repeat > 1 else ""
    print(
        f"Evaluating {len(questions)} question(s){runs} with strategy={strategy} ...\n"
    )
    results = [
        _evaluate_one_with_retry(q, strategy, personas)
        for q in questions
        for _ in range(repeat)
    ]
    print_report(results)
    if repeat > 1:
        print_repeat_summary(results)
    if verbose:
        print_verbose_cases(results)
