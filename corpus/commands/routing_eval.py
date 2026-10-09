"""The `routing-eval` command: does the router choose the right flow for a kind of question?

The router's first decision is *which flow* a question takes: read the best passages
(``lookup``), answer exactly from the metadata (``exact``: count/list/sum/overview), or say
the request is not supported yet (``unsupported``). ``corpus/data/routing_cases.json`` lists
questions with the flow each should get; this runs the real planner on them and compares.

It runs the real decider (the planner, then the scope: a few cheap SQL lookups, but no retrieval and no answer), so it is cheap, and it measures the
*decision*, which the golden-set eval cannot: that eval grades answers, and a wrongly routed
question can still happen to look fine, or fail in a way that does not say why.

Corpus specific (the cases are about this corpus's decisions), hence under ``corpus/``.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CASES = PROJECT_ROOT / "corpus" / "data" / "routing_cases.json"
QUESTIONS = PROJECT_ROOT / "corpus" / "data" / "questions.json"


def flow_of_decision(decision) -> tuple[str, str]:
    """The flow a decision puts a question on, and what it was.

    Args:
        decision: What :class:`query.decision.PlanningDecider` decided.

    Returns:
        ``(flow, detail)``. The flow is ``lookup`` (read documents), ``exact`` (answer from
        the metadata), ``unsupported``, ``survey`` (a question about many documents that
        are not named: not built yet), ``lookup with an empty restriction`` (the question's
        filters select no document, which is never what the question wanted: it turns a
        question that can be read into "no documents match", and an answer-level eval would
        show it only as a refusal) or ``failed`` (the planner gave no usable plan).
    """
    from query.decision import AnswerExactly, ReadDocuments
    from query.outcome import DeclineReason

    if isinstance(decision, ReadDocuments):
        return "lookup", f"profile {decision.profile!r}"
    if isinstance(decision, AnswerExactly):
        return "exact", f"plan {decision.plan.operation.value!r}"
    reason = decision.declined.reason
    detail = decision.declined.detail or ""
    if reason is DeclineReason.NOT_SUPPORTED:
        return "unsupported", detail
    if reason is DeclineReason.SURVEY_NOT_YET:
        return "survey", detail
    if reason is DeclineReason.NO_MATCHING_DOCUMENTS:
        return "lookup with an empty restriction", detail
    return "failed", f"could not plan: {detail}"


def load_cases(
    cases_path: Path = CASES, questions_path: Path = QUESTIONS
) -> list[dict]:
    """Read the cases, resolving ``golden_id`` references to the question text.

    Raises:
        ValueError: If a case has neither or both of ``question`` and ``golden_id``, an
            unknown ``expected`` flow, or refers to a golden id that does not exist.
    """
    cases = json.loads(cases_path.read_text(encoding="utf-8"))["cases"]
    golden = {
        q["id"]: q["question"]
        for q in json.loads(questions_path.read_text(encoding="utf-8"))
    }
    resolved = []
    for index, case in enumerate(cases):
        if ("question" in case) == ("golden_id" in case):
            raise ValueError(
                f"case {index}: give exactly one of 'question' and 'golden_id'"
            )
        if case.get("expected") not in ("lookup", "exact", "unsupported", "survey"):
            raise ValueError(
                f"case {index}: unknown expected flow {case.get('expected')!r}"
            )
        text = case.get("question") or golden.get(case["golden_id"])
        if text is None:
            raise ValueError(f"case {index}: no golden question {case['golden_id']!r}")
        resolved.append({**case, "question": text})
    return resolved


@app.command(name="routing-eval")
def routing_eval(
    repeat: Annotated[
        int, typer.Option(min=1, help="Plan every case this many times.")
    ] = 1,
) -> None:
    """Run the planner on the routing cases and compare the flow it picks with the expected one."""
    from config import settings
    from drivers.llm import get_answer_driver
    from query.composition import build_planning
    from query.facts import QueryFactsReader

    decider, _ = build_planning(settings, get_answer_driver())
    facts_reader = QueryFactsReader()

    per_expected: dict[str, list[bool]] = defaultdict(list)
    wrong: list[str] = []
    for case in load_cases():
        for _ in range(repeat):
            got, detail = flow_of_decision(
                decider.decide(facts_reader.read(case["question"]))
            )
            ok = got == case["expected"]
            per_expected[case["expected"]].append(ok)
            if not ok:
                wrong.append(
                    f"{case['question']}\n    expected {case['expected']}, got {got} ({detail})\n"
                    f"    {case['note']}"
                )

    print(f"\n{'expected flow':<16}{'runs':>6}{'right':>8}")
    for flow in ("lookup", "exact", "unsupported", "survey"):
        results = per_expected[flow]
        if results:
            print(f"{flow:<16}{len(results):>6}{sum(results) / len(results):>8.0%}")
    total = [ok for results in per_expected.values() for ok in results]
    print(f"{'ALL':<16}{len(total):>6}{sum(total) / len(total):>8.0%}")
    for line in wrong:
        print(f"\nwrong: {line}")
