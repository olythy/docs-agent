"""The `routing-eval` command: does the router choose the right flow for a kind of question?

The router's first decision is *which flow* a question takes: read the best passages
(``lookup``), answer exactly from the metadata (``exact``: count/list/sum/overview), or say
the request is not supported yet (``unsupported``). ``corpus/data/routing_cases.json`` lists
questions with the flow each should get; this runs the real planner on them and compares.

It runs only the planner (no SQL, no retrieval, no answer), so it is cheap, and it measures the
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


def flow_of(operation: str) -> str:
    """The flow an operation puts a question on: ``lookup``, ``unsupported`` or ``exact``."""
    if operation in ("lookup", "unsupported"):
        return operation
    return "exact"  # count, list, sum, overview: answered from the metadata


def restriction_problem(plan, executor) -> str | None:
    """Why the restriction a lookup plan puts on the retrieval is wrong, if it is.

    A lookup is restricted to the documents its type and filters select. A restriction
    that selects *no document* is never what the question wanted (the corpus holds
    documents the question is about): it turns a question that can be read into "no
    documents match". It is what two contradicting filters on one key do, which an
    answer-level eval would show only as a refusal, never as a routing failure.

    Args:
        plan: The plan as the router treats it (a lookup).
        executor: Runs a plan (:class:`metadata.executor.PlanExecutor`).

    Returns:
        A description of the problem, or ``None`` when there is no restriction or it
        selects at least one document.
    """
    if plan.doc_type is None and not plan.filters:
        return None
    result = executor.execute(plan)
    if result.count == 0:
        return f"the restriction selects no document ({result.explanation})"
    return None


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
        if case.get("expected") not in ("lookup", "exact", "unsupported"):
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
    from document_store import DocumentStore
    from drivers.llm import get_answer_driver
    from metadata.clock import SystemClock
    from metadata.compiler import PlanCompiler
    from metadata.date_ranges import DateRangeResolver
    from metadata.executor import PlanExecutor
    from metadata.planner import (
        LLMQueryPlanner,
        PlanningFailed,
        collect_known_values,
        load_catalogs,
    )
    from query.router import as_routed
    from store import extract_identifier_tokens

    clock = SystemClock()
    store = DocumentStore()
    catalogs = load_catalogs(store)
    known = collect_known_values(store, catalogs)
    compiler = PlanCompiler(DateRangeResolver(clock))
    planner = LLMQueryPlanner(get_answer_driver(), compiler, clock)
    executor = PlanExecutor(store, compiler)

    per_expected: dict[str, list[bool]] = defaultdict(list)
    wrong: list[str] = []
    for case in load_cases():
        for _ in range(repeat):
            try:
                plan = planner.plan(case["question"], catalogs, known)
                routed = as_routed(plan)  # what the router really does with it
                got = flow_of(routed.operation.value)
                detail = (
                    f"planned {plan.operation.value!r}"
                    + (f" with residual {plan.residual!r}" if plan.residual else "")
                    + f", type {plan.doc_type!r}"
                )
                # A lookup that names an identifier is not restricted (the router
                # lets the retrieval find the document); the others are.
                problem = (
                    restriction_problem(routed, executor)
                    if got == "lookup"
                    and not extract_identifier_tokens(case["question"])
                    else None
                )
                if problem:
                    got, detail = "lookup with an empty restriction", problem
            except PlanningFailed as failure:
                got, detail = "failed", f"could not plan: {failure.reason}"
            ok = got == case["expected"]
            per_expected[case["expected"]].append(ok)
            if not ok:
                wrong.append(
                    f"{case['question']}\n    expected {case['expected']}, got {got} ({detail})\n"
                    f"    {case['note']}"
                )

    print(f"\n{'expected flow':<16}{'runs':>6}{'right':>8}")
    for flow in ("lookup", "exact", "unsupported"):
        results = per_expected[flow]
        if results:
            print(f"{flow:<16}{len(results):>6}{sum(results) / len(results):>8.0%}")
    total = [ok for results in per_expected.values() for ok in results]
    print(f"{'ALL':<16}{len(total):>6}{sum(total) / len(total):>8.0%}")
    for line in wrong:
        print(f"\nwrong: {line}")
