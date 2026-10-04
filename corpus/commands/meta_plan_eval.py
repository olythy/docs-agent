"""The `meta-plan-eval` command: does question -> plan -> SQL give the exact answer?

Generates count and list questions in Hungarian from the values already stored in
``document_meta`` (the court, the kind of decision, the decision date), computes
each question's exact expected answer *in Python*, independently of the plan
compiler, then lets the real planner and executor answer it and compares.
Scoring is set arithmetic, no LLM grader: a count must equal the expected number,
and a list is scored with precision/recall of document sets.

This measures the *planner and compiler*. Whether the stored values are right is
measured separately by ``meta-accuracy``. The question templates (and the
Hungarian month and decision-kind words) are specific to this corpus, which is why
this lives under ``corpus/`` and never in the generic metadata core.
"""

import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import typer

app = typer.Typer()

DOC_TYPE = "court_decision"

_MONTHS = [
    "január", "február", "március", "április", "május", "június",
    "július", "augusztus", "szeptember", "október", "november", "december",
]  # fmt: skip
_KIND_WORDS = {"judgment": "ítéletet", "order": "végzést"}
#: A list question is only generated when the answer fits the plan's default limit.
_MAX_LIST = 50


@dataclass(frozen=True)
class StoredDoc:
    """The stored facts the questions are generated from."""

    content_hash: str
    body: str | None
    kind: str | None
    decided: date | None


@dataclass(frozen=True)
class EvalCase:
    """One question with its exactly computed answer.

    Attributes:
        template: Which question shape produced it.
        question: The question, in Hungarian.
        operation: ``count`` or ``list``.
        expected: The content hashes of the documents that match.
    """

    template: str
    question: str
    operation: str
    expected: frozenset[str]


def build_cases(
    docs: Sequence[StoredDoc], today: date, per_template: int, seed: int
) -> list[EvalCase]:
    """Generate up to ``per_template`` questions of each shape, reproducibly.

    Only combinations that match at least one document are used, and a document
    whose relevant value is unknown never counts as a match (the comparison is
    against what is *stored*, so the unknown documents are reported separately by
    the executor).

    Args:
        docs: Every document with its stored facts.
        today: The date the relative questions are asked on.
        per_template: How many questions to draw per template.
        seed: Seeds the draw.
    """
    rng = random.Random(seed)
    cases: list[EvalCase] = []

    def draw(template: str, groups: dict, make, operation: str, cap: int | None = None):
        keys = sorted(groups)
        rng.shuffle(keys)
        taken = 0
        for key in keys:
            hashes = frozenset(groups[key])
            if cap is not None and len(hashes) > cap:
                continue
            cases.append(EvalCase(template, make(*key), operation, hashes))
            taken += 1
            if taken == per_template:
                break

    by_body_kind: dict[tuple, set] = defaultdict(set)
    by_year: dict[tuple, set] = defaultdict(set)
    by_month: dict[tuple, set] = defaultdict(set)
    by_body_year: dict[tuple, set] = defaultdict(set)
    for d in docs:
        if d.body and d.kind in _KIND_WORDS:
            by_body_kind[(d.body, d.kind)].add(d.content_hash)
        if d.decided:
            by_year[(d.decided.year,)].add(d.content_hash)
            by_month[(d.decided.year, d.decided.month)].add(d.content_hash)
        if d.body and d.decided:
            by_body_year[(d.body, d.decided.year)].add(d.content_hash)

    draw(
        "count_body_kind",
        by_body_kind,
        lambda body, kind: f"Hány {_KIND_WORDS[kind]} hozott a(z) {body}?",
        "count",
    )
    draw(
        "count_year",
        by_year,
        lambda year: f"Hány határozat kelt {year}-ben?",
        "count",
    )
    draw(
        "count_month",
        by_month,
        lambda year, month: (
            f"Hány határozat kelt {year} {_MONTHS[month - 1]} hónapjában?"
        ),
        "count",
    )
    last_year = {k: v for k, v in by_month.items() if k[0] == today.year - 1}
    draw(
        "count_relative_month",
        {(month,): hashes for (_, month), hashes in last_year.items()},
        lambda month: f"Hány határozat kelt tavaly {_MONTHS[month - 1]} hónapjában?",
        "count",
    )
    draw(
        "list_body_year",
        by_body_year,
        lambda body, year: f"Sorold fel a(z) {body} {year}-ben kelt határozatait.",
        "list",
        cap=_MAX_LIST,
    )
    return cases


@dataclass(frozen=True)
class Score:
    """How one answer compares with the expected set.

    Attributes:
        exact: A count equals the expected number (or a list is exactly the set).
        precision: Share of returned documents that are expected (lists).
        recall: Share of expected documents that were returned (lists).
    """

    exact: bool
    precision: float
    recall: float


def score_list(expected: frozenset[str], returned: Sequence[str]) -> Score:
    """Score a returned document list against the expected set."""
    got = set(returned)
    hit = len(got & expected)
    return Score(
        exact=got == expected,
        precision=hit / len(got) if got else (1.0 if not expected else 0.0),
        recall=hit / len(expected) if expected else 1.0,
    )


def score_count(expected: frozenset[str], counted: int) -> Score:
    """Score a count: it is right only if it equals the expected number."""
    right = counted == len(expected)
    return Score(right, float(right), float(right))


def _load_docs() -> list[StoredDoc]:
    from document_store import DocumentStore

    store = DocumentStore()
    body = {v.content_hash: v.value_text for _, v in store.list_values("issuing_body")}
    kind = {v.content_hash: v.value_text for _, v in store.list_values("document_kind")}
    decided = {
        v.content_hash: v.value_date for _, v in store.list_values("decision_date")
    }
    every = set(body) | set(kind) | set(decided)
    return [
        StoredDoc(h, body.get(h), kind.get(h), decided.get(h)) for h in sorted(every)
    ]


@app.command(name="meta-plan-eval")
def meta_plan_eval(
    per_template: int = typer.Option(4, help="Questions drawn per template."),
    seed: int = typer.Option(1, help="Seeds the question draw."),
    show: int = typer.Option(5, help="How many wrong answers to print in full."),
) -> None:
    """Ask generated count/list questions through the real planner and score them exactly."""
    from document_store import DocumentStore
    from drivers.llm import get_answer_driver
    from metadata.clock import SystemClock
    from metadata.compiler import PlanCompiler
    from metadata.date_ranges import DateRangeResolver
    from metadata.executor import PlanExecutor
    from metadata.plan import PlanError
    from metadata.planner import LLMQueryPlanner, PlanningFailed
    from models import KeyStatus

    clock = SystemClock()
    store = DocumentStore()
    compiler = PlanCompiler(DateRangeResolver(clock))
    planner = LLMQueryPlanner(get_answer_driver(), compiler, clock)
    executor = PlanExecutor(store, compiler)
    keys = store.list_keys(DOC_TYPE, KeyStatus.APPROVED)

    cases = build_cases(_load_docs(), clock.today(), per_template, seed)
    outcomes: dict[str, list[Score]] = defaultdict(list)
    wrong: list[str] = []
    for case in cases:
        try:
            plan = planner.plan(case.question, DOC_TYPE, keys)
            result = executor.execute(plan)
        except (PlanningFailed, PlanError) as exc:
            outcomes[case.template].append(Score(False, 0.0, 0.0))
            wrong.append(f"{case.question}\n    FAILED: {exc}")
            continue
        if case.operation == "count":
            got = score_count(case.expected, result.count or 0)
        else:
            got = score_list(case.expected, [str(row[0]) for row in result.documents])
        outcomes[case.template].append(got)
        if not got.exact:
            wrong.append(
                f"{case.question}\n    expected {len(case.expected)}, got "
                f"{result.count} (+{result.unknown} unknown); plan: {result.explanation}"
            )

    print(f"\n{'template':<24}{'n':>4}{'exact':>8}{'precision':>11}{'recall':>9}")
    for template, scores in outcomes.items():
        n = len(scores)
        print(
            f"{template:<24}{n:>4}{sum(s.exact for s in scores) / n:>8.0%}"
            f"{sum(s.precision for s in scores) / n:>11.0%}"
            f"{sum(s.recall for s in scores) / n:>9.0%}"
        )
    for line in wrong[:show]:
        print(f"\nwrong: {line}")
    print(
        """
How to read this:
  exact      a count equals the expected number / a list is exactly the expected set.
  Expected answers are computed in Python from the stored values, not by the compiler.
  Documents whose key is not extracted yet are 'unknown' and cannot match, so run
  this on full coverage; the printed (+K unknown) shows the gap per wrong answer."""
    )
