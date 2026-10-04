"""The `meta-plan-eval` command: does question -> plan -> SQL give the exact answer?

Builds count and list questions whose exact answers are known, lets the real
planner and executor answer them, and scores the result by set arithmetic (no LLM
grader). It measures the *planner and compiler*; whether the stored values are
right is measured separately by ``meta-accuracy``.

How a question is made, and why:

1. The code picks a *fact* from the stored values: a scope (all documents, one
   issuing body, one kind) and a period ("last week", "last year's October", "in
   the last 30 days", ...), and computes the exact expected document set in Python.
2. An LLM only *phrases* that fact as a natural question in the chosen language,
   so the questions are not tied to hand-written, single-language templates. It
   is never asked to compute anything, so the expected answer stays independent
   of any model. A phrasing that loses the scope's text value (e.g. the court's
   name) is rejected and reported, not silently kept.
3. Relative periods ("next week") need data near "today", but the corpus is all
   past, so the eval runs on a *fixed clock* set inside the data's date range.
   The planner is told that date, and the expected ranges are resolved with it.

The period arithmetic uses the same ``DateRangeResolver`` as the compiler; that
resolver has its own calendar tests, so what is measured here is whether the
planner *chose the right period description*. The corpus-specific part is only
which stored keys it reads (``issuing_body``, ``document_kind``,
``decision_date``), which is why this lives under ``corpus/``.
"""

import calendar
import random
import statistics
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date

import typer

from metadata.clock import FixedClock
from metadata.date_ranges import DateRange, DateRangeResolver

app = typer.Typer()

DOC_TYPE = "court_decision"

#: A list question is only generated when the answer fits the plan's default limit.
_MAX_LIST = 50
#: Share of the generated questions that ask for a list rather than a count.
_LIST_SHARE = 0.3


@dataclass(frozen=True)
class StoredDoc:
    """The stored facts the questions are generated from."""

    content_hash: str
    body: str | None
    kind: str | None
    decided: date | None


@dataclass(frozen=True)
class Fact:
    """A question's meaning, before it is phrased, with its exact answer.

    Attributes:
        template: The period shape, e.g. ``last_week`` (what the report groups by).
        operation: ``count`` or ``list``.
        scope: The documents asked about, in English words (empty for all).
        period: The period as a person would say it, in English.
        expected: The content hashes of the documents that match.
        must_mention: Text values a phrasing has to keep (e.g. the court's name).
    """

    template: str
    operation: str
    scope: str
    period: str
    expected: frozenset[str]
    must_mention: tuple[str, ...] = ()

    def description(self) -> str:
        """The fact in plain English, for the phrasing step."""
        what = (
            "the number of documents"
            if self.operation == "count"
            else "a list of the documents"
        )
        scope = f" {self.scope}" if self.scope else ""
        return f"{what}{scope} dated {self.period}"


def _periods(rng: random.Random, years: Sequence[int]):
    """Yield ``(template, spoken period, date spec)`` for a spread of period shapes."""
    year = rng.choice(years)
    month = rng.randint(1, 12)
    name = calendar.month_name[month]
    quarter = rng.randint(1, 4)
    yield "calendar_year", f"in {year}", {"kind": "calendar", "year": year}
    yield (
        "calendar_month",
        f"in {name} {year}",
        {"kind": "calendar", "year": year, "month": month},
    )
    yield (
        "calendar_quarter",
        f"in the {quarter}. quarter of {year}",
        {"kind": "calendar", "year": year, "quarter": quarter},
    )
    yield (
        "last_years_month",
        f"in {name} of last year",
        {"kind": "calendar", "year_offset": -1, "month": month},
    )
    for template, phrase, unit, offset in (
        ("last_year", "last year", "year", -1),
        ("last_quarter", "last quarter", "quarter", -1),
        ("last_month", "last month", "month", -1),
        ("this_month", "this month", "month", 0),
        ("next_month", "next month", "month", 1),
        ("last_week", "last week", "week", -1),
        ("this_week", "this week", "week", 0),
        ("next_week", "next week", "week", 1),
    ):
        yield template, phrase, {"kind": "relative", "unit": unit, "offset": offset}
    yield (
        "rolling_past",
        "in the last 30 days",
        {"kind": "rolling", "unit": "day", "count": 30},
    )
    yield (
        "rolling_future",
        "in the next 14 days",
        {"kind": "rolling", "unit": "day", "count": 14, "direction": "future"},
    )


def _kind_words(kind: str) -> str:
    return kind.replace("_", " ") + "s"


def build_facts(
    docs: Sequence[StoredDoc], today: date, per_template: int, seed: int
) -> list[Fact]:
    """Generate up to ``per_template`` facts for each period shape, reproducibly.

    Only combinations that match at least one document are used. A document whose
    relevant value is unknown never matches (the comparison is against what is
    *stored*; the executor reports the unknown ones separately).

    Args:
        docs: Every document with its stored facts.
        today: The (fixed) date the relative periods are resolved against.
        per_template: How many facts to draw per period shape.
        seed: Seeds the draw.
    """
    rng = random.Random(seed)
    resolver = DateRangeResolver(FixedClock(today))
    dated = [d for d in docs if d.decided]
    years = sorted({d.decided.year for d in dated if d.decided})
    if not years:
        return []
    scopes: list[tuple[str, tuple[str, ...], Callable[[StoredDoc], bool]]] = [
        ("", (), lambda d: True)
    ]
    for body in sorted({d.body for d in dated if d.body}):
        scopes.append((f"issued by {body}", (body,), lambda d, b=body: d.body == b))
    for kind in sorted({d.kind for d in dated if d.kind}):
        scopes.append(
            (f"that are {_kind_words(kind)}", (), lambda d, k=kind: d.kind == k)
        )

    drawn: dict[str, list[Fact]] = defaultdict(list)
    for template, period, spec in _periods(rng, years):
        window: DateRange = resolver.resolve(spec)
        in_window = [
            d for d in dated if d.decided and window.start <= d.decided <= window.end
        ]
        order = list(range(len(scopes)))
        rng.shuffle(order)
        for index in order:
            if len(drawn[template]) >= per_template:
                break
            scope, mention, matches = scopes[index]
            hashes = frozenset(d.content_hash for d in in_window if matches(d))
            if not hashes:
                continue
            as_list = len(hashes) <= _MAX_LIST and rng.random() < _LIST_SHARE
            drawn[template].append(
                Fact(
                    template,
                    "list" if as_list else "count",
                    scope,
                    period,
                    hashes,
                    mention,
                )
            )
    return [fact for facts in drawn.values() for fact in facts]


def pick_today(docs: Sequence[StoredDoc]) -> date:
    """A fixed 'today' inside the data's date range: the median decision date.

    Relative periods around it ("last week", "next month") then contain documents.
    """
    dates = sorted(d.decided for d in docs if d.decided)
    if not dates:
        raise ValueError("no stored decision dates to pick a 'today' from")
    return dates[len(dates) // 2]


def keeps_mentions(question: str, must_mention: Sequence[str]) -> bool:
    """Whether a phrased question still contains every required text value."""
    folded = " ".join(question.lower().split())
    return all(" ".join(m.lower().split()) in folded for m in must_mention)


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


_PHRASING_PROMPT = """Write ONE natural question, in {language}, that a person would ask to get exactly this:

    {description}

Rules:
- Keep the period the way a person says it (e.g. "last week", "in March 2023"); do not turn it into dates and do not compute anything.
- Keep any proper name exactly as given.
- Vary the wording naturally; use "how many" for a number and "list/which" for a list.
- Output ONLY the question.
"""


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
    per_template: int = typer.Option(2, help="Questions drawn per period shape."),
    seed: int = typer.Option(1, help="Seeds the draw."),
    language: str = typer.Option(
        "Hungarian", help="Language the questions are phrased in."
    ),
    today: str = typer.Option(
        "", help="Fixed 'today' (YYYY-MM-DD); default: the median decision date."
    ),
    show: int = typer.Option(5, help="How many wrong answers to print in full."),
) -> None:
    """Phrase generated count/list facts as questions, answer them through the planner, score exactly."""
    from document_store import DocumentStore
    from drivers.llm import get_answer_driver
    from metadata.compiler import PlanCompiler
    from metadata.executor import PlanExecutor
    from metadata.plan import PlanError
    from metadata.planner import LLMQueryPlanner, PlanningFailed, collect_known_values
    from models import KeyStatus

    docs = _load_docs()
    fixed_today = date.fromisoformat(today) if today else pick_today(docs)
    clock = FixedClock(fixed_today)
    store = DocumentStore()
    compiler = PlanCompiler(DateRangeResolver(clock))
    llm = get_answer_driver()
    planner = LLMQueryPlanner(llm, compiler, clock)
    executor = PlanExecutor(store, compiler)
    keys = store.list_keys(DOC_TYPE, KeyStatus.APPROVED)
    known_values = collect_known_values(store, keys)

    print(f"fixed today for this run: {fixed_today}")
    outcomes: dict[str, list[Score]] = defaultdict(list)
    wrong: list[str] = []
    rejected = 0
    for fact in build_facts(docs, fixed_today, per_template, seed):
        question = (
            llm.run_tool_calling_turn(
                [
                    {
                        "role": "user",
                        "content": _PHRASING_PROMPT.format(
                            language=language, description=fact.description()
                        ),
                    }
                ]
            ).content
            or ""
        ).strip()
        if not question or not keeps_mentions(question, fact.must_mention):
            rejected += 1
            print(
                f"  phrasing rejected (lost a name): {question!r} <- {fact.description()}"
            )
            continue
        try:
            result = executor.execute(
                planner.plan(question, DOC_TYPE, keys, known_values)
            )
        except (PlanningFailed, PlanError) as exc:
            outcomes[fact.template].append(Score(False, 0.0, 0.0))
            wrong.append(f"{question}\n    FAILED: {exc}")
            continue
        if fact.operation == "count":
            got = score_count(fact.expected, result.count or 0)
        else:
            got = score_list(fact.expected, [str(row[0]) for row in result.documents])
        outcomes[fact.template].append(got)
        if not got.exact:
            wrong.append(
                f"{question}\n    meant: {fact.description()}\n    expected {len(fact.expected)}, got "
                f"{result.count} (+{result.unknown} unknown); executed: {result.explanation}"
            )

    print(f"\n{'period shape':<20}{'n':>4}{'exact':>8}{'precision':>11}{'recall':>9}")
    for template, scores in outcomes.items():
        n = len(scores)
        print(
            f"{template:<20}{n:>4}{sum(s.exact for s in scores) / n:>8.0%}"
            f"{sum(s.precision for s in scores) / n:>11.0%}"
            f"{sum(s.recall for s in scores) / n:>9.0%}"
        )
    everything = [s for scores in outcomes.values() for s in scores]
    if everything:
        print(
            f"{'ALL':<20}{len(everything):>4}"
            f"{statistics.mean(s.exact for s in everything):>8.0%}"
        )
    if rejected:
        print(f"\n{rejected} phrasing(s) rejected and not asked (listed above).")
    for line in wrong[:show]:
        print(f"\nwrong: {line}")
    print(
        """
How to read this:
  exact      a count equals the expected number / a list is exactly the expected set.
  Expected answers are computed in Python from the stored values; an LLM only phrased
  the questions. Documents whose key is not extracted yet are 'unknown' and cannot
  match, so run this on full coverage; the printed (+K unknown) shows the gap.
  The same model phrases and plans: a shared blind spot is possible, so read the
  'wrong' cases yourself rather than trusting the percentage alone."""
    )
