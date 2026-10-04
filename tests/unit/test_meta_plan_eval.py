"""Tests for the pure parts of corpus.commands.meta_plan_eval."""

from datetime import date

from corpus.commands.meta_plan_eval import (
    StoredDoc,
    build_facts,
    keeps_mentions,
    pick_today,
    score_count,
    score_list,
)

TODAY = date(2024, 3, 13)  # a Wednesday


def _docs():
    return [
        StoredDoc("a", "Kúria", "judgment", date(2024, 3, 5)),  # last week
        StoredDoc("b", "Kúria", "order", date(2024, 3, 20)),  # next week
        StoredDoc("c", "Debreceni Ítélőtábla", "judgment", date(2023, 10, 5)),
        StoredDoc("d", None, None, None),  # nothing known: never matches anything
    ]


def test_expected_sets_are_resolved_against_the_fixed_today():
    facts = build_facts(_docs(), TODAY, per_template=50, seed=1)

    last = {f.scope: f.expected for f in facts if f.template == "last_week"}
    nxt = {f.scope: f.expected for f in facts if f.template == "next_week"}
    assert last[""] == {"a"} and nxt[""] == {"b"}
    assert last["issued by Kúria"] == {"a"}
    assert all("d" not in f.expected for f in facts)


def test_a_scope_with_a_name_requires_the_phrasing_to_keep_it():
    facts = build_facts(_docs(), TODAY, per_template=50, seed=1)

    named = next(f for f in facts if f.scope == "issued by Kúria")
    assert named.must_mention == ("Kúria",)
    assert next(f for f in facts if f.scope == "").must_mention == ()


def test_the_description_reads_as_plain_english_for_the_phrasing_step():
    facts = build_facts(_docs(), TODAY, per_template=50, seed=1)

    fact = next(
        f for f in facts if f.template == "last_week" and f.scope == "issued by Kúria"
    )
    assert fact.description() in (
        "the number of documents issued by Kúria dated last week",
        "a list of the documents issued by Kúria dated last week",
    )


def test_the_draw_is_reproducible_and_capped_per_template():
    one = build_facts(_docs(), TODAY, per_template=1, seed=7)

    assert one == build_facts(_docs(), TODAY, per_template=1, seed=7)
    templates = [f.template for f in one]
    assert len(templates) == len(set(templates))


def test_today_is_the_median_decision_date_and_needs_dates():
    assert pick_today(_docs()) == date(2024, 3, 5)


def test_a_phrasing_that_lost_the_name_is_detected():
    assert keeps_mentions("Hány ítéletet hozott a  KÚRIA tavaly?", ("Kúria",))
    assert not keeps_mentions("Hány ítéletet hozott a bíróság?", ("Kúria",))


def test_scoring():
    assert score_count(frozenset("ab"), 2).exact
    assert not score_count(frozenset("ab"), 3).exact
    partial = score_list(frozenset("abc"), ["a", "b", "x"])
    assert (partial.exact, round(partial.precision, 2), round(partial.recall, 2)) == (
        False,
        0.67,
        0.67,
    )
    assert score_list(frozenset(), []).exact
