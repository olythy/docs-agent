"""Tests for the pure parts of corpus.commands.meta_plan_eval."""

from datetime import date

from corpus.commands.meta_plan_eval import (
    StoredDoc,
    build_cases,
    score_count,
    score_list,
)

TODAY = date(2026, 10, 4)


def _docs():
    return [
        StoredDoc("a", "Kúria", "judgment", date(2025, 10, 3)),
        StoredDoc("b", "Kúria", "judgment", date(2025, 10, 20)),
        StoredDoc("c", "Kúria", "order", date(2024, 1, 5)),
        StoredDoc("d", None, None, None),  # nothing known: never matches anything
    ]


def test_expected_sets_come_from_stored_values_only():
    cases = {c.template: c for c in build_cases(_docs(), TODAY, 10, seed=1)}

    assert all("d" not in c.expected for c in build_cases(_docs(), TODAY, 10, seed=1))
    assert cases["count_relative_month"].expected == {"a", "b"}
    assert "tavaly október hónapjában" in cases["count_relative_month"].question


def test_the_draw_is_reproducible_and_capped_per_template():
    first = build_cases(_docs(), TODAY, 1, seed=7)
    again = build_cases(_docs(), TODAY, 1, seed=7)

    assert first == again
    templates = [c.template for c in first]
    assert len(templates) == len(set(templates))


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
