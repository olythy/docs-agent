"""Tests for corpus.commands.meta_accuracy's pure parts."""

from corpus.commands.meta_accuracy import _KIND_TO_META, _alnum, agreement


def test_agreement_counts_matches_and_lists_the_disagreements():
    triples = [("a.docx", "x", "x"), ("b.docx", "x", "y"), ("c.docx", "z", "z")]

    ok, total, bad = agreement(triples)

    assert (ok, total) == (2, 3)
    assert bad == [("b.docx", "x", "y")]


def test_agreement_on_nothing_is_zero_of_zero():
    assert agreement([]) == (0, 0, [])


def test_court_names_compare_ignoring_accents_case_and_punctuation():
    assert _alnum("Budapest Környéki Törvényszék") == _alnum(
        "budapest kornyeki torvenyszek."
    )
    assert _alnum("Egri Törvényszék") != _alnum("Debreceni Törvényszék")


def test_every_canonical_kind_token_maps_to_a_coarse_decision_type():
    assert set(_KIND_TO_META) == {
        "judgment",
        "partial_judgment",
        "interim_judgment",
        "order",
        "other",
    }
    assert {
        _KIND_TO_META[k] for k in ("judgment", "partial_judgment", "interim_judgment")
    } == {"Ítélet"}
    assert _KIND_TO_META["order"] == "Végzés"
