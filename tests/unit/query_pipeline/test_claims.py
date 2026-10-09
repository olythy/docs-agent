"""Cutting an answer into sentences and reading their labels (shared by the answerer and the eval)."""

import pytest

from query.claims import (
    cited_labels,
    plain_sentence,
    sentence_spans,
    split_claims,
    strip_labels,
)


def test_sentences_are_cut_at_a_full_stop_before_a_capital_or_a_label_and_at_line_breaks():
    text = "First claim [1]. Second claim [2]\nThird claim. [3] Fourth one."

    spans = sentence_spans(text)

    assert [text[a:b] for a, b in spans] == [
        "First claim [1].",
        "Second claim [2]",
        "Third claim.",
        "[3] Fourth one.",
    ]


def test_a_number_with_dots_is_not_a_sentence_end():
    text = "The amount is 1.234 Ft and the case 4.P.20.409/2023/4 stays whole."

    assert len(sentence_spans(text)) == 1


def test_an_empty_or_blank_text_has_no_claims():
    assert sentence_spans("") == []
    assert split_claims("") == [] and split_claims("  \n  ") == []


@pytest.mark.parametrize(
    ("sentence", "plain"),
    [
        ("It was so [1][2].", "It was so."),
        ("It was so [ 1 , 3 ] .", "It was so."),
        ("It was   so  [2]", "It was so"),
        ("In [2023] it was so.", "In [2023] it was so."),
    ],
)
def test_a_sentence_without_its_labels_is_tidy(sentence, plain):
    assert plain_sentence(sentence) == plain


def test_short_sentences_are_not_claims():
    assert split_claims("Yes [1]. The court dismissed the claim [2].") == [
        ("The court dismissed the claim.", [2])
    ]


def test_labels_are_read_once_in_order_and_years_are_not_labels():
    assert cited_labels("a [2] b [1, 2] c [2023] d [3;1]") == [2, 1, 3]


def test_labels_can_be_taken_out_and_a_bracket_can_keep_the_others():
    assert strip_labels("a [1][2] b [1, 3] c", {1}) == "a [2] b [3] c"
    assert strip_labels("a [1] b", {1}) == "a b"
    assert strip_labels("a [1] b", set()) == "a [1] b"
    assert (
        strip_labels("a [1][2][3] b", {2}) == "a [1][3] b"
    )  # the run keeps its one space
    assert strip_labels("a [1][2] b", {1, 2}) == "a b"
