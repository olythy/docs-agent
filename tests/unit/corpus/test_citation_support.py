"""The chunk-level citation check (a scripted judge): which pairs it asks about and what counts as wrong."""

from typing import Any

import pytest

from corpus.citation_support import (
    check_citation_support,
    split_claims,
)
from models import Citation

C1 = Citation(1, 101, "a.docx", 0)
C2 = Citation(2, 102, "b.docx", 5)
TEXT = {101: "chunk one says X", 102: "chunk two says Y"}


class Judge:
    """Returns scripted verdicts and remembers what it was asked."""

    def __init__(self, verdicts: Any):
        self.verdicts, self.asked = verdicts, []

    def __call__(self, pairs) -> Any:
        self.asked.append(list(pairs))
        return self.verdicts(pairs) if callable(self.verdicts) else self.verdicts


def check(text, citations, judge, unknown=(), refused=False):
    return check_citation_support(text, citations, TEXT, unknown, judge, refused)


def test_every_sentence_label_pair_is_judged_against_that_chunk_alone():
    judge = Judge([True, False])

    result = check("The court granted X [1]. It also ordered Y [2].", [C1, C2], judge)

    assert judge.asked == [
        [
            ("The court granted X.", "chunk one says X"),
            ("It also ordered Y.", "chunk two says Y"),
        ]
    ]
    assert (result.supported, result.unsupported, result.wrong) == (1, 1, 1)


def test_citing_all_labels_after_one_claim_makes_a_pair_for_each():
    judge = Judge([True, False])

    result = check("The amount is 733910 forint [1][2].", [C1, C2], judge)

    assert [p.label for p in result.pairs] == [1, 2]
    assert result.wrong == 1  # over-citation: the second chunk does not state it


def test_an_unknown_label_is_a_wrong_citation_even_with_nothing_to_judge():
    judge = Judge([])

    result = check("It says so [9].", [], judge, unknown=(9,))

    assert result.pairs == () and judge.asked == []
    assert result.wrong == 1 and result.unknown_labels == (9,)


def test_the_sources_block_and_what_follows_are_not_read_as_claims():
    text = "The court dismissed the claim [1].\n\nSources:\n[1] a.docx, section 0\n\nNOTE that a scope note follows here."
    judge = Judge([True])

    check(text, [C1], judge)

    assert len(judge.asked[0]) == 1


def test_an_answer_that_states_something_and_cites_nothing_is_uncited():
    result = check("The court dismissed the claim.", [], Judge([]))

    assert result.uncited and result.wrong == 0


def test_a_refusal_is_not_uncited():
    result = check(
        "I could not find this information in the provided documents.",
        [],
        Judge([]),
        refused=True,
    )

    assert not result.uncited


def test_a_judge_that_gives_no_usable_answer_leaves_the_pairs_unjudged_not_wrong():
    for verdicts in (None, [True]):  # no answer; or the wrong number of answers
        result = check(
            "The first claim is here [1]. And more words here [2].",
            [C1, C2],
            Judge(verdicts),
        )

        assert result.unjudged == 2 and result.wrong == 0 and result.supported == 0


def test_a_label_whose_chunk_text_is_unknown_is_skipped_not_judged():
    judge = Judge([True])

    result = check_citation_support("Claim one is here [1].", [C1], {}, (), judge)

    assert result.pairs == () and judge.asked == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "A fact is here [1]. Another one is there [2][3].",
            [("A fact is here.", [1]), ("Another one is there.", [2, 3])],
        ),
        (
            "First line is long enough [1]\nSecond line is long enough [2]",
            [("First line is long enough", [1]), ("Second line is long enough", [2])],
        ),
        (
            "Short. This sentence is long enough without a label.",
            [("This sentence is long enough without a label.", [])],
        ),
        (
            "The year [2023] is not a label, and the amount 1.234 Ft is not a split.",
            [
                (
                    "The year [2023] is not a label, and the amount 1.234 Ft is not a split.",
                    [],
                )
            ],
        ),
        ("", []),
    ],
)
def test_the_answer_is_split_into_claims_with_their_labels(text, expected):
    assert split_claims(text) == expected


def test_the_dict_for_the_evaluation_record_lists_the_unsupported_claims():
    result = check(
        "The first claim is true [1]. The second claim is true [2].",
        [C1, C2],
        Judge([True, False]),
        unknown=(7,),
    )

    record = result.to_dict()

    assert (
        record["pairs"] == 2 and record["supported"] == 1 and record["unsupported"] == 1
    )
    assert record["wrong"] == 2 and record["unknown_labels"] == [7]
    assert record["unsupported_claims"] == [
        {"label": 2, "sentence": "The second claim is true."}
    ]
