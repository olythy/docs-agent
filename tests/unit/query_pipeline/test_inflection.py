"""Hungarian case endings stuck to an identifier, and the facts that read them."""

import pytest

from query.facts import QueryFactsReader
from query.inflection import strip_case_ending


@pytest.mark.parametrize(
    ("token", "stem"),
    [
        ("4.P.20.409/2023/4-es", "4.P.20.409/2023/4"),
        ("BH2019.45-re", "BH2019.45"),
        ("4.P.20.409/2023/4-ban", "4.P.20.409/2023/4"),
        ("HU001-nak", "HU001"),
        (
            "4.P.20.409/2023/4-ít",
            "4.P.20.409/2023/4",
        ),  # a stored 2-letter suffix: its stem still matches
    ],
)
def test_an_ending_is_taken_off(token, stem):
    assert strip_case_ending(token) == stem


@pytest.mark.parametrize(
    "token",
    [
        "4.P.20.409/2023/4",  # nothing to strip
        "4.P.20.409/2023/4-ítélet",  # a long tail is part of the identifier
        "8.P.21.329/2024/12-III",  # upper case: part of the identifier
        "4.P.20.409/2023/4-1",  # digits
        "abc-es",  # no digit before the hyphen
    ],
)
def test_the_rest_is_left_as_it_is(token):
    assert strip_case_ending(token) == token


class TestFacts:
    def test_the_identifier_is_read_without_its_ending(self):
        facts = QueryFactsReader().read(
            "Mi volt a 4.P.20.409/2023/4-es ügyben a döntés?"
        )

        assert facts.identifiers == ("4.P.20.409/2023/4",)

    def test_the_same_number_with_and_without_an_ending_is_one_identifier(self):
        facts = QueryFactsReader().read(
            "A 4.P.20.409/2023/4-es és a 4.P.20.409/2023/4 ügy"
        )

        assert facts.identifiers == ("4.P.20.409/2023/4",)
