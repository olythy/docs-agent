"""Tests for query.time_filter.extract_years."""

import pytest

from query.time_filter import extract_years


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("a 2020 és 2022 között hozott ítéletek", [2020, 2021, 2022]),
        ("a 2020-2022 közötti időszakban", [2020, 2021, 2022]),
        ("between 2019 and 2021", [2019, 2020, 2021]),
        ("2022-ben hozott határozatok", [2022]),
        # Separate mentions stay separate: 2022 is NOT implied.
        ("a 2021-es és a 2023-as években", [2021, 2023]),
        ("nincs benne év", []),
        ("2099-ben", []),
    ],
)
def test_extract_years(question, expected):
    assert extract_years(question) == expected


def test_years_inside_case_numbers_are_ignored():
    """A year that is part of a case number is not a time period asked about."""
    assert extract_years("a P.20.457/2018/11. és 18.K.700.752/2022/2. ügyek") == []


def test_case_number_year_ignored_but_real_period_kept():
    assert extract_years("a P.20.457/2018/11. ügy és a 2021-es határozatok") == [2021]


def test_oversized_range_is_not_expanded():
    assert extract_years("1990-2030 között") == []
