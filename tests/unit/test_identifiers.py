"""The rule for comparing identifiers as they are written (pure)."""

import pytest

from metadata.identifiers import (
    compact_identifier,
    compact_pattern,
    identifier_compact_contains,
    identifier_contains,
    identifier_matches,
    is_specific_enough,
    normalize_identifier,
    partial_pattern,
)


class TestNormalize:
    @pytest.mark.parametrize(
        ("written", "normal"),
        [
            ("4.P.20.409/2023/4", "4.p.20.409/2023/4"),
            ("  4.P.20.409/2023/4. ", "4.p.20.409/2023/4"),  # spaces, a full stop
            ("10. P. 20.277/2019/77.", "10.p.20.277/2019/77"),  # spaces inside
            ("4.P.20.409/2023/4-", "4.p.20.409/2023/4"),  # a trailing hyphen
            ("/4.P.20.409/2023/4/", "4.p.20.409/2023/4"),  # slashes at the ends
            ("x y/1", "xy/1"),  # a no-break space
            ("ＡＢ１２３", "ab123"),  # full-width letters and digits
            ("4.P.20.409/2023/4-ítélet", "4.p.20.409/2023/4-ítélet"),  # kept inside
        ],
    )
    def test_the_normal_form(self, written, normal):
        assert normalize_identifier(written) == normal

    def test_nothing_but_punctuation_normalises_to_nothing(self):
        assert normalize_identifier(" ./-,;: ") == ""


class TestMatches:
    @pytest.mark.parametrize(
        ("stored", "wanted"),
        [
            ("4.P.20.409/2023/4", "4.P.20.409/2023/4"),  # the same
            ("4.P.20.409/2023/4.", "4.P.20.409/2023/4"),  # a full stop
            ("4.p.20.409/2023/4", "4.P.20.409/2023/4"),  # case
            ("10. P. 20.277/2019/77.", "10.P.20.277/2019/77"),  # spaces
            (
                "4.P.20.409/2023/4-ítélet",
                "4.P.20.409/2023/4",
            ),  # a suffix after a hyphen
            ("4.P.20.409/2023/4/ítélet", "4.P.20.409/2023/4"),  # after a slash
            ("8.P.21.329/2024/12-III", "8.P.21.329/2024/12"),  # a roman suffix
            ("11.P.20.377/2023/11JAV", "11.P.20.377/2023/11"),  # directly after a digit
            (
                "104.K.702.368/2022/22",
                "104.K.702.368/2022",
            ),  # the case, for its decision
        ],
    )
    def test_a_written_variant_is_the_same_identifier(self, stored, wanted):
        assert identifier_matches(stored, wanted)

    @pytest.mark.parametrize(
        ("stored", "wanted"),
        [
            ("4.P.20.409/2023/40", "4.P.20.409/2023/4"),  # more digits: another number
            ("4.P.20.409/2023/4", "4.P.20.409/2023/40"),  # the stored one is shorter
            ("4.P.20.409/2023/5", "4.P.20.409/2023/4"),
            ("14.P.20.409/2023/4", "4.P.20.409/2023/4"),  # not a prefix
            ("5.4.P.20.409/2023/4", "4.P.20.409/2023/4"),
        ],
    )
    def test_a_different_number_is_not_matched(self, stored, wanted):
        assert not identifier_matches(stored, wanted)

    @pytest.mark.parametrize(
        ("stored", "wanted"),
        [
            (
                "4.P.20.409/2023/4",
                "4.P",
            ),  # a short wish as a prefix would match everything
            ("HU001-A", "HU001"),
            ("2023-01-15-7", "2023-01"),
        ],
    )
    def test_a_short_wish_is_only_matched_when_equal(self, stored, wanted):
        assert not identifier_matches(stored, wanted)

    def test_and_it_still_matches_when_equal(self):
        assert identifier_matches("HU001", "hu001.")
        assert identifier_matches("4.P", "4.p")

    def test_an_empty_wish_matches_nothing(self):
        assert not identifier_matches("4.P.20.409/2023/4", "")
        assert not identifier_matches("4.P.20.409/2023/4", " ./ ")


class TestSpecificEnough:
    """Only an identifier that says enough may be looked for as a part of another."""

    @pytest.mark.parametrize(
        "wanted",
        [
            "20.277/2019/77",
            "P.20.277/2019/77",
            "P20487",
            "HU-BUD-2024",
            "10. P. 20.277/2019/77.",
        ],
    )
    def test_enough(self, wanted):
        assert is_specific_enough(wanted)

    @pytest.mark.parametrize(
        "wanted",
        [
            "4.P",  # too short
            "2023-01",  # a date: digits and one separator only
            "2023-01-15",  # a whole date: long enough, but a date
            "15.01.2023",
            "2023.01.15.",
            "123456",  # a bare number
            "ABCDEF",  # no digit
            "1/2/3",  # structure but too few characters
            "",
            " ./ ",
        ],
    )
    def test_not_enough(self, wanted):
        assert not is_specific_enough(wanted)


class TestContains:
    @pytest.mark.parametrize(
        ("stored", "wanted"),
        [
            ("10.P.20.277/2019/77", "P.20.277/2019/77"),  # the leading number left out
            ("10.P.20.277/2019/77", "20.277/2019/77"),  # and the letters too
            ("4.P.20.487/2020/221-ítélet", "20.487/2020/221"),  # a suffix after it
            ("10.P.20.277/2019/77", "10. P. 20.277/2019/77."),  # as written, spaced
            (
                "PREFIX-2024/00123",
                "2024/00123",
            ),  # not a court: any composite identifier
            ("10.p.20.277/2019/77", "P.20.277/2019/77"),  # case
        ],
    )
    def test_the_wanted_one_is_a_part_of_the_stored_one(self, stored, wanted):
        assert identifier_contains(stored, wanted)

    @pytest.mark.parametrize(
        ("stored", "wanted"),
        [
            ("10.P.20.277/2019/77", "0.277/2019/77"),  # starts inside a number
            ("10.P.20.277/2019/777", "20.277/2019/77"),  # ends inside a number
            ("10.P.20.277/2019/77", "20.277/2018/77"),  # another year
            ("10.P.20.277/2019/77", "4.P"),  # not specific enough
            ("2023-01-15", "2023-01"),  # a date is not an identifier part
            ("10.P.20.277/2019/77", "nothing"),
            ("10.P.20.277/2019/77", ""),
        ],
    )
    def test_anything_else_is_not(self, stored, wanted):
        assert not identifier_contains(stored, wanted)

    def test_the_second_occurrence_is_found_when_the_first_does_not_qualify(self):
        """'77' inside '777' first, then a proper one."""
        assert identifier_contains(
            "A.20.277/2019/777 B.20.277/2019/77", "20.277/2019/77"
        )


class TestPattern:
    @pytest.mark.parametrize("weak", ["4.P", "2023-01", "123456", ""])
    def test_no_pattern_is_made_for_a_wish_that_is_too_weak(self, weak):
        with pytest.raises(ValueError, match="too weak"):
            partial_pattern(weak)

    def test_the_pattern_is_the_rule_as_a_regular_expression(self):
        pattern = partial_pattern(" P.20.277/2019/77. ")

        assert pattern == r"(^|[^[:alnum:]])p\.20\.277/2019/77([^0-9]|$)"


class TestCompact:
    """Separators ignored: the last resort for an identifier written with other separators."""

    @pytest.mark.parametrize(
        ("text", "compact"),
        [
            ("P.20103.2022.19", "p20103202219"),
            (
                "4.P.20.103/2022/19-ítélet",
                "4p20103202219tlet",
            ),  # accented letters drop out
            ("  A 1/2 ", "a12"),
            ("", ""),
        ],
    )
    def test_the_compact_form(self, text, compact):
        assert compact_identifier(text) == compact

    @pytest.mark.parametrize(
        ("stored", "wanted"),
        [
            (
                "4.P.20.103/2022/19",
                "P.20103.2022.19",
            ),  # dots for slashes (the q0008 case)
            ("4.P.20.103/2022/19-ítélet", "P.20103.2022.19"),
            ("10.P.20.277/2019/77", "20277/2019/77"),  # slashes only
            ("10.P.20.277/2019/77", "10 P 20 277 2019 77"),  # spaces for everything
            ("PREFIX-2024/00123", "2024.00123"),
        ],
    )
    def test_the_wanted_one_is_the_stored_one_with_other_separators(
        self, stored, wanted
    ):
        assert identifier_compact_contains(stored, wanted)

    @pytest.mark.parametrize(
        ("stored", "wanted"),
        [
            ("10.P.20.277/2019/777", "P.20277.2019.77"),  # more digits follow
            ("10.P.20.277/2018/77", "P.20277.2019.77"),  # another year
            (
                "10.P.20.277/2019/77",
                "0.277.2019.77",
            ),  # starts inside a number (digit-initial)
            ("10.P.20.277/2019/77", "4.P"),  # too weak
            ("10.P.20.277/2019/77", "123456"),
            ("10.P.20.277/2019/77", ""),
        ],
    )
    def test_anything_else_is_not(self, stored, wanted):
        assert not identifier_compact_contains(stored, wanted)

    def test_a_letter_initial_wish_may_follow_the_digits_of_a_series(self):
        """The leading '4' of '4.P...' is a series, the wish starts at the 'P'."""
        assert identifier_compact_contains("4.P.20.103/2022/19", "P.20103.2022.19")

    def test_a_digit_initial_wish_may_not_start_inside_a_number(self):
        assert not identifier_compact_contains("14.P.20.277/2019/77", "4P20277201977")
        assert identifier_compact_contains("4.P.20.277/2019/77", "4P20277201977")

    def test_the_pattern_is_the_rule_as_a_regular_expression(self):
        assert compact_pattern("P.20103.2022.19") == "p20103202219([^0-9]|$)"
        assert compact_pattern("20.277/2019/77") == "(^|[^0-9])20277201977([^0-9]|$)"

    def test_no_pattern_is_made_for_a_wish_that_is_too_weak(self):
        with pytest.raises(ValueError, match="too weak"):
            compact_pattern("4.P")
