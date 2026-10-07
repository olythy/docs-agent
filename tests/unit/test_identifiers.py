"""The rule for comparing identifiers as they are written (pure)."""

import pytest

from metadata.identifiers import identifier_matches, normalize_identifier


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

    def test_an_empty_wish_matches_nothing(self):
        assert not identifier_matches("4.P.20.409/2023/4", "")
        assert not identifier_matches("4.P.20.409/2023/4", " ./ ")
