"""Tests for metadata.date_parsers and metadata.verification, with real-shaped court text."""

from datetime import date
from decimal import Decimal

import pytest

from metadata.date_parsers import HungarianDateParser
from metadata.verification import EvidenceVerifier, numbers_in
from models import KeyStatus, MetaKey, ValueType

TEXT = (
    "A Budapest Környéki Törvényszék kötelezte az alperest 57.709.272,- Ft megváltási ár "
    "megfizetésére és 629 920 Ft perköltségre. Budapest, 2024. december 16. dr. Példa Béla s.k. bíró"
)


def _key(value_type, allowed=None):
    return MetaKey(
        "court_decision",
        "k",
        value_type,
        "d",
        allowed_values=allowed,
        status=KeyStatus.APPROVED,
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Budapest, 2024. december 16.", {date(2024, 12, 16)}),
        ("kelt 2022.05.12. napján", {date(2022, 5, 12)}),
        ("2024-12-16", {date(2024, 12, 16)}),
        # the older written form that the first extractor missed
        ("Balassagyarmat, 2021. év február 18. napján", {date(2021, 2, 18)}),
        ("Békéscsaba, 2024. év május hó 22. napján", {date(2024, 5, 22)}),
        ("2023. február 30.", set()),  # not a real day
        ("nincs itt dátum", set()),
    ],
)
def test_hungarian_date_parser(text, expected):
    assert HungarianDateParser().dates_in(text) == expected


def test_numbers_in_reads_thousands_separators_and_decimal_commas():
    assert numbers_in(TEXT) >= {Decimal(57709272), Decimal(629920)}
    assert numbers_in("12,5 százalék") == {Decimal("12.5")}


def test_a_date_that_the_quote_states_is_verified():
    result = EvidenceVerifier(TEXT).verify(
        _key(ValueType.DATE), "2024-12-16", "Budapest, 2024. december 16."
    )

    assert result.quote_exact and result.value_derivable and result.verified


def test_a_date_that_the_quote_does_not_state_is_not_verified():
    result = EvidenceVerifier(TEXT).verify(
        _key(ValueType.DATE), "2023-12-16", "Budapest, 2024. december 16."
    )

    assert result.quote_exact  # the quote is real ...
    assert not result.value_derivable  # ... but it does not say that date
    assert not result.verified


def test_an_invented_quote_is_not_verified_even_if_the_value_fits_it():
    result = EvidenceVerifier(TEXT).verify(
        _key(ValueType.DATE), "2020-01-01", "Budapest, 2020. január 1."
    )

    assert result.value_derivable and not result.quote_loose
    assert not result.verified


def test_a_number_is_verified_against_its_quote():
    key = _key(ValueType.NUMBER)
    verifier = EvidenceVerifier(TEXT)

    assert verifier.verify(key, "57709272", "57.709.272,- Ft megváltási ár").verified
    assert verifier.verify(key, "629920", "629 920 Ft perköltségre").verified
    assert not verifier.verify(key, "190500", "629 920 Ft perköltségre").verified


def test_a_categorical_value_must_be_in_the_allowed_list():
    key = _key(ValueType.TEXT, allowed=("judgment", "order", "other"))
    verifier = EvidenceVerifier("meghozta a következő I T É L E T E T")

    assert verifier.verify(key, "judgment", "I T É L E T E T").verified
    # the letter-spaced Hungarian wording itself is not a canonical token
    assert not verifier.verify(key, "I T É L E T E T", "I T É L E T E T").verified


def test_the_loose_quote_check_tolerates_whitespace_and_accent_noise():
    verifier = EvidenceVerifier("Budapest   Környéki\nTörvényszék")

    result = verifier.verify(_key(ValueType.TEXT), "x", "Budapest Kornyeki Torvenyszek")

    assert result.quote_loose and not result.quote_exact


def test_a_missing_value_or_quote_confirms_nothing():
    verifier = EvidenceVerifier(TEXT)

    assert not verifier.verify(_key(ValueType.TEXT), None, "x").verified
    assert not verifier.verify(_key(ValueType.TEXT), "x", None).verified
    assert not verifier.verify(_key(ValueType.TEXT), "x", "").verified
