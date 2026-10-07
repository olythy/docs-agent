"""The identifier resolver (no database): what it asks the source and how it reports."""

from collections.abc import Sequence

import pytest

from metadata.identifier_resolver import IdentifierResolver, ResolvedIdentifiers
from metadata.identifiers import identifier_matches


class FakeSource:
    """Applies the real rule in Python to a small table of documents."""

    def __init__(self, documents: dict[int, list[str]]):
        self.documents = documents
        self.asked: list[list[str]] = []

    def documents_with_identifiers(self, wanted: Sequence[str]):
        self.asked.append(list(wanted))
        return [
            (position, doc_id)
            for position, w in enumerate(wanted, start=1)
            for doc_id, values in sorted(self.documents.items())
            if any(identifier_matches(v, w) for v in values)
        ]


DOCS = {
    1: ["4.P.20.409/2023/4"],
    2: ["27.P.20.339/2021/37", "27.P.20.339/2021/37-ítélet"],
    3: ["8.P.21.329/2024/12-III", "8.P.XI.21.329/2024/12."],
    4: ["HU001"],
    5: ["HU001-A"],  # a second document with a number that continues it
}


def resolve(*identifiers, documents=None):
    source = FakeSource(documents or DOCS)
    return IdentifierResolver(source).resolve(identifiers), source


class TestResolve:
    def test_an_identifier_resolves_to_the_document_that_carries_it(self):
        result, _ = resolve("4.P.20.409/2023/4")

        assert result.documents == {"4.P.20.409/2023/4": (1,)}
        assert result.document_ids == (1,)
        assert result.unresolved == ()

    def test_a_written_variant_resolves_too(self):
        result, _ = resolve("  4.p.20.409/2023/4. ", "8.P.21.329/2024/12")

        assert result.documents["  4.p.20.409/2023/4. "] == (1,)
        assert result.documents["8.P.21.329/2024/12"] == (3,)

    def test_both_numbers_of_a_decision_find_it(self):
        """The header form and the stated case number belong to one document."""
        result, _ = resolve("8.P.21.329/2024/12", "8.P.XI.21.329/2024/12")

        assert result.document_ids == (3,)

    def test_a_number_that_several_documents_carry_is_reported_and_none_is_chosen(self):
        result, _ = resolve("HU001")

        assert result.documents["HU001"] == (4, 5)
        assert result.ambiguous == ("HU001",)

    def test_an_identifier_no_document_carries_is_unresolved(self):
        result, _ = resolve("99.P.99.999/2099/1", "4.P.20.409/2023/4")

        assert result.unresolved == ("99.P.99.999/2099/1",)
        assert result.documents["99.P.99.999/2099/1"] == ()
        assert result.document_ids == (1,)  # the resolved one is kept

    def test_a_longer_number_is_not_the_wanted_one(self):
        result, _ = resolve(
            "4.P.20.409/2023"
        )  # a prefix: matches ".../4" (a non-digit follows)

        assert result.documents["4.P.20.409/2023"] == (1,)
        none, _ = resolve("4.P.20.409/20")  # followed by a digit: another number
        assert none.unresolved == ("4.P.20.409/20",)

    def test_the_union_is_sorted_and_has_each_document_once(self):
        result, _ = resolve(
            "27.P.20.339/2021/37", "27.P.20.339/2021", "4.P.20.409/2023/4"
        )

        assert result.document_ids == (1, 2)

    def test_the_identifiers_come_back_in_the_order_they_were_asked(self):
        result, _ = resolve("99/1", "4.P.20.409/2023/4", "88/2")

        assert list(result.documents) == ["99/1", "4.P.20.409/2023/4", "88/2"]


class TestWhatIsAsked:
    def test_one_lookup_for_all_identifiers_with_normalised_values(self):
        _, source = resolve("4.P.20.409/2023/4.", "27.P.20.339/2021/37")

        assert source.asked == [["4.p.20.409/2023/4", "27.p.20.339/2021/37"]]

    def test_the_same_number_written_twice_is_asked_once(self):
        result, source = resolve("4.P.20.409/2023/4", "4.p.20.409/2023/4.")

        assert source.asked == [["4.p.20.409/2023/4"]]
        assert result.documents["4.P.20.409/2023/4"] == (1,)
        assert result.documents["4.p.20.409/2023/4."] == (1,)

    @pytest.mark.parametrize("junk", ["", "  ", " ./ ", "--"])
    def test_nothing_but_punctuation_resolves_to_nothing_and_is_not_asked(self, junk):
        result, source = resolve(junk)

        assert result.unresolved == (junk,)
        assert source.asked == []  # no lookup at all

    def test_no_identifiers_means_no_lookup(self):
        result, source = resolve()

        assert result == ResolvedIdentifiers({}) and source.asked == []


def test_an_empty_result_has_no_documents_and_nothing_unresolved():
    empty = ResolvedIdentifiers({})

    assert (empty.document_ids, empty.unresolved, empty.ambiguous) == ((), (), ())
