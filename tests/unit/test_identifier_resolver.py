"""The identifier resolver (no database): what it asks the source and how it reports."""

import re
from collections.abc import Sequence
from typing import ClassVar

import pytest

from metadata.identifier_resolver import IdentifierResolver, ResolvedIdentifiers
from metadata.identifiers import (
    compact_identifier,
    identifier_matches,
    normalize_identifier,
)


class FakeSource:
    """Applies the real rule in Python to a small table of documents."""

    def __init__(self, documents: dict[int, list[str]]):
        self.documents = documents
        self.asked: list[list[str]] = []
        self.asked_patterns: list[list[str]] = []

    def documents_with_identifiers(self, wanted: Sequence[str]):
        self.asked.append(list(wanted))
        return [
            (position, doc_id)
            for position, w in enumerate(wanted, start=1)
            for doc_id, values in sorted(self.documents.items())
            if any(identifier_matches(v, w) for v in values)
        ]

    def documents_matching_identifier_patterns(
        self, patterns: Sequence[str], ignoring_separators: bool = False
    ):
        self.asked_patterns.append(list(patterns))
        subject = compact_identifier if ignoring_separators else normalize_identifier
        out = []
        for position, pattern in enumerate(patterns, start=1):
            # PostgreSQL's POSIX class, as Python's re spells it; the real equality of the
            # two spellings is held by the database test, not by this fake.
            compiled = re.compile(pattern.replace("[^[:alnum:]]", r"[\W_]"))
            for doc_id, values in sorted(self.documents.items()):
                if any(compiled.search(subject(v)) for v in values):
                    out.append((position, doc_id))
        return out


DOCS = {
    1: ["4.P.20.409/2023/4"],
    2: ["27.P.20.339/2021/37", "27.P.20.339/2021/37-ítélet"],
    3: ["8.P.21.329/2024/12-III", "8.P.XI.21.329/2024/12."],
    4: ["HU-2024-001"],
    5: ["HU-2024-001-A"],  # a second document with a number that continues it
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
        result, _ = resolve("HU-2024-001")

        assert result.documents["HU-2024-001"] == (4, 5)
        assert result.ambiguous == ("HU-2024-001",)

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


class TestPartialIdentifiers:
    """An identifier the first step finds nothing for is looked for as a part of a stored one."""

    DOCS: ClassVar[dict[int, list[str]]] = {
        1: ["4.P.20.409/2023/4"],
        2: ["14.P.20.409/2023/4"],  # the same series and number at another office
        3: ["10.P.20.277/2019/77"],
    }

    def resolve(self, *identifiers):
        source = FakeSource(self.DOCS)
        return IdentifierResolver(source).resolve(identifiers), source

    def test_the_end_of_an_identifier_finds_the_document(self):
        result, _ = self.resolve("P.20.277/2019/77")

        assert result.documents["P.20.277/2019/77"] == (3,)
        assert result.partial == ("P.20.277/2019/77",)  # and it says it was only a part

    def test_a_part_shared_by_several_documents_gives_all_of_them(self):
        result, _ = self.resolve("P.20.409/2023/4")

        assert result.documents["P.20.409/2023/4"] == (1, 2)
        assert result.ambiguous == ("P.20.409/2023/4",)

    def test_an_identifier_found_as_written_is_never_looked_for_as_a_part(self):
        result, source = self.resolve("4.P.20.409/2023/4")

        assert result.documents["4.P.20.409/2023/4"] == (1,)  # not (1, 2)
        assert result.partial == ()
        assert source.asked_patterns == []  # the second step did not run at all

    def test_only_the_identifiers_still_unresolved_go_to_the_second_step(self):
        _, source = self.resolve("4.P.20.409/2023/4", "P.20.277/2019/77")

        assert len(source.asked_patterns) == 1 and len(source.asked_patterns[0]) == 1

    def test_a_weak_identifier_is_not_looked_for_as_a_part(self):
        result, source = self.resolve("4.P", "2023-01")

        assert result.unresolved == ("4.P", "2023-01")
        assert source.asked_patterns == []

    def test_an_identifier_that_is_nowhere_stays_unresolved(self):
        result, _ = self.resolve("P.99.999/2099/1")

        assert result.unresolved == ("P.99.999/2099/1",)
        assert result.partial == ()

    def test_exact_and_partial_resolutions_are_told_apart(self):
        result, _ = self.resolve("4.P.20.409/2023/4", "P.20.277/2019/77")

        assert result.partial == ("P.20.277/2019/77",)
        assert result.document_ids == (1, 3)


class TestCompactIdentifiers:
    """The last step: the same number written with other separators."""

    DOCS: ClassVar[dict[int, list[str]]] = {
        1: ["4.P.20.103/2022/19-ítélet"],
        2: ["6.P.20.277/2019/77"],
    }

    def resolve(self, *identifiers):
        source = FakeSource(self.DOCS)
        return IdentifierResolver(source).resolve(identifiers), source

    def test_dots_for_slashes_find_the_document_and_it_is_reported(self):
        result, _ = self.resolve("P.20103.2022.19")  # the q0008 case

        assert result.documents["P.20103.2022.19"] == (1,)
        assert result.compact == ("P.20103.2022.19",)
        assert result.partial == ()
        assert result.approximate == ("P.20103.2022.19",)

    def test_it_is_the_last_resort_not_tried_when_the_part_step_found_something(self):
        _, source = self.resolve("P.20.277/2019/77")  # a part of a stored one

        assert len(source.asked_patterns) == 1  # the part step only

    def test_it_is_tried_only_for_what_the_earlier_steps_did_not_find(self):
        _, source = self.resolve("P.20.277/2019/77", "P.20103.2022.19")

        assert len(source.asked_patterns) == 2
        assert (
            len(source.asked_patterns[1]) == 1
        )  # one identifier left for the last step

    def test_a_weak_identifier_does_not_get_that_far(self):
        result, source = self.resolve("4.P")

        assert result.unresolved == ("4.P",)
        assert source.asked_patterns == []

    def test_an_identifier_found_as_written_is_not_approximate(self):
        result, _ = self.resolve("4.P.20.103/2022/19")

        assert result.documents["4.P.20.103/2022/19"] == (1,)
        assert result.approximate == ()
