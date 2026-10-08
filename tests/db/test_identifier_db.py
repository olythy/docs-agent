"""DB-gated tests of the identifier type: the SQL must be the Python rule, and the migration safe.

``metadata/identifiers.py`` defines how identifiers are compared; the compiled SQL repeats
the rule on the stored value. They are written twice, so a test has to keep them equal:
every pair of a stored value and a wished one below is decided by both, and they must
agree. Requires AGENT_ENV=test -- see tests/db/conftest.py.
"""

from datetime import date

import psycopg2
import pytest

from config import settings
from document_store import DocumentStore
from metadata.clock import FixedClock
from metadata.compiler import PlanCompiler
from metadata.date_ranges import DateRangeResolver
from metadata.identifiers import identifier_matches
from metadata.plan import parse_plan
from migrations import base  # noqa: F401  (the package must be importable)
from models import (
    Document,
    KeyStatus,
    MetaKey,
    MetaSource,
    MetaValue,
    ValueType,
)

pytestmark = [
    pytest.mark.db,
    pytest.mark.skipif(settings.AGENT_ENV != "test", reason="AGENT_ENV is not 'test'"),
]

DT = "court_decision"

#: Stored values, as written, with the awkward cases: spaces, case, full stops, suffixes,
#: a longer number, characters that mean something to LIKE, full-width and no-break forms.
STORED = [
    "4.P.20.409/2023/4",
    "4.P.20.409/2023/4-ítélet",
    "4.P.20.409/2023/40",
    "4.P.20.409/2023/4.",
    "  4.p.20.409/2023/4 ",
    "10. P. 20.277/2019/77.",
    "10.P.20.277/2019/77/ítélet",
    "A_B%1",
    "AXB%1",
    "ＡＢ１２３",
    "x y/1",
    "8.P.21.329/2024/12-III",
    "8.P.XI.21.329/2024/12.",
    "104.K.702.368/2022",
    "104.K.702.368/2022/22",
    "11.P.20.377/2023/11JAV",
]

WISHED = [
    "4.P.20.409/2023/4",
    "4.P.20.409/2023/40",
    "4.p.20.409/2023/4.",
    "10.P.20.277/2019/77",
    "10. P. 20.277/2019/77.",
    "A_B%1",
    "a_b",
    "ab123",
    "xy/1",
    "8.P.21.329/2024/12",
    "8.P.XI.21.329/2024/12",
    "104.K.702.368/2022",
    "104.K.702.368/2022/22",
    "11.P.20.377/2023/11",
    "nothing like these",
]


def _h(n: int) -> str:
    return f"{n:064x}"


@pytest.fixture
def world(db_conn):
    store = DocumentStore()
    store.ensure_type(DT)
    store.upsert_key(
        MetaKey(
            DT,
            "document_identifier",
            ValueType.IDENTIFIER,
            "d",
            status=KeyStatus.APPROVED,
        )
    )
    for n, value in enumerate(STORED):
        store.upsert_document(Document(_h(n), f"doc_{n}.docx", summary="s"))
        store.set_document_type(_h(n), DT)
        store.add_value(
            MetaValue(_h(n), "document_identifier", 1, MetaSource.LLM, value_text=value)
        )
    return store


def _documents_found(store, op, value):
    compiler = PlanCompiler(DateRangeResolver(FixedClock(date(2026, 10, 7))))
    plan = parse_plan(
        {
            "document_type": DT,
            "operation": "list",
            "filters": [{"key": "document_identifier", "op": op, "value": value}],
            "limit": 100,
        },
        [DT],
    )
    keys = store.list_keys(DT, KeyStatus.APPROVED)
    query = compiler.compile(plan, keys)
    rows = store.execute_query(query.sql, query.params)
    return {row[1] for row in rows}  # source_file


@pytest.mark.parametrize("wished", WISHED)
def test_the_sql_decides_exactly_as_the_python_rule_does(world, wished):
    found = _documents_found(world, "eq", wished)

    expected = {
        f"doc_{n}.docx" for n, v in enumerate(STORED) if identifier_matches(v, wished)
    }
    assert found == expected


def test_the_awkward_cases_are_really_in_play(world):
    """Guards the test above against passing because nothing ever matches."""
    assert _documents_found(world, "eq", "4.P.20.409/2023/4") == {
        "doc_0.docx",  # the same
        "doc_1.docx",  # a suffix
        "doc_3.docx",  # a full stop
        "doc_4.docx",  # case and spaces
    }  # and not doc_2 (".../40")
    assert _documents_found(world, "eq", "ab123") == {"doc_9.docx"}  # full-width
    assert _documents_found(world, "eq", "a_b%1") == {
        "doc_7.docx"
    }  # not AXB%1: _ is literal


def test_in_is_any_of_them(world):
    found = _documents_found(
        world, "in", ["10.P.20.277/2019/77", "104.K.702.368/2022/22"]
    )

    assert found == {"doc_5.docx", "doc_6.docx", "doc_14.docx"}


def test_contains_looks_in_the_normalised_form(world):
    found = _documents_found(world, "contains", "p.20.277")

    assert found == {"doc_5.docx", "doc_6.docx"}


def test_the_migration_widens_the_rule_and_its_down_is_safe_with_data(db_conn):
    """Down turns identifier keys back into text (the old rule would refuse them), up widens again."""
    from importlib import import_module

    migration = import_module("migrations.0007_allow_identifier_value_type")
    step = migration.AllowIdentifierValueType()
    DocumentStore().ensure_type("t")  # a key belongs to a document type
    with db_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO meta_keys (doc_type, key, value_type, description) "
            "VALUES ('t', 'k', 'identifier', 'd');"
        )
    try:
        step.down(db_conn)
        with db_conn.cursor() as cur:
            cur.execute("SELECT value_type FROM meta_keys WHERE key = 'k';")
            row = cur.fetchone()
            assert row is not None and row[0] == "text"
            with pytest.raises(psycopg2.errors.CheckViolation):
                cur.execute(
                    "INSERT INTO meta_keys (doc_type, key, value_type, description) "
                    "VALUES ('t', 'j', 'identifier', 'd');"
                )
        db_conn.rollback()  # leave the schema as the migrations made it
    finally:
        db_conn.rollback()


def test_after_the_migration_an_identifier_key_can_be_stored(db_conn):
    DocumentStore().ensure_type("t")
    with db_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO meta_keys (doc_type, key, value_type, description) "
            "VALUES ('t', 'k', 'identifier', 'd') RETURNING value_type;"
        )
        row = cur.fetchone()

    assert row is not None and row[0] == "identifier"


# ---- the resolver's lookup: DocumentStore.documents_with_identifiers --------------------


def _doc(
    store,
    n,
    doc_type,
    key,
    value,
    key_type=ValueType.IDENTIFIER,
    status=KeyStatus.APPROVED,
):
    """A document of a type, with one value under a key of the given type and status."""
    store.ensure_type(doc_type)
    store.upsert_key(MetaKey(doc_type, key, key_type, "d", status=status))
    store.upsert_document(Document(_h(n), f"d{n}.docx", summary="s"))
    store.set_document_type(_h(n), doc_type)
    store.add_value(MetaValue(_h(n), key, 1, MetaSource.LLM, value_text=value))


def _ids(store, rows, names):
    """(position, document_id) rows as {wanted: {file names}}."""
    by_name = dict(store.execute_query("SELECT id, source_file FROM documents", ()))
    out: dict[str, set[str]] = {}
    for position, document_id in rows:
        out.setdefault(names[position - 1], set()).add(by_name[document_id])
    return out


class TestDocumentsWithIdentifiers:
    def test_it_applies_the_same_rule_as_the_python_one(self, world):
        wanted = [w for w in WISHED if w.strip()]
        from metadata.identifiers import normalize_identifier

        normalised = list(dict.fromkeys(normalize_identifier(w) for w in wanted))

        rows = world.documents_with_identifiers(normalised)

        found = _ids(world, rows, normalised)
        for n in normalised:
            expected = {
                f"doc_{i}.docx"
                for i, v in enumerate(STORED)
                if identifier_matches(v, n)
            }
            assert found.get(n, set()) == expected

    def test_it_finds_the_document_by_a_written_variant(self, world):
        rows = world.documents_with_identifiers(
            ["8.p.21.329/2024/12", "8.p.xi.21.329/2024/12"]
        )

        assert _ids(world, rows, ["a", "b"]) == {
            "a": {"doc_11.docx"},
            "b": {"doc_12.docx"},
        }

    def test_a_longer_number_is_not_found_by_its_shorter_prefix(self, world):
        rows = world.documents_with_identifiers(["4.p.20.409/2023/4"])

        found = _ids(world, rows, ["w"])["w"]
        assert "doc_2.docx" not in found  # ".../40"
        assert {"doc_0.docx", "doc_1.docx"} <= found

    def test_each_pair_comes_once_even_when_a_document_has_several_matching_values(
        self, db_conn
    ):
        store = DocumentStore()
        _doc(store, 1, "t", "k", "AB-2024-001")
        store.add_value(  # a second written form of the same number, under the same key
            MetaValue(
                _h(1),
                "k",
                1,
                MetaSource.LLM,
                value_text="AB-2024-001-ítélet",
                ordinal=1,
            )
        )

        rows = store.documents_with_identifiers(["ab-2024-001"])

        document_id = store.execute_query("SELECT id FROM documents", ())[0][0]
        assert rows == [(1, document_id)]  # once, although two values match

    def test_nothing_is_asked_nothing_is_found(self, world):
        assert world.documents_with_identifiers([]) == []

    def test_an_empty_wish_matches_nothing(self, world):
        assert world.documents_with_identifiers([""]) == []

    def test_positions_follow_the_order_of_the_wishes(self, world):
        rows = world.documents_with_identifiers(
            ["104.k.702.368/2022/22", "10.p.20.277/2019/77"]
        )

        positions = {p for p, _ in rows}
        assert positions == {1, 2}
        assert [p for p, _ in rows] == sorted(p for p, _ in rows)

    def test_only_keys_of_type_identifier_count(self, db_conn):
        store = DocumentStore()
        _doc(store, 1, "t", "number_key", "AB-1", ValueType.IDENTIFIER)
        _doc(
            store, 2, "t", "free_text", "AB-1", ValueType.TEXT
        )  # the same text, another type

        rows = store.documents_with_identifiers(["ab-1"])

        assert _ids(store, rows, ["w"]) == {"w": {"d1.docx"}}

    def test_a_key_that_is_not_approved_does_not_count(self, db_conn):
        store = DocumentStore()
        _doc(store, 1, "t", "k_ok", "AB-1")
        _doc(store, 2, "t", "k_retired", "AB-1", status=KeyStatus.RETIRED)
        _doc(store, 3, "t", "k_proposed", "AB-1", status=KeyStatus.PROPOSED)

        rows = store.documents_with_identifiers(["ab-1"])

        assert _ids(store, rows, ["w"]) == {"w": {"d1.docx"}}

    def test_several_identifier_keys_of_one_type_are_all_searched(self, db_conn):
        store = DocumentStore()
        _doc(store, 1, "t", "case_id", "AB-1")
        store.upsert_key(
            MetaKey(
                "t", "invoice_id", ValueType.IDENTIFIER, "d", status=KeyStatus.APPROVED
            )
        )
        store.upsert_document(Document(_h(2), "d2.docx", summary="s"))
        store.set_document_type(_h(2), "t")
        store.add_value(
            MetaValue(_h(2), "invoice_id", 1, MetaSource.LLM, value_text="AB-1")
        )

        rows = store.documents_with_identifiers(["ab-1"])

        assert _ids(store, rows, ["w"]) == {"w": {"d1.docx", "d2.docx"}}

    def test_a_key_of_another_document_type_does_not_apply_to_a_document(self, db_conn):
        """The key must belong to the document's own type."""
        store = DocumentStore()
        _doc(store, 1, "t", "k", "AB-1")
        store.ensure_type("u")
        store.upsert_key(
            MetaKey("u", "other", ValueType.IDENTIFIER, "d", status=KeyStatus.APPROVED)
        )
        store.upsert_document(Document(_h(2), "d2.docx", summary="s"))
        store.set_document_type(_h(2), "u")
        store.add_value(
            MetaValue(_h(2), "k", 1, MetaSource.LLM, value_text="AB-1")
        )  # a key of "t"

        rows = store.documents_with_identifiers(["ab-1"])

        assert _ids(store, rows, ["w"]) == {"w": {"d1.docx"}}

    def test_a_resolver_over_the_real_store_end_to_end(self, world):
        from metadata.identifier_resolver import IdentifierResolver

        result = IdentifierResolver(world).resolve(
            ["8.P.XI.21.329/2024/12", "4.P.20.409/2023/40", "no/such/1"]
        )

        def id_of(name):
            return world.execute_query(
                "SELECT id FROM documents WHERE source_file = %s", (name,)
            )[0][0]

        assert result.documents["8.P.XI.21.329/2024/12"] == (id_of("doc_12.docx"),)
        # ".../40" is its own document
        assert result.documents["4.P.20.409/2023/40"] == (id_of("doc_2.docx"),)
        assert result.unresolved == ("no/such/1",)


# ---- the partial lookup: a wished identifier as a PART of a stored one ------------------

PART_STORED = [
    "10.P.20.277/2019/77",
    "14.P.20.277/2019/77",  # the same series and number at another office
    "10.P.20.277/2019/777",  # a longer number
    "4.P.20.487/2020/221-ítélet",
    "PREFIX-2024/00123",
    "10.P.20.277/2018/77",  # another year
    "2023-01-15-7",  # something with a date in it
    "HU-BUD-2024-17",
    "ÍTÉLET.20.100/2021/5",  # a non-ASCII letter right before a separator
    "x 20.277/2019/77",  # a space before the part
]

PART_WISHED = [
    "P.20.277/2019/77",
    "20.277/2019/77",
    "10. P. 20.277/2019/77.",
    "0.277/2019/77",  # starts inside a number
    "20.277/2019/7",  # ends inside a number
    "P.20.487/2020/221",
    "2024/00123",
    "BUD-2024-17",
    "20.100/2021/5",
    "4.P",  # too weak
    "2023-01",  # a date
    "123456",  # a bare short number
    "nothing like these",
]


@pytest.fixture
def parts(db_conn):
    store = DocumentStore()
    store.ensure_type(DT)
    store.upsert_key(
        MetaKey(
            DT,
            "document_identifier",
            ValueType.IDENTIFIER,
            "d",
            status=KeyStatus.APPROVED,
        )
    )
    for n, value in enumerate(PART_STORED):
        store.upsert_document(Document(_h(n), f"part_{n}.docx", summary="s"))
        store.set_document_type(_h(n), DT)
        store.add_value(
            MetaValue(_h(n), "document_identifier", 1, MetaSource.LLM, value_text=value)
        )
    return store


class TestPartialLookup:
    @pytest.mark.parametrize("wished", PART_WISHED)
    def test_the_sql_decides_exactly_as_the_python_rule_does(self, parts, wished):
        from metadata.identifiers import (
            identifier_contains,
            is_specific_enough,
            partial_pattern,
        )

        expected = {
            f"part_{n}.docx"
            for n, v in enumerate(PART_STORED)
            if identifier_contains(v, wished)
        }
        if not is_specific_enough(wished):
            assert expected == set()  # the Python rule says no, and no pattern is made
            with pytest.raises(ValueError, match="too weak"):
                partial_pattern(wished)
            return

        rows = parts.documents_matching_identifier_patterns([partial_pattern(wished)])
        by_id = dict(parts.execute_query("SELECT id, source_file FROM documents", ()))
        assert {by_id[document_id] for _, document_id in rows} == expected

    def test_the_cases_that_matter_are_really_in_play(self, parts):
        from metadata.identifiers import partial_pattern

        def files(wished):
            rows = parts.documents_matching_identifier_patterns(
                [partial_pattern(wished)]
            )
            by_id = dict(
                parts.execute_query("SELECT id, source_file FROM documents", ())
            )
            return {by_id[i] for _, i in rows}

        assert files("P.20.277/2019/77") == {"part_0.docx", "part_1.docx"}
        # not part_9: "x 20.277/..." has an x straight before the number, no separator
        assert files("20.277/2019/77") == {"part_0.docx", "part_1.docx"}
        assert files("0.277/2019/77") == set()  # not inside a number
        assert files("20.277/2019/7") == set()  # nor ending inside one
        assert files("2024/00123") == {"part_4.docx"}
        assert files("20.100/2021/5") == {
            "part_8.docx"
        }  # a non-ASCII letter before the separator

    def test_positions_follow_the_order_of_the_patterns(self, parts):
        from metadata.identifiers import partial_pattern

        rows = parts.documents_matching_identifier_patterns(
            [partial_pattern("2024/00123"), partial_pattern("P.20.487/2020/221")]
        )

        assert [p for p, _ in rows] == [1, 2]

    def test_no_patterns_find_nothing(self, parts):
        assert parts.documents_matching_identifier_patterns([]) == []

    def test_a_resolver_over_the_real_store_tells_exact_from_partial(self, parts):
        from metadata.identifier_resolver import IdentifierResolver

        result = IdentifierResolver(parts).resolve(
            ["10.P.20.277/2019/77", "P.20.487/2020/221", "99.P.99.999/2099/1"]
        )

        by_id = dict(parts.execute_query("SELECT id, source_file FROM documents", ()))
        assert {by_id[i] for i in result.documents["10.P.20.277/2019/77"]} == {
            "part_0.docx"
        }
        assert {by_id[i] for i in result.documents["P.20.487/2020/221"]} == {
            "part_3.docx"
        }
        assert result.partial == ("P.20.487/2020/221",)  # found only as a part
        assert result.unresolved == ("99.P.99.999/2099/1",)


COMPACT_WISHED = [
    "P.20103.2022.19",
    "P.20.103/2022/19",
    "20277/2019/77",
    "10 P 20 277 2019 77",
    "2024.00123",
    "P.20277.2019.77",
    "P.20277.2019.7",  # ends inside a number
    "0.277.2019.77",  # starts inside a number
    "4.P",  # too weak
    "123456",  # a short bare number
    "nothing like these",
]


class TestCompactLookup:
    @pytest.fixture
    def compact_store(self, db_conn):
        store = DocumentStore()
        store.ensure_type(DT)
        store.upsert_key(
            MetaKey(
                DT,
                "document_identifier",
                ValueType.IDENTIFIER,
                "d",
                status=KeyStatus.APPROVED,
            )
        )
        stored = [
            "4.P.20.103/2022/19-ítélet",
            "10.P.20.277/2019/77",
            "10.P.20.277/2019/777",
            "10.P.20.277/2018/77",
            "PREFIX-2024/00123",
            "14.P.20.277/2019/77",
        ]
        for n, value in enumerate(stored):
            store.upsert_document(Document(_h(n), f"c_{n}.docx", summary="s"))
            store.set_document_type(_h(n), DT)
            store.add_value(
                MetaValue(
                    _h(n), "document_identifier", 1, MetaSource.LLM, value_text=value
                )
            )
        return store, stored

    @pytest.mark.parametrize("wished", COMPACT_WISHED)
    def test_the_sql_decides_exactly_as_the_python_rule_does(
        self, compact_store, wished
    ):
        from metadata.identifiers import (
            compact_pattern,
            identifier_compact_contains,
            is_specific_enough,
        )

        store, stored = compact_store
        expected = {
            f"c_{n}.docx"
            for n, v in enumerate(stored)
            if identifier_compact_contains(v, wished)
        }
        if not is_specific_enough(wished):
            assert expected == set()
            with pytest.raises(ValueError, match="too weak"):
                compact_pattern(wished)
            return

        rows = store.documents_matching_identifier_patterns(
            [compact_pattern(wished)], ignoring_separators=True
        )
        by_id = dict(store.execute_query("SELECT id, source_file FROM documents", ()))
        assert {by_id[i] for _, i in rows} == expected

    def test_the_cases_that_matter_are_really_in_play(self, compact_store):
        from metadata.identifiers import compact_pattern

        store, _ = compact_store
        by_id = dict(store.execute_query("SELECT id, source_file FROM documents", ()))

        def files(wished):
            rows = store.documents_matching_identifier_patterns(
                [compact_pattern(wished)], ignoring_separators=True
            )
            return {by_id[i] for _, i in rows}

        assert files("P.20103.2022.19") == {"c_0.docx"}  # dots for slashes
        assert files("P.20277.2019.77") == {
            "c_1.docx",
            "c_5.docx",
        }  # the series left out
        assert files("P.20277.2019.7") == set()  # not inside "...77" or "...777"

    def test_the_default_still_matches_the_normalised_value(self, compact_store):
        """Without the flag a separator-less pattern finds nothing in the normalised value."""
        from metadata.identifiers import compact_pattern

        store, _ = compact_store

        assert (
            store.documents_matching_identifier_patterns(
                [compact_pattern("P.20103.2022.19")]
            )
            == []
        )

    def test_a_resolver_over_the_real_store_reports_it_as_compact(self, compact_store):
        from metadata.identifier_resolver import IdentifierResolver

        store, _ = compact_store

        result = IdentifierResolver(store).resolve(
            ["P.20103.2022.19", "10.P.20.277/2019/77"]
        )

        assert result.compact == ("P.20103.2022.19",)
        assert result.approximate == (
            "P.20103.2022.19",
        )  # the other was found as written
