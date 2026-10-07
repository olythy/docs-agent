"""Tests for metadata.compiler.PlanCompiler (pure: no database)."""

from datetime import date
from decimal import Decimal

import pytest

from metadata.clock import FixedClock
from metadata.compiler import PlanCompiler
from metadata.date_ranges import DateRangeResolver
from metadata.plan import Filter, FilterOp, Operation, PlanError, QueryPlan, parse_plan
from models import KeyStatus, MetaKey, ValueType

DT = "court_decision"


def _k(name, vtype, allowed=None, status=KeyStatus.APPROVED, version=1):
    return MetaKey(
        DT, name, vtype, "d", allowed_values=allowed, status=status, version=version
    )


KEYS = [
    _k("issuing_body", ValueType.TEXT),
    _k("decision_date", ValueType.DATE),
    _k("document_kind", ValueType.TEXT, allowed=("judgment", "order", "other")),
    _k("legal_costs", ValueType.NUMBER),
    _k("appealed", ValueType.BOOL),
    _k("document_identifier", ValueType.IDENTIFIER),
    _k("draft_key", ValueType.TEXT, status=KeyStatus.PROPOSED),
]


@pytest.fixture
def compiler():
    return PlanCompiler(DateRangeResolver(FixedClock(date(2026, 10, 4))))


def _plan(**kw):
    kw.setdefault("operation", "count")
    kw.setdefault("document_type", DT)
    return parse_plan(kw, [DT, "invoice"])


def test_every_user_value_is_a_bound_parameter_never_part_of_the_sql(compiler):
    evil = "x'; DROP TABLE documents; --"
    compiled = compiler.compile(
        _plan(filters=[{"key": "issuing_body", "op": "eq", "value": evil}]), KEYS
    )

    assert evil not in compiled.sql
    assert evil in compiled.params
    assert "issuing_body" not in compiled.sql  # the key name is a parameter too
    assert "issuing_body" in compiled.params
    assert compiled.sql.count("%s") == len(compiled.params)


def test_a_count_with_a_text_and_a_date_filter(compiler):
    """The question that started the design: last October, at one court."""
    compiled = compiler.compile(
        _plan(
            filters=[
                {"key": "issuing_body", "op": "eq", "value": "Debreceni Ítélőtábla"},
                {
                    "key": "decision_date",
                    "op": "between",
                    "value": {"kind": "calendar", "year_offset": -1, "month": 10},
                },
            ]
        ),
        KEYS,
    )

    assert compiled.sql.startswith("SELECT count(*) FROM documents d WHERE")
    assert compiled.sql.count("EXISTS") == 2
    assert compiled.params == (
        "court_decision",
        "issuing_body", "Debreceni Ítélőtábla",
        "decision_date", date(2025, 10, 1), date(2025, 10, 31),
    )  # fmt: skip
    assert compiled.explanation == (
        "court_decision documents: issuing_body = 'Debreceni Ítélőtábla' AND "
        "decision_date between 2025-10-01 and 2025-10-31"
    )


def test_a_plan_without_filters_counts_the_documents_of_its_type(compiler):
    compiled = compiler.compile(_plan(), KEYS)

    assert compiled.sql == (
        "SELECT count(*) FROM documents d "
        "WHERE (d.document_type IS NOT NULL AND d.document_type = %s)"
    )
    assert compiled.params == ("court_decision",)
    assert compiled.explanation == "court_decision documents"


def test_documents_without_a_type_are_the_unknown_of_a_plan_without_filters(compiler):
    """An untyped document might be of this type, so a count says '+K unknown'."""
    compiled = compiler.compile(_plan(), KEYS)

    assert compiled.unknown_sql is not None
    assert "d.document_type IS NULL OR d.document_type = %s" in compiled.unknown_sql
    assert compiled.unknown_params == ("court_decision", "court_decision")


def test_a_lookup_that_names_no_type_reads_everything_and_has_nothing_unknown(compiler):
    compiled = compiler.compile(_plan(document_type=None, operation="lookup"), KEYS)

    assert compiled.sql == "SELECT count(*) FROM documents d WHERE TRUE"
    assert compiled.unknown_sql is None
    assert compiled.explanation == "all documents"
    assert compiled.selection.sql == "SELECT d.id FROM documents d WHERE TRUE"


def test_the_type_condition_is_a_bound_parameter_and_other_types_keys_are_unusable(
    compiler,
):
    invoice_keys = [
        MetaKey(
            "invoice", "issuing_body", ValueType.TEXT, "d", status=KeyStatus.APPROVED
        )
    ]
    plan = _plan(
        document_type="invoice",
        filters=[{"key": "issuing_body", "op": "eq", "value": "x"}],
    )

    compiled = compiler.compile(plan, [*KEYS, *invoice_keys])

    assert "invoice" not in compiled.sql and "invoice" in compiled.params
    with pytest.raises(PlanError, match="unknown or unapproved key"):
        compiler.compile(plan, KEYS)  # the court keys do not belong to an invoice plan


def test_the_unknown_query_finds_documents_that_might_match_but_cannot_be_decided(
    compiler,
):
    compiled = compiler.compile(
        _plan(filters=[{"key": "issuing_body", "op": "eq", "value": "X"}]), KEYS
    )

    assert compiled.unknown_sql is not None
    assert "document_meta_status" in compiled.unknown_sql
    assert "'present', 'confirmed_absent'" in compiled.unknown_sql
    # "might match" = (satisfied OR the key is not known for this document) for every filter,
    # and "cannot be decided" = NOT (all filters satisfied)
    assert "OR NOT EXISTS (SELECT 1 FROM document_meta_status" in compiled.unknown_sql
    assert " AND NOT (" in compiled.unknown_sql
    assert compiled.unknown_sql.count("%s") == len(compiled.unknown_params)
    # the status check uses the key's *current* version, so an old extraction counts as unknown
    assert 1 in compiled.unknown_params


def test_the_known_check_follows_the_keys_version():
    versioned = [_k("issuing_body", ValueType.TEXT, version=3)]
    compiled = PlanCompiler(DateRangeResolver(FixedClock(date(2026, 10, 4)))).compile(
        _plan(filters=[{"key": "issuing_body", "op": "eq", "value": "X"}]), versioned
    )

    assert 3 in compiled.unknown_params


def test_list_and_overview_return_documents_with_a_total_and_lookup_only_counts(
    compiler,
):
    listed = compiler.compile(_plan(operation="list", limit=7), KEYS)
    overview = compiler.compile(_plan(operation="overview"), KEYS)
    lookup = compiler.compile(_plan(operation="lookup"), KEYS)

    assert "d.content_hash, d.source_file FROM" in listed.sql and listed.params[-1] == 7
    assert "d.summary" in overview.sql and "d.summary" not in listed.sql
    # a lookup does not fetch documents: a retrieval is restricted with the
    # selection instead, so there is no list to cap
    assert lookup.sql.startswith(
        "SELECT count(*) FROM documents d WHERE (d.document_type"
    )
    assert lookup.count_sql is None
    assert all(c.count_sql for c in (listed, overview))
    assert compiler.compile(_plan(), KEYS).count_sql is None  # a count is its own total


def test_a_sum_totals_a_number_key_over_the_matching_documents(compiler):
    compiled = compiler.compile(
        _plan(
            operation="sum",
            sum_key="legal_costs",
            filters=[{"key": "document_kind", "op": "eq", "value": "judgment"}],
        ),
        KEYS,
    )

    assert "sum(m.value_number)" in compiled.sql
    assert compiled.params[0] == "legal_costs" and "judgment" in compiled.params


def test_a_grouped_count_breaks_down_by_a_text_key(compiler):
    compiled = compiler.compile(_plan(group_by="document_kind", limit=5), KEYS)

    assert "GROUP BY 1 ORDER BY 2 DESC" in compiled.sql
    assert compiled.params == ("document_kind", "court_decision", 5)
    assert (
        compiled.count_sql is not None
    )  # the groups need not add up to the number of documents


def test_text_operators(compiler):
    ne = compiler.compile(
        _plan(filters=[{"key": "issuing_body", "op": "ne", "value": "A"}]), KEYS
    )
    many = compiler.compile(
        _plan(filters=[{"key": "issuing_body", "op": "in", "value": ["A", "B"]}]), KEYS
    )
    like = compiler.compile(
        _plan(filters=[{"key": "issuing_body", "op": "contains", "value": "100%_ok"}]),
        KEYS,
    )

    assert "<> %s" in ne.sql
    assert "= ANY(%s)" in many.sql and ["A", "B"] in many.params
    assert "ILIKE %s ESCAPE" in like.sql
    assert (
        "%100\\%\\_ok%" in like.params
    )  # the user's % and _ are escaped, not wildcards


def test_number_operators(compiler):
    gt = compiler.compile(
        _plan(filters=[{"key": "legal_costs", "op": "gt", "value": 5000000}]), KEYS
    )
    between = compiler.compile(
        _plan(
            filters=[{"key": "legal_costs", "op": "between", "value": [100, "250.5"]}]
        ),
        KEYS,
    )

    assert "> %s" in gt.sql and Decimal(5000000) in gt.params
    assert "BETWEEN %s AND %s" in between.sql and Decimal("250.5") in between.params


@pytest.mark.parametrize(
    ("op", "expected_param", "symbol"),
    [
        ("gt", date(2025, 10, 31), ">"),
        ("gte", date(2025, 10, 1), ">="),
        ("lt", date(2025, 10, 1), "<"),
        ("lte", date(2025, 10, 31), "<="),
    ],
)
def test_comparing_a_date_against_a_whole_month(compiler, op, expected_param, symbol):
    """ "after October 2025" means after its last day; "from October 2025" from its first."""
    compiled = compiler.compile(
        _plan(
            filters=[
                {
                    "key": "decision_date",
                    "op": op,
                    "value": {"kind": "calendar", "year": 2025, "month": 10},
                }
            ]
        ),
        KEYS,
    )

    assert f"{symbol} %s" in compiled.sql and expected_param in compiled.params


def test_an_iso_date_string_is_a_one_day_range(compiler):
    compiled = compiler.compile(
        _plan(filters=[{"key": "decision_date", "op": "eq", "value": "2024-12-16"}]),
        KEYS,
    )

    assert compiled.params[-2:] == (date(2024, 12, 16), date(2024, 12, 16))


def test_a_bool_filter(compiler):
    compiled = compiler.compile(
        _plan(filters=[{"key": "appealed", "op": "eq", "value": True}]), KEYS
    )

    assert True in compiled.params
    assert compiled.explanation == "court_decision documents: appealed is true"


@pytest.mark.parametrize(
    ("flt", "message"),
    [
        ({"key": "no_such", "op": "eq", "value": "x"}, "unknown or unapproved key"),
        (
            {"key": "draft_key", "op": "eq", "value": "x"},
            "unknown or unapproved key",
        ),  # proposed, not approved
        ({"key": "issuing_body", "op": "gt", "value": "x"}, "does not fit text key"),
        (
            {"key": "legal_costs", "op": "contains", "value": 5},
            "does not fit number key",
        ),
        (
            {"key": "decision_date", "op": "ne", "value": "2024-01-01"},
            "does not fit date key",
        ),
        ({"key": "appealed", "op": "gt", "value": True}, "does not fit bool key"),
        (
            {"key": "document_kind", "op": "eq", "value": "verdict"},
            "not an allowed value",
        ),
        (
            {"key": "document_kind", "op": "in", "value": ["judgment", "verdict"]},
            "not an allowed value",
        ),
        ({"key": "issuing_body", "op": "eq", "value": ""}, "non-empty string"),
        ({"key": "issuing_body", "op": "in", "value": []}, "non-empty list"),
        ({"key": "legal_costs", "op": "gt", "value": "a lot"}, "not a number"),
        ({"key": "legal_costs", "op": "gt", "value": True}, "needs a number"),
        ({"key": "legal_costs", "op": "between", "value": [9, 1]}, "low <= high"),
        ({"key": "legal_costs", "op": "between", "value": [1]}, r"\[low, high\]"),
        ({"key": "appealed", "op": "eq", "value": "yes"}, "true or false"),
        (
            {"key": "decision_date", "op": "eq", "value": "16/12/2024"},
            "not an ISO date",
        ),
        (
            {
                "key": "decision_date",
                "op": "between",
                "value": {"kind": "calendar", "month": 3},
            },
            "invalid date",
        ),
    ],
)
def test_a_plan_that_does_not_fit_the_catalog_is_rejected(compiler, flt, message):
    with pytest.raises(PlanError, match=message):
        compiler.compile(_plan(filters=[flt]), KEYS)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"operation": "sum", "sum_key": "issuing_body"}, "not a number"),
        ({"operation": "sum", "sum_key": "nope"}, "unknown or unapproved key"),
        (
            {"operation": "count", "group_by": "decision_date"},
            "cannot group by the date key",
        ),
        ({"operation": "count", "group_by": "nope"}, "unknown or unapproved key"),
    ],
)
def test_sum_and_group_by_are_checked_too(compiler, kwargs, message):
    with pytest.raises(PlanError, match=message):
        compiler.compile(_plan(**kwargs), KEYS)


def test_the_catalog_of_another_doc_type_is_not_usable(compiler):
    other = [
        MetaKey(
            "invoice", "issuing_body", ValueType.TEXT, "d", status=KeyStatus.APPROVED
        )
    ]

    with pytest.raises(PlanError, match="unknown or unapproved key"):
        compiler.compile(
            _plan(filters=[{"key": "issuing_body", "op": "eq", "value": "x"}]), other
        )


def test_a_hand_built_plan_with_a_raw_operator_value_is_still_checked(compiler):
    plan = QueryPlan(
        DT, Operation.COUNT, (Filter("issuing_body", FilterOp.EQ, 123),)
    )  # not a string

    with pytest.raises(PlanError, match="non-empty string"):
        compiler.compile(plan, KEYS)


def test_every_plan_carries_its_documents_as_a_sub_select_for_restricting_retrieval(
    compiler,
):
    plan = _plan(
        operation="lookup",
        filters=[{"key": "issuing_body", "op": "eq", "value": "Kúria"}],
    )

    selection = compiler.compile(plan, KEYS).selection

    assert selection.sql.startswith(
        "SELECT d.id FROM documents d WHERE (d.document_type IS NOT NULL"
    )
    assert (
        "Kúria" not in selection.sql and "Kúria" in selection.params
    )  # bound, never spliced
    assert selection.sql.count("%s") == len(selection.params)
    assert compiler.compile(_plan(), KEYS).selection.params == ("court_decision",)


def test_an_exact_text_match_ignores_a_trailing_sentence_full_stop_on_both_sides(
    compiler,
):
    """A value copied out of a sentence often carries the sentence's own full stop."""
    eq = compiler.compile(
        _plan(
            filters=[{"key": "issuing_body", "op": "eq", "value": "4.P.20.409/2023/4."}]
        ),
        KEYS,
    )
    many = compiler.compile(
        _plan(filters=[{"key": "issuing_body", "op": "in", "value": ["A.", "B;"]}]),
        KEYS,
    )

    assert "rtrim(m.value_text, '.,;:') = %s" in eq.sql
    assert "4.P.20.409/2023/4" in eq.params and "4.P.20.409/2023/4." not in eq.params
    assert [
        "A",
        "B",
    ] in many.params and "rtrim(m.value_text, '.,;:') = ANY(%s)" in many.sql
    assert "'4.P.20.409/2023/4.'" in eq.explanation  # the user sees what was asked


def test_contains_also_ignores_a_trailing_sentence_full_stop(compiler):
    compiled = compiler.compile(
        _plan(filters=[{"key": "issuing_body", "op": "contains", "value": "Kúria."}]),
        KEYS,
    )

    assert "%Kúria%" in compiled.params


class TestDateSet:
    """Several separate periods of one date key are one filter, any of them may hold."""

    def _compile(self, compiler, value):
        return compiler.compile(
            _plan(filters=[{"key": "decision_date", "op": "in", "value": value}]), KEYS
        )

    def test_two_years_are_an_or_of_two_ranges(self, compiler):
        compiled = self._compile(
            compiler,
            [{"kind": "calendar", "year": 2021}, {"kind": "calendar", "year": 2023}],
        )

        assert compiled.sql.count("EXISTS") == 1  # one filter, not two
        assert " OR " in compiled.sql
        assert compiled.params == (
            "court_decision",
            "decision_date",
            date(2021, 1, 1), date(2021, 12, 31),
            date(2023, 1, 1), date(2023, 12, 31),
        )  # fmt: skip
        assert compiled.explanation == (
            "court_decision documents: decision_date in "
            "[2021-01-01..2021-12-31, 2023-01-01..2023-12-31]"
        )
        assert compiled.sql.count("%s") == len(compiled.params)

    def test_a_single_period_in_a_list_is_that_period(self, compiler):
        compiled = self._compile(compiler, [{"kind": "calendar", "year": 2022}])

        assert compiled.params[-2:] == (date(2022, 1, 1), date(2022, 12, 31))

    @pytest.mark.parametrize("value", [[], "2021", {"kind": "calendar", "year": 2021}])
    def test_the_value_must_be_a_non_empty_list(self, compiler, value):
        with pytest.raises(PlanError, match="non-empty list"):
            self._compile(compiler, value)

    def test_a_bad_spec_in_the_list_is_named(self, compiler):
        with pytest.raises(PlanError, match="invalid date"):
            self._compile(compiler, [{"kind": "calendar", "year": 2021}, {"kind": "x"}])

    def test_it_does_not_fit_a_text_key_of_another_kind_by_accident(self, compiler):
        """`in` was always valid for text; a date set must not change that."""
        compiled = compiler.compile(
            _plan(filters=[{"key": "issuing_body", "op": "in", "value": ["A", "B"]}]),
            KEYS,
        )

        assert "= ANY(%s)" in compiled.sql


class TestIdentifierKey:
    """An identifier is compared in its normalised form, by the rule of metadata.identifiers."""

    def _compile(self, compiler, op, value):
        return compiler.compile(
            _plan(filters=[{"key": "document_identifier", "op": op, "value": value}]),
            KEYS,
        )

    def test_eq_binds_the_normalised_wish_its_prefix_form_and_its_length(
        self, compiler
    ):
        compiled = self._compile(compiler, "eq", " 4.P.20.409/2023/4. ")

        assert compiled.params == (
            "court_decision",
            "document_identifier",
            "4.p.20.409/2023/4",  # equal to ...
            "4.p.20.409/2023/4%",  # ... or continued by something
            17,  # (that is not a digit: checked at this length)
        )
        assert "~ '[0-9]'" in compiled.sql
        assert compiled.sql.count("%s") == len(compiled.params)

    def test_the_stored_value_is_normalised_in_the_sql_the_same_way(self, compiler):
        sql = self._compile(compiler, "eq", "A1/2").sql

        for step in ("normalize(m.value_text, NFKC)", "lower(", "'\\s+'", "btrim("):
            assert step in sql

    def test_in_is_an_or_of_the_same_condition(self, compiler):
        compiled = self._compile(compiler, "in", ["A1/2", "B3/4"])

        assert (
            compiled.sql.count(" OR ") >= 3
        )  # one per identifier, plus the two inside
        assert compiled.params[2:5] == ("a1/2", "a1/2%", 4)
        assert compiled.params[5:8] == ("b3/4", "b3/4%", 4)

    def test_contains_is_a_substring_of_the_normalised_form(self, compiler):
        compiled = self._compile(compiler, "contains", "P.20.409")

        assert "LIKE %s" in compiled.sql
        assert compiled.params[-1] == "%p.20.409%"

    def test_like_characters_in_the_wish_are_escaped(self, compiler):
        compiled = self._compile(compiler, "eq", "a_b%c")

        assert compiled.params[3] == "a\\_b\\%c%"

    def test_the_value_is_never_part_of_the_sql(self, compiler):
        evil = "x'; DROP TABLE documents; --"

        compiled = self._compile(compiler, "eq", evil)

        assert "DROP" not in compiled.sql

    @pytest.mark.parametrize("value", ["", "  ", " ./ ", 5, None])
    def test_a_wish_without_an_identifier_in_it_is_refused(self, compiler, value):
        with pytest.raises(PlanError, match="non-empty string"):
            self._compile(compiler, "eq", value)

    @pytest.mark.parametrize("value", [[], "x", ["a", " "]])
    def test_in_needs_a_list_of_identifiers(self, compiler, value):
        with pytest.raises(PlanError):
            self._compile(compiler, "in", value)

    @pytest.mark.parametrize("op", ["ne", "gt", "between"])
    def test_other_operators_do_not_fit_an_identifier(self, compiler, op):
        with pytest.raises(PlanError, match="does not fit identifier key"):
            self._compile(compiler, op, "A1/2")
