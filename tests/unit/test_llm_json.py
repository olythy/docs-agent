"""Tests for llm_json.extract_json's duplicate-key guard."""

import pytest

from llm_json import DuplicateKeyError, extract_json

# What a model really wrote: two filters fused into one object (no "}, {" between them).
FUSED = """```json
{
  "document_type": "court_decision",
  "operation": "count",
  "filters": [
    {
      "key": "issuing_body",
      "op": "eq",
      "value": "Balassagyarmati Törvényszék"
    ,
    "key": "decision_date",
    "op": "between",
    "value": {"kind": "calendar", "year_offset": -1, "month": 5}
    }
  ]
}
```"""


def test_by_default_the_last_duplicate_wins_as_it_always_did():
    """The tolerant mode other callers (the grader) rely on is unchanged."""
    result = extract_json(FUSED)

    assert result["filters"] == [
        {
            "key": "decision_date",
            "op": "between",
            "value": {"kind": "calendar", "year_offset": -1, "month": 5},
        }
    ]  # the court filter is silently gone: exactly what the strict mode prevents


def test_strict_mode_refuses_an_object_that_repeats_a_key():
    with pytest.raises(DuplicateKeyError, match="the key 'key' twice"):
        extract_json(FUSED, reject_duplicate_keys=True)


def test_strict_mode_accepts_ordinary_json_including_nested_and_repeated_names_in_different_objects():
    text = '{"a": [{"key": 1}, {"key": 2}], "b": {"key": 3}}'

    assert extract_json(text, reject_duplicate_keys=True) == {
        "a": [{"key": 1}, {"key": 2}],
        "b": {"key": 3},
    }


def test_strict_mode_still_applies_the_escape_repair():
    assert extract_json(r'{"a": "x \§ y"}', reject_duplicate_keys=True) == {
        "a": "x \\§ y"
    }
