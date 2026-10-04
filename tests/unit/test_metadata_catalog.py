"""Tests for metadata.catalog.load_catalog_seed (file validation; no database)."""

import json

import pytest

from metadata.catalog import load_catalog_seed
from models import KeyStatus, ValueType


def _write(tmp_path, data):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


GOOD = {
    "doc_type": "invoice",
    "keys": [
        {"key": "total_amount", "value_type": "number", "description": "Gross total."},
        {
            "key": "currency",
            "value_type": "text",
            "description": "Currency code.",
            "allowed_values": ["HUF", "EUR"],
            "multi_valued": False,
        },
    ],
}


def test_a_valid_catalog_loads_with_approved_keys(tmp_path):
    keys = load_catalog_seed(_write(tmp_path, GOOD))

    assert [(k.doc_type, k.key, k.value_type, k.status) for k in keys] == [
        ("invoice", "total_amount", ValueType.NUMBER, KeyStatus.APPROVED),
        ("invoice", "currency", ValueType.TEXT, KeyStatus.APPROVED),
    ]
    assert keys[1].allowed_values == ("HUF", "EUR")


def test_the_shipped_court_decision_catalog_is_valid():
    keys = load_catalog_seed(
        __import__("pathlib").Path("corpus/data/meta_catalog.json")
    )

    assert {k.key for k in keys} >= {"issuing_body", "decision_date", "document_kind"}
    kind = next(k for k in keys if k.key == "document_kind")
    assert kind.allowed_values and "judgment" in kind.allowed_values


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d["keys"][0].update(key="Összeg"), "English snake_case"),
        (lambda d: d["keys"][0].update(key="Total-Amount"), "English snake_case"),
        (lambda d: d["keys"][1].update(key="total_amount"), "duplicate"),
        (lambda d: d["keys"][0].pop("description"), "description"),
        (lambda d: d["keys"][0].update(value_type="money"), "value_type"),
        (lambda d: d["keys"][0].update(status="maybe"), "status"),
        (lambda d: d["keys"][0].update(allowed_values=["a"]), "allowed_values"),
        (lambda d: d.update(doc_type=""), "doc_type"),
    ],
)
def test_a_malformed_catalog_is_rejected(tmp_path, mutate, message):
    data = json.loads(json.dumps(GOOD))
    mutate(data)

    with pytest.raises(ValueError, match=message):
        load_catalog_seed(_write(tmp_path, data))
