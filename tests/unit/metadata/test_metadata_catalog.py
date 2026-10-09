"""Tests for metadata.catalog.load_catalog_seed (file validation; no database)."""

import json
from pathlib import Path

import pytest

from metadata.catalog import load_catalog_seed
from models import KeyStatus, TypeStatus, ValueType


def _write(tmp_path, data):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


GOOD = {
    "types": [
        {
            "type": "invoice",
            "name": "Invoice",
            "description": "A bill for goods or services, with a total and a due date.",
            "keys": [
                {
                    "key": "total_amount",
                    "value_type": "number",
                    "description": "Gross total.",
                },
                {
                    "key": "currency",
                    "value_type": "text",
                    "description": "Currency code.",
                    "allowed_values": ["HUF", "EUR"],
                    "multi_valued": False,
                },
            ],
        },
        {
            "type": "contract",
            "name": "Contract",
            "description": "An agreement between parties.",
            "keys": [
                # the same key name under another type is fine: keys belong to a type
                {"key": "total_amount", "value_type": "number", "description": "Value."}
            ],
        },
    ]
}


def test_a_valid_catalog_loads_its_types_and_keys_as_approved(tmp_path):
    catalog = load_catalog_seed(_write(tmp_path, GOOD))

    assert [(t.type, t.name, t.status) for t in catalog.types] == [
        ("invoice", "Invoice", TypeStatus.APPROVED),
        ("contract", "Contract", TypeStatus.APPROVED),
    ]
    assert catalog.types[0].description.startswith("A bill for goods")
    assert [(k.doc_type, k.key, k.value_type, k.status) for k in catalog.keys] == [
        ("invoice", "total_amount", ValueType.NUMBER, KeyStatus.APPROVED),
        ("invoice", "currency", ValueType.TEXT, KeyStatus.APPROVED),
        ("contract", "total_amount", ValueType.NUMBER, KeyStatus.APPROVED),
    ]
    assert catalog.keys[1].allowed_values == ("HUF", "EUR")


def test_the_shipped_court_decision_catalog_is_valid():
    catalog = load_catalog_seed(Path("corpus/data/meta_catalog.json"))

    assert [t.type for t in catalog.types] == ["court_decision"]
    assert catalog.types[0].description  # what the classifier and planner read
    assert {k.key for k in catalog.keys} >= {
        "issuing_body",
        "decision_date",
        "document_kind",
        "document_identifier",
    }
    kind = next(k for k in catalog.keys if k.key == "document_kind")
    assert kind.allowed_values and "judgment" in kind.allowed_values


def test_the_old_single_type_format_is_refused_with_how_to_convert_it(tmp_path):
    old = {"doc_type": "invoice", "keys": GOOD["types"][0]["keys"]}

    with pytest.raises(ValueError, match=r"old single-type format; wrap it as"):
        load_catalog_seed(_write(tmp_path, old))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d["types"][0]["keys"][0].update(key="Összeg"), "English snake_case"),
        (
            lambda d: d["types"][0]["keys"][0].update(key="Total-Amount"),
            "English snake_case",
        ),
        (
            lambda d: d["types"][0]["keys"][1].update(key="total_amount"),
            "duplicate key",
        ),
        (lambda d: d["types"][0]["keys"][0].pop("description"), "description"),
        (lambda d: d["types"][0]["keys"][0].update(value_type="money"), "value_type"),
        (lambda d: d["types"][0]["keys"][0].update(status="maybe"), "status"),
        (
            lambda d: d["types"][0]["keys"][0].update(allowed_values=["a"]),
            "allowed_values",
        ),
        (lambda d: d["types"][0].update(type="Számla"), "English snake_case"),
        (lambda d: d["types"][1].update(type="invoice"), "duplicate type"),
        (lambda d: d["types"][0].pop("name"), "needs a name"),
        (lambda d: d["types"][0].update(description="  "), "needs a description"),
        (lambda d: d["types"][0].update(keys="nope"), "'keys' must be a list"),
        (lambda d: d.update(types=[]), "non-empty list"),
        (lambda d: d.update(types=["invoice"]), "must be an object"),
    ],
)
def test_a_malformed_catalog_is_rejected(tmp_path, mutate, message):
    data = json.loads(json.dumps(GOOD))
    mutate(data)

    with pytest.raises(ValueError, match=message):
        load_catalog_seed(_write(tmp_path, data))


def test_a_catalog_that_is_not_an_object_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="must be a JSON object"):
        load_catalog_seed(_write(tmp_path, ["invoice"]))
