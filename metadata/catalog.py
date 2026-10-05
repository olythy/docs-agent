"""Validating, loading and versioning the key catalog.

The catalog is *data*: what keys exist for a document type, their type,
description and allowed values. A corpus ships its catalog as a JSON file; this
module validates it and writes it to ``meta_keys``, bumping a key's ``version``
when its description changes so values extracted under the old definition can be
found and refreshed.

Key exports:
    load_catalog_seed -- Parse and validate a catalog file into MetaKey objects.
    KeyCatalog        -- Imports a seed into the database with version tracking.
    ImportResult      -- What an import changed.
"""

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from document_store import DocumentStore
from models import KeyStatus, MetaKey, ValueType

_KEY_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def load_catalog_seed(path: Path) -> list[MetaKey]:
    """Read and validate a catalog file.

    The file is ``{"doc_type": "...", "keys": [{"key", "value_type",
    "description", "example"?, "allowed_values"?, "multi_valued"?, "status"?}]}``.
    Keys default to ``approved`` here: a seed file is curated by a person, unlike a
    key an extractor proposes.

    Args:
        path: The JSON file.

    Returns:
        The keys, in file order.

    Raises:
        ValueError: On a malformed file, a non-English-snake_case key name, an
            unknown type or status, a duplicate key, or allowed values on a key
            that is not text.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    doc_type = data.get("doc_type")
    if not isinstance(doc_type, str) or not doc_type:
        raise ValueError(f"{path}: 'doc_type' must be a non-empty string")
    keys: list[MetaKey] = []
    seen: set[str] = set()
    for raw in data.get("keys", []):
        name = raw.get("key", "")
        if not _KEY_NAME.match(name):
            raise ValueError(f"{path}: key {name!r} must be English snake_case")
        if name in seen:
            raise ValueError(f"{path}: duplicate key {name!r}")
        seen.add(name)
        if not raw.get("description"):
            raise ValueError(f"{path}: key {name!r} needs a description")
        value_type = _enum(ValueType, raw.get("value_type"), "value_type", name, path)
        status = _enum(KeyStatus, raw.get("status", "approved"), "status", name, path)
        allowed = raw.get("allowed_values")
        if allowed is not None and value_type is not ValueType.TEXT:
            raise ValueError(f"{path}: key {name!r}: allowed_values need a text key")
        keys.append(
            MetaKey(
                doc_type=doc_type,
                key=name,
                value_type=value_type,
                description=raw["description"],
                example=raw.get("example"),
                allowed_values=None if allowed is None else tuple(allowed),
                multi_valued=bool(raw.get("multi_valued", False)),
                status=status,
            )
        )
    return keys


def _enum[E: StrEnum](
    enum: type[E], raw: object, field: str, key: str, path: Path
) -> E:
    """Convert ``raw`` to ``enum``, naming the field, key and file when it is not valid."""
    try:
        return enum(raw)
    except ValueError:
        allowed = ", ".join(member.value for member in enum)
        raise ValueError(
            f"{path}: key {key!r}: unknown {field} {raw!r} (allowed: {allowed})"
        ) from None


@dataclass(frozen=True)
class ImportResult:
    """What importing a catalog seed changed.

    Attributes:
        added: Keys that did not exist.
        revised: Keys whose description or definition changed (version bumped).
        unchanged: Keys that already matched.
    """

    added: int
    revised: int
    unchanged: int


class KeyCatalog:
    """Imports a catalog seed, keeping each key's version in step with its meaning.

    Args:
        store: Where the catalog lives.
    """

    def __init__(self, store: DocumentStore) -> None:
        self._store = store

    def import_seed(self, keys: list[MetaKey]) -> ImportResult:
        """Write ``keys`` to the catalog.

        A new key starts at version 1. An existing key whose definition (type,
        description, allowed values, multiplicity) changed gets ``version + 1``,
        so values produced under the old definition are recognisably stale. A key
        that already matches is left alone, which makes re-importing a no-op.
        """
        added = revised = unchanged = 0
        for doc_type in dict.fromkeys(key.doc_type for key in keys):
            self._store.ensure_type(doc_type)
        for key in keys:
            existing = {k.key: k for k in self._store.list_keys(key.doc_type)}.get(
                key.key
            )
            if existing is None:
                self._store.upsert_key(key)
                added += 1
            elif _same_definition(existing, key):
                unchanged += 1
            else:
                self._store.upsert_key(_with_version(key, existing.version + 1))
                revised += 1
        return ImportResult(added, revised, unchanged)


def _same_definition(a: MetaKey, b: MetaKey) -> bool:
    return (
        a.value_type == b.value_type
        and a.description == b.description
        and a.example == b.example
        and a.allowed_values == b.allowed_values
        and a.multi_valued == b.multi_valued
        and a.status == b.status
    )


def _with_version(key: MetaKey, version: int) -> MetaKey:
    return MetaKey(
        doc_type=key.doc_type,
        key=key.key,
        value_type=key.value_type,
        description=key.description,
        example=key.example,
        allowed_values=key.allowed_values,
        multi_valued=key.multi_valued,
        status=key.status,
        version=version,
    )
