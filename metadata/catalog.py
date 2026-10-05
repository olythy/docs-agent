"""Validating, loading and versioning the key catalog.

The catalog is *data*: which document types exist (a name and a description the
classifier and the planner read) and, for each, what keys exist, their type,
description and allowed values. A deployment ships its catalog as a JSON file
that may declare many types; this module validates it and writes it to
``document_types`` and ``meta_keys``, bumping a key's ``version`` when its
description changes so values extracted under the old definition can be found
and refreshed. A type declared in the file is ``approved``: the file is the
reviewed route to approval, as the CLI is for a type or key an LLM proposed.

Key exports:
    Catalog           -- The types and keys a catalog file declares.
    CatalogError      -- A malformed or inconsistent catalog file.
    load_catalog_seed -- Parse and validate a catalog file.
    KeyCatalog        -- Imports a catalog into the database with version tracking.
    ImportResult      -- What an import changed.
"""

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from document_store import DocumentStore
from models import DocumentType, KeyStatus, MetaKey, TypeStatus, ValueType

_KEY_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class CatalogError(ValueError):
    """A catalog file that is malformed or inconsistent."""


@dataclass(frozen=True)
class Catalog:
    """What a catalog file declares.

    Attributes:
        types: The document types, all ``approved`` (declared by a person).
        keys: Every type's keys, in file order, each carrying its ``doc_type``.
    """

    types: list[DocumentType]
    keys: list[MetaKey]


def load_catalog_seed(path: Path) -> Catalog:
    """Read and validate a catalog file.

    The file is ``{"types": [{"type", "name", "description", "keys": [{"key",
    "value_type", "description", "example"?, "allowed_values"?, "multi_valued"?,
    "status"?}]}]}``. Types and keys default to ``approved`` here: a catalog file
    is curated by a person, unlike a type or key an extractor proposes.

    Args:
        path: The JSON file.

    Returns:
        The types and keys, in file order.

    Raises:
        CatalogError: (a ``ValueError``) On a malformed file; the old single-type format (a top-level
            ``doc_type``); a type or key name that is not English snake_case; a
            missing name or description; a duplicate type or key; an unknown value
            type or status; or allowed values on a key that is not text.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise CatalogError(f"{path}: a catalog must be a JSON object")
    if "doc_type" in data:
        raise CatalogError(
            f"{path}: this is the old single-type format; wrap it as "
            '{"types": [{"type": <doc_type>, "name": ..., "description": ..., '
            '"keys": [...]}]}'
        )
    raw_types = data.get("types")
    if not isinstance(raw_types, list) or not raw_types:
        raise CatalogError(f"{path}: 'types' must be a non-empty list")
    types: list[DocumentType] = []
    keys: list[MetaKey] = []
    seen_types: set[str] = set()
    for raw_type in raw_types:
        doc_type = _document_type(raw_type, seen_types, path)
        types.append(doc_type)
        keys += _keys_of(doc_type.type, raw_type.get("keys", []), path)
    return Catalog(types=types, keys=keys)


def _document_type(raw: object, seen: set[str], path: Path) -> DocumentType:
    """Validate one type entry of the catalog file."""
    if not isinstance(raw, dict):
        raise CatalogError(f"{path}: every entry of 'types' must be an object")
    name = raw.get("type", "")
    if not isinstance(name, str) or not _KEY_NAME.match(name):
        raise CatalogError(f"{path}: type {name!r} must be English snake_case")
    if name in seen:
        raise CatalogError(f"{path}: duplicate type {name!r}")
    seen.add(name)
    for field in ("name", "description"):
        if not isinstance(raw.get(field), str) or not raw[field].strip():
            raise CatalogError(f"{path}: type {name!r} needs a {field}")
    if not isinstance(raw.get("keys", []), list):
        raise CatalogError(f"{path}: type {name!r}: 'keys' must be a list")
    return DocumentType(name, raw["name"], raw["description"], TypeStatus.APPROVED)


def _keys_of(doc_type: str, raw_keys: list, path: Path) -> list[MetaKey]:
    """Validate the keys of one type."""
    keys: list[MetaKey] = []
    seen: set[str] = set()
    for raw in raw_keys:
        name = raw.get("key", "")
        if not _KEY_NAME.match(name):
            raise CatalogError(f"{path}: key {name!r} must be English snake_case")
        if name in seen:
            raise CatalogError(f"{path}: duplicate key {name!r} in type {doc_type!r}")
        seen.add(name)
        if not raw.get("description"):
            raise CatalogError(f"{path}: key {name!r} needs a description")
        value_type = _enum(ValueType, raw.get("value_type"), "value_type", name, path)
        status = _enum(KeyStatus, raw.get("status", "approved"), "status", name, path)
        allowed = raw.get("allowed_values")
        if allowed is not None and value_type is not ValueType.TEXT:
            raise CatalogError(f"{path}: key {name!r}: allowed_values need a text key")
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
        raise CatalogError(
            f"{path}: key {key!r}: unknown {field} {raw!r} (allowed: {allowed})"
        ) from None


@dataclass(frozen=True)
class ImportResult:
    """What importing a catalog changed.

    Attributes:
        added: Keys that did not exist.
        revised: Keys whose description or definition changed (version bumped).
        unchanged: Keys that already matched.
        types_added: Document types that did not exist.
        types_updated: Types whose name, description or status changed.
    """

    added: int
    revised: int
    unchanged: int
    types_added: int = 0
    types_updated: int = 0


class KeyCatalog:
    """Imports a catalog, keeping each key's version in step with its meaning.

    Args:
        store: Where the catalog lives.
    """

    def __init__(self, store: DocumentStore) -> None:
        self._store = store

    def import_catalog(self, catalog: Catalog) -> ImportResult:
        """Write a whole catalog: its types first, then its keys.

        A declared type is written as it is in the file (name, description,
        ``approved``); the file is the source of truth for the types it names.

        Returns:
            What changed, for types and for keys.
        """
        types_added = types_updated = 0
        for doc_type in catalog.types:
            existing = self._store.get_type(doc_type.type)
            if existing is None:
                types_added += 1
            elif existing != doc_type:
                types_updated += 1
            if existing != doc_type:
                self._store.upsert_type(doc_type)
        keys = self.import_seed(catalog.keys)
        return ImportResult(
            keys.added, keys.revised, keys.unchanged, types_added, types_updated
        )

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
