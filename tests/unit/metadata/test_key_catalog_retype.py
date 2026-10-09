"""Retyping a key between text and identifier keeps its values valid (no database).

A key's version is how the system knows which values were extracted under which
definition; a changed definition makes them stale and the documents are extracted again.
``text`` and ``identifier`` store and extract in the same way, so changing between them
alone must not do that, or a new type would cost a full re-extraction.
"""

from dataclasses import replace

from metadata.catalog import Catalog, KeyCatalog
from models import DocumentType, KeyStatus, MetaKey, TypeStatus, ValueType

DT = "court_decision"


class FakeStore:
    def __init__(self, keys):
        self.keys = {(k.doc_type, k.key): k for k in keys}

    def ensure_type(self, doc_type):
        return None

    def list_keys(self, doc_type, status=None):
        return [k for (t, _), k in self.keys.items() if t == doc_type]

    def upsert_key(self, key):
        self.keys[(key.doc_type, key.key)] = key

    def get_type(self, doc_type):
        return None

    def upsert_type(self, doc_type):
        return None


def _key(vtype=ValueType.TEXT, **kw):
    kw.setdefault("description", "Identifier(s) of this document.")
    return MetaKey(
        DT,
        "document_identifier",
        vtype,
        status=KeyStatus.APPROVED,
        multi_valued=True,
        version=3,
        **kw,
    )


def _import(store, key):
    doc_type = DocumentType(DT, "Court decision", "d", TypeStatus.APPROVED)
    return KeyCatalog(store).import_catalog(Catalog([doc_type], [key]))  # type: ignore[arg-type]


def test_text_to_identifier_changes_the_type_and_keeps_the_version():
    store = FakeStore([_key(ValueType.TEXT)])

    result = _import(store, replace(_key(ValueType.IDENTIFIER), version=1))

    stored = store.keys[(DT, "document_identifier")]
    assert stored.value_type is ValueType.IDENTIFIER
    assert stored.version == 3  # the values extracted under it are still valid
    assert (result.retyped, result.revised, result.unchanged) == (1, 0, 0)


def test_identifier_back_to_text_is_the_same():
    store = FakeStore([_key(ValueType.IDENTIFIER)])

    result = _import(store, _key(ValueType.TEXT))

    assert store.keys[(DT, "document_identifier")].version == 3
    assert result.retyped == 1


def test_a_retype_together_with_a_new_description_is_a_real_revision():
    store = FakeStore([_key(ValueType.TEXT)])

    result = _import(
        store, _key(ValueType.IDENTIFIER, description="Now every number it states.")
    )

    assert store.keys[(DT, "document_identifier")].version == 4  # stale: extract again
    assert (result.revised, result.retyped) == (1, 0)


def test_a_change_to_another_kind_of_type_is_a_real_revision():
    store = FakeStore([_key(ValueType.IDENTIFIER)])

    result = _import(store, _key(ValueType.NUMBER))

    assert store.keys[(DT, "document_identifier")].version == 4
    assert (result.revised, result.retyped) == (1, 0)


def test_importing_the_same_identifier_key_again_changes_nothing():
    store = FakeStore([_key(ValueType.IDENTIFIER)])

    result = _import(store, _key(ValueType.IDENTIFIER))

    assert (result.unchanged, result.retyped, result.revised) == (1, 0, 0)
