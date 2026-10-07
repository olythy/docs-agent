"""Turning a candidate's text into the typed value the database stores.

Key exports:
    to_meta_value -- Build a MetaValue from a Candidate, or None if the text
                     is not a valid value of the key's type.
"""

from datetime import date
from decimal import Decimal, InvalidOperation

from metadata.sources import Candidate
from models import MetaKey, MetaSource, MetaValue, ValueType


def to_meta_value(
    key: MetaKey,
    candidate: Candidate,
    content_hash: str,
    source: MetaSource,
    ordinal: int = 0,
    page: int | None = None,
) -> MetaValue | None:
    """Convert ``candidate`` to a typed :class:`models.MetaValue`.

    Args:
        key: The catalog key (its type decides the column).
        candidate: The proposed value.
        content_hash: The document.
        source: Where the value came from.
        ordinal: Position among a multi-valued key's rows.
        page: Page/section number of the evidence chunk, if known.

    Returns:
        The typed value, or ``None`` when the text cannot be read as the key's
        type (an unparseable date, a non-number): such a value is not stored.
    """
    text = candidate.value.strip()

    def build(**value: object) -> MetaValue:
        return MetaValue(
            content_hash=content_hash,
            key=key.key,
            key_version=key.version,
            source=source,
            unit=candidate.unit,
            ordinal=ordinal,
            evidence=candidate.evidence,
            evidence_chunk_index=candidate.evidence_chunk_index,
            page=page,
            **value,  # type: ignore[arg-type]  # exactly one typed value column, set below
        )

    try:
        if key.value_type in (ValueType.TEXT, ValueType.IDENTIFIER):
            return build(value_text=text)
        if key.value_type is ValueType.NUMBER:
            return build(value_number=Decimal(text))
        if key.value_type is ValueType.DATE:
            return build(value_date=date.fromisoformat(text))
        if text.lower() in {"true", "yes"}:
            return build(value_bool=True)
        if text.lower() in {"false", "no"}:
            return build(value_bool=False)
    except (ValueError, InvalidOperation):
        return None
    return None
    return None
