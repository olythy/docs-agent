"""Where candidate metadata values come from.

A :class:`MetaSource` turns one document's text into candidate values for some
catalog keys. Sources are interchangeable behind that contract, so a corpus
picks the ones that fit it: the general :class:`LLMMetaSource`, and an adapter
such as :class:`ChunkMetadataSource` that takes a fact an earlier, deterministic
ingestion step already extracted. The runner treats them identically apart from
one rule: values from the LLM must be verified against their quote, values from a
deterministic source are trusted as extracted.

Key exports:
    Candidate          -- One proposed value, before verification and typing.
    ProposedKey        -- A key the extractor suggests adding to the catalog.
    SourceResult       -- What a source returned for one document.
    MetaSource         -- The contract.
    LLMMetaSource      -- The general extractor (one LLM call per document).
    ChunkMetadataSource-- Copies a field already present in the chunk metadata.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from drivers.llm import AnswerDriver
from llm_json import extract_json
from models import MetaKey, RetrievedChunk
from models import MetaSource as MetaSourceKind


@dataclass(frozen=True)
class Candidate:
    """A value a source proposes for a key, as text, before it is verified and typed.

    Attributes:
        key: The catalog key.
        value: The value as text (a date as ISO ``YYYY-MM-DD``, a number plain).
        unit: A unit or currency, if any.
        evidence: A verbatim quote showing the value (required of the LLM source).
        evidence_chunk_index: Which chunk the quote is in, when the source knows.
    """

    key: str
    value: str
    unit: str | None = None
    evidence: str | None = None
    evidence_chunk_index: int | None = None


@dataclass(frozen=True)
class ProposedKey:
    """A new key an extractor suggests; kept ``proposed`` until a person approves it."""

    key: str
    description: str
    value_type: str


@dataclass(frozen=True)
class SourceResult:
    """What a source produced for one document.

    Attributes:
        candidates: The proposed values (a key may be absent: nothing was found).
        proposed_keys: New keys the source thinks the catalog is missing.
        failed: The source could not produce a usable answer (e.g. the reply was
            not JSON), so a missing key means "unknown", not "absent".
    """

    candidates: list[Candidate] = field(default_factory=list)
    proposed_keys: list[ProposedKey] = field(default_factory=list)
    failed: bool = False


class MetaSource(ABC):
    """Produces candidate values for some catalog keys from one document."""

    #: How values from this source are labelled when stored.
    kind: MetaSourceKind

    #: Whether the runner must verify each candidate against its quote.
    needs_verification: bool

    @abstractmethod
    def supports(self, key: MetaKey) -> bool:
        """Whether this source can produce values for ``key``."""

    @abstractmethod
    def extract(
        self, chunks: list[RetrievedChunk], keys: list[MetaKey]
    ) -> SourceResult:
        """Return candidates for ``keys`` (all of which this source supports).

        Args:
            chunks: The text to read; ``content`` is the body, without prefixes.
            keys: The keys to extract.
        """


class ChunkMetadataSource(MetaSource):
    """Copies a field that ingestion already extracted into the chunks' metadata.

    An adapter for the corpora where a deterministic extractor exists (this
    project's regex date extractor writes ``document_date`` onto every chunk).
    The value is trusted as extracted, so no quote is required.

    Args:
        mapping: Catalog key -> chunk-metadata field it is copied from.
    """

    kind = MetaSourceKind.DETERMINISTIC
    needs_verification = False

    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = mapping

    def supports(self, key: MetaKey) -> bool:
        return key.key in self._mapping

    def extract(
        self, chunks: list[RetrievedChunk], keys: list[MetaKey]
    ) -> SourceResult:
        candidates = []
        for key in keys:
            field_name = self._mapping[key.key]
            for chunk in chunks:
                value = getattr(chunk.metadata, field_name, None)
                if value:
                    candidates.append(Candidate(key.key, str(value)))
                    break
        return SourceResult(candidates=candidates)


class LLMMetaSource(MetaSource):
    """The general extractor: one LLM call per document, with a quote for every value.

    The model is told to reuse a catalog key whenever its description fits, to
    choose categorical values from the key's allowed list, to quote the
    document verbatim as evidence, and to omit a key the text does not state.

    Args:
        llm: The answer driver used for the call.
    """

    kind = MetaSourceKind.LLM
    needs_verification = True

    def __init__(self, llm: AnswerDriver) -> None:
        self._llm = llm

    def supports(self, key: MetaKey) -> bool:
        return True

    def extract(
        self, chunks: list[RetrievedChunk], keys: list[MetaKey]
    ) -> SourceResult:
        reply = (
            self._llm.run_tool_calling_turn(
                [{"role": "user", "content": self._prompt(chunks, keys)}]
            ).content
            or ""
        )
        try:
            data = extract_json(reply)
        except (ValueError, json.JSONDecodeError):
            return SourceResult(failed=True)
        wanted = {k.key for k in keys}
        candidates = []
        for row in data.get("values", []):
            if not isinstance(row, dict) or row.get("key") not in wanted:
                continue
            if row.get("value") in (None, ""):
                continue
            candidates.append(
                Candidate(
                    key=row["key"],
                    value=str(row["value"]),
                    unit=row.get("unit"),
                    evidence=row.get("evidence"),
                    evidence_chunk_index=self._chunk_of(chunks, row.get("evidence")),
                )
            )
        proposed = [
            ProposedKey(p["key"], p.get("description", ""), p.get("value_type", "text"))
            for p in data.get("proposed_keys", [])
            if isinstance(p, dict) and p.get("key")
        ]
        return SourceResult(candidates=candidates, proposed_keys=proposed)

    @staticmethod
    def _chunk_of(chunks: list[RetrievedChunk], evidence: object) -> int | None:
        """Find which chunk contains the quote (whitespace-insensitive), if any."""
        if not isinstance(evidence, str) or not evidence.strip():
            return None
        needle = " ".join(evidence.split()).lower()
        for chunk in chunks:
            if needle in " ".join(chunk.content.split()).lower():
                return chunk.metadata.chunk_index
        return None

    @staticmethod
    def _prompt(chunks: list[RetrievedChunk], keys: list[MetaKey]) -> str:
        described = []
        for k in keys:
            line = f"- {k.key} ({k.value_type.value}): {k.description}"
            if k.allowed_values:
                line += f" Allowed values (choose exactly one): {', '.join(k.allowed_values)}."
            if k.example:
                line += f" Example: {k.example}"
            described.append(line)
        text = "\n\n".join(
            f"[chunk {c.metadata.chunk_index}]\n{c.content}" for c in chunks
        )
        return (
            "Extract metadata from the document excerpts below.\n\n"
            "KEYS (use these exact English key names):\n"
            + "\n".join(described)
            + "\n\n"
            "RULES:\n"
            "- Use an existing key whenever its description fits. Only if NONE fits may you add a new English "
            "snake_case key under 'proposed_keys'.\n"
            "- Values keep the document's own language, except: dates as ISO YYYY-MM-DD, numbers as plain numbers "
            "without separators (put the currency in 'unit'), and keys with allowed values take one of those tokens.\n"
            "- Every value MUST have 'evidence': a short VERBATIM quote (max 200 characters) copied exactly from the "
            "excerpts, in the document's own wording, that shows the value.\n"
            "- Omit a key the excerpts do not state. Never guess.\n\n"
            'Output ONLY JSON: {"values":[{"key":"...","value":"...","unit":null,"evidence":"..."}],'
            '"proposed_keys":[{"key":"...","description":"...","value_type":"text|number|date"}]}\n\n'
            f"EXCERPTS:\n{text}"
        )
