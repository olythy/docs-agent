"""Deciding what kind of document a document is.

A :class:`TypeClassifier` reads the opening of one document and the list of known
document types (each a name and a description) and either picks one or proposes a
new one. :class:`LLMTypeClassifier` asks a language model.

The classifier only *decides*; it does not write anything and it does not trust
itself: its answer carries a verbatim quote from the document, which the runner
verifies, and a proposed type is only ever stored as ``proposed`` (unusable in
queries until a person approves it).

Key exports:
    TypeProposal      -- A type the classifier suggests.
    Classification    -- What the classifier decided.
    TypeClassifier    -- The contract.
    LLMTypeClassifier -- The model-backed classifier.
"""

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from drivers.llm import AnswerDriver
from llm_json import extract_json
from models import DocumentType


@dataclass(frozen=True)
class TypeProposal:
    """A new document type the classifier suggests (English snake_case ``type``)."""

    type: str
    name: str
    description: str


@dataclass(frozen=True)
class Classification:
    """What the classifier decided about one document.

    Exactly one of ``type`` and ``proposal`` is set unless ``failed``.

    Attributes:
        type: The name of an existing type the document belongs to.
        proposal: A new type, when none of the existing ones fits.
        evidence: A verbatim quote from the document that shows its type.
        failed: The answer was unusable (not JSON, or neither/both of the above),
            so the document stays unclassified and is retried on the next run.
    """

    type: str | None = None
    proposal: TypeProposal | None = None
    evidence: str | None = None
    failed: bool = False


class TypeClassifier(ABC):
    """Decides a document's type from its opening text."""

    @abstractmethod
    def classify(self, text: str, types: Sequence[DocumentType]) -> Classification:
        """Classify one document.

        Args:
            text: The opening of the document (a summary and its first passages).
            types: The types that may be chosen (approved ones and already
                proposed ones; never retired ones).
        """


class LLMTypeClassifier(TypeClassifier):
    """Asks a language model to pick or propose a type, with a quote as evidence.

    Args:
        llm: The driver used for the call.
    """

    def __init__(self, llm: AnswerDriver) -> None:
        self._llm = llm

    def classify(self, text: str, types: Sequence[DocumentType]) -> Classification:
        reply = (
            self._llm.run_tool_calling_turn(
                [{"role": "user", "content": self._prompt(text, types)}]
            ).content
            or ""
        )
        try:
            data = extract_json(reply, reject_duplicate_keys=True)
        except (ValueError, json.JSONDecodeError):
            return Classification(failed=True)
        chosen, raw = data.get("type"), data.get("proposal")
        evidence = data.get("evidence")
        if not isinstance(evidence, str):
            evidence = None
        if (chosen is None) == (raw is None):  # neither, or both
            return Classification(failed=True)
        if chosen is not None:
            if not isinstance(chosen, str):
                return Classification(failed=True)
            return Classification(type=chosen, evidence=evidence)
        proposal = self._proposal(raw)
        if proposal is None:
            return Classification(failed=True)
        return Classification(proposal=proposal, evidence=evidence)

    @staticmethod
    def _proposal(raw: object) -> TypeProposal | None:
        if not isinstance(raw, dict):
            return None
        fields = [raw.get(name) for name in ("type", "name", "description")]
        if not all(isinstance(f, str) and f.strip() for f in fields):
            return None
        return TypeProposal(*fields)  # type: ignore[arg-type]

    @staticmethod
    def _prompt(text: str, types: Sequence[DocumentType]) -> str:
        known = "\n".join(
            f"- {t.type} ({t.status.value}): {t.name}. {t.description}" for t in types
        )
        return (
            "Decide what KIND of document the text below is.\n\n"
            "KNOWN TYPES:\n"
            f"{known or '(none yet)'}\n\n"
            "RULES:\n"
            '- If one known type fits, answer with its exact name in "type" and set '
            '"proposal" to null. A type marked (proposed) may be chosen too.\n'
            '- Only if NONE fits, set "type" to null and describe a new kind in '
            '"proposal": an English snake_case "type", a short "name", and a '
            '"description" of the KIND of document in general (not of this one).\n'
            '- "evidence" must be a short VERBATIM quote (max 200 characters) copied '
            "exactly from the text that shows the type, such as the heading.\n"
            "- Never guess: base the decision only on the text.\n\n"
            'Output ONLY JSON: {"type": "..." or null, "proposal": {"type": "...", '
            '"name": "...", "description": "..."} or null, "evidence": "..."}\n\n'
            f"TEXT:\n{text}"
        )
