"""Does each citation of an answer point at a chunk that supports the sentence it follows?

The brief for this evaluation asks that every citation point at exactly the document AND
section that supports the claim, with zero wrong citations. Resolving a label to a real
chunk (``query.answering``) makes a citation *real*; this checks that it is *right*: for
every (sentence, cited chunk) pair a separate model call says whether the chunk states what
the sentence claims. It is evaluation-side, like the other graders: the model that wrote the
answer never grades it.

A citation counts as wrong when the chunk does not support the sentence, or when the label
named no excerpt at all. An answer that is not a refusal and cites nothing is reported as
uncited.

Key exports:
    CitationSupport      -- The result for one answer.
    ClaimCitation        -- One (sentence, cited chunk) pair and its verdict.
    split_claims         -- The (sentence, labels) pairs of an answer's text.
    check_citation_support -- Runs the check with a given judge.
    llm_judge            -- The judge that asks the answer driver.
"""

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from llm_json import extract_json
from models import Citation

#: Where an answer's own text ends: the sources block and anything the service adds after it.
_SOURCES_MARK = "\n\nSources:"

#: A sentence ends at . ! ? followed by space and a capital letter or a bracket label.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÖŐÚÜŰ\[])|\n+")

_LABELS = re.compile(r"\[\s*\d{1,2}(?:\s*[,;]\s*\d{1,2})*\s*\]")

#: How much of a chunk the judge reads.
_MAX_EXCERPT_CHARS = 3000

#: Sentences shorter than this carry no claim worth checking ("Igen.", a lone label).
_MIN_CLAIM_CHARS = 12

Judge = Callable[[Sequence[tuple[str, str]]], list[bool] | None]


@dataclass(frozen=True)
class ClaimCitation:
    """One cited chunk and the sentence that cites it.

    Attributes:
        label: The excerpt label.
        chunk_id: The chunk the label resolves to.
        sentence: The sentence, without its labels.
        supported: The judge's verdict; ``None`` when the judge gave no usable answer.
    """

    label: int
    chunk_id: int
    sentence: str
    supported: bool | None


@dataclass(frozen=True)
class CitationSupport:
    """The citation check of one answer.

    Attributes:
        pairs: Every (sentence, cited chunk) pair judged.
        unknown_labels: Labels that named no excerpt (wrong citations).
        uncited: The answer states something but cites nothing.
    """

    pairs: tuple[ClaimCitation, ...]
    unknown_labels: tuple[int, ...]
    uncited: bool

    @property
    def supported(self) -> int:
        """How many pairs the judge found supported."""
        return sum(1 for p in self.pairs if p.supported is True)

    @property
    def unsupported(self) -> int:
        """How many pairs the judge found NOT supported."""
        return sum(1 for p in self.pairs if p.supported is False)

    @property
    def unjudged(self) -> int:
        """How many pairs got no verdict (the judge's reply was unusable)."""
        return sum(1 for p in self.pairs if p.supported is None)

    @property
    def wrong(self) -> int:
        """Wrong citations: unsupported pairs plus labels naming no excerpt."""
        return self.unsupported + len(self.unknown_labels)

    def to_dict(self) -> dict:
        """A plain dict for the evaluation's result record."""
        return {
            "pairs": len(self.pairs),
            "supported": self.supported,
            "unsupported": self.unsupported,
            "unjudged": self.unjudged,
            "unknown_labels": list(self.unknown_labels),
            "wrong": self.wrong,
            "uncited": self.uncited,
            "unsupported_claims": [
                {"label": p.label, "sentence": p.sentence}
                for p in self.pairs
                if p.supported is False
            ],
        }


def split_claims(text: str) -> list[tuple[str, list[int]]]:
    """The answer's sentences with the labels each cites; sentences without labels too.

    Args:
        text: The answer text; the "Sources:" block and what follows it are not read.

    Returns:
        ``(sentence without its labels, labels)`` in order. Sentences too short to be a
        claim are left out.
    """
    from query.answering import cited_labels

    body = text.split(_SOURCES_MARK, 1)[0]
    claims: list[tuple[str, list[int]]] = []
    for sentence in _SENTENCE_END.split(body):
        labels = cited_labels(sentence)
        plain = re.sub(
            r"\s+([.,;:!?])", r"\1", re.sub(r"\s+", " ", _LABELS.sub("", sentence))
        ).strip()
        if len(plain) >= _MIN_CLAIM_CHARS:
            claims.append((plain, labels))
    return claims


def check_citation_support(
    answer_text: str,
    citations: Sequence[Citation],
    chunk_text: dict[int, str],
    unknown_labels: Sequence[int],
    judge: Judge,
    refused: bool = False,
) -> CitationSupport:
    """Check every (sentence, cited chunk) pair of an answer.

    Args:
        answer_text: The answer.
        citations: The citations the system resolved (label to chunk).
        chunk_text: The text of each chunk by ``chunk_id``.
        unknown_labels: Labels that named no excerpt.
        judge: Takes ``(sentence, excerpt)`` pairs and returns one verdict per pair, or
            ``None`` if it cannot.
        refused: The answer is a refusal: it is expected to cite nothing.
    """
    by_label = {c.label: c for c in citations}
    wanted: list[tuple[int, int, str]] = []  # (label, chunk_id, sentence)
    for sentence, labels in split_claims(answer_text):
        for label in labels:
            citation = by_label.get(label)
            if citation is not None and citation.chunk_id in chunk_text:
                wanted.append((label, citation.chunk_id, sentence))
    verdicts: list[bool] | None = None
    if wanted:
        verdicts = judge(
            [(sentence, chunk_text[chunk_id]) for _, chunk_id, sentence in wanted]
        )
        if verdicts is not None and len(verdicts) != len(wanted):
            verdicts = None
    pairs = tuple(
        ClaimCitation(
            label, chunk_id, sentence, None if verdicts is None else verdicts[i]
        )
        for i, (label, chunk_id, sentence) in enumerate(wanted)
    )
    uncited = not refused and not citations and not unknown_labels
    return CitationSupport(pairs, tuple(unknown_labels), uncited)


def llm_judge(pairs: Sequence[tuple[str, str]]) -> list[bool] | None:
    """Ask the answer driver (a different call from the one that wrote the answer).

    Args:
        pairs: ``(sentence, excerpt)`` pairs.

    Returns:
        One verdict per pair, or ``None`` if the reply could not be read.
    """
    from drivers.llm import get_answer_driver

    items = [
        {"id": i, "claim": sentence, "excerpt": excerpt[:_MAX_EXCERPT_CHARS]}
        for i, (sentence, excerpt) in enumerate(pairs, start=1)
    ]
    prompt = (
        "You check the citations of a question-answering system. For each item, decide "
        "whether the EXCERPT itself states what the CLAIM says. 'supported' is true only "
        "if the excerpt contains the fact(s) of the claim; it is false if the excerpt is "
        "only about the same case or topic, or does not contain the fact, or contradicts "
        "it. Judge each item on its own excerpt only.\n\n"
        f"{json.dumps(items, ensure_ascii=False)}\n\n"
        'Output ONLY this JSON, one entry per item in the same order: {"supported": '
        "[true, false, ...]}"
    )
    driver = get_answer_driver()
    reply = driver.run_tool_calling_turn(
        [{"role": "user", "content": prompt}], temperature=0.0
    ).content
    try:
        data = extract_json(reply or "")
    except (ValueError, json.JSONDecodeError):
        return None
    verdicts = data.get("supported") if isinstance(data, dict) else None
    if not isinstance(verdicts, list) or not all(isinstance(v, bool) for v in verdicts):
        return None
    return verdicts
