"""The sentences of an answer and the excerpt labels each one cites.

An answer cites with labels in square brackets (``[2]``, ``[1][3]``). The citation checks
work sentence by sentence: this module cuts an answer into sentences, reads the labels of a
sentence, and takes labels out of it. It is shared by the answerer (which verifies the
citations it will show) and by the evaluation (which checks them from outside), so both cut
an answer the same way.

Key exports:
    cited_labels   -- The labels a text cites, each once, in order.
    strip_labels   -- A text without some labels.
    sentence_spans -- Where the sentences of a text are.
    plain_sentence -- A sentence as a claim: no labels, tidy spacing.
    split_claims   -- (claim, labels) for every sentence long enough to be a claim.
"""

import re
from collections.abc import Collection

#: A bracketed list of one- or two-digit excerpt labels: ``[2]``, ``[1, 3]``. Not a year
#: (``[2023]``) or a footnote of any length: the labels are the numbers 1..top_k.
LABELS = re.compile(r"\[\s*(\d{1,2}(?:\s*[,;]\s*\d{1,2})*)\s*\]")

#: A sentence ends at . ! ? followed by space and a capital letter or a bracket label, or at
#: a line break.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÖŐÚÜŰ\[])|\n+")

#: Sentences shorter than this carry no claim worth checking ("Igen.", a lone label).
MIN_CLAIM_CHARS = 12


def cited_labels(text: str) -> list[int]:
    """The excerpt labels a text cites, each once, in the order first used."""
    found: list[int] = []
    for group in LABELS.findall(text):
        for number in re.split(r"\s*[,;]\s*", group):
            label = int(number)
            if label not in found:
                found.append(label)
    return found


def strip_labels(text: str, labels: Collection[int]) -> str:
    """``text`` without the given labels.

    Adjacent brackets (``[1][2]``) are one run with the space before it: a bracket that loses
    all its labels goes, one that keeps some is rewritten without the others (``[1, 3]`` minus
    1 is ``[3]``), and the run keeps its space as long as anything of it stays.
    """
    if not labels:
        return text

    def rewrite_bracket(found: re.Match[str]) -> str:
        numbers = [
            n for n in re.split(r"\s*[,;]\s*", found.group(1)) if int(n) not in labels
        ]
        return f"[{', '.join(numbers)}]" if numbers else ""

    def rewrite_run(found: re.Match[str]) -> str:
        kept = LABELS.sub(rewrite_bracket, found.group(2))
        return found.group(1) + kept if kept else ""

    return re.sub(r"([ \t]*)((?:" + LABELS.pattern + r")+)", rewrite_run, text)


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """``(start, end)`` of each non-empty sentence of ``text``; what lies between is whitespace."""
    spans: list[tuple[int, int]] = []
    start = 0
    for separator in _SENTENCE_END.finditer(text):
        spans.append((start, separator.start()))
        start = separator.end()
    spans.append((start, len(text)))
    return [(a, b) for a, b in spans if b > a]


def plain_sentence(sentence: str) -> str:
    """A sentence as a claim: its labels taken out and the spacing before punctuation tidied."""
    without = re.sub(r"\s+", " ", LABELS.sub("", sentence))
    return re.sub(r"\s+([.,;:!?])", r"\1", without).strip()


def split_claims(text: str) -> list[tuple[str, list[int]]]:
    """The sentences of ``text`` with the labels each cites.

    Returns:
        ``(sentence without its labels, labels)`` in order. Sentences too short to be a
        claim are left out.
    """
    claims: list[tuple[str, list[int]]] = []
    for start, end in sentence_spans(text):
        sentence = text[start:end]
        plain = plain_sentence(sentence)
        if len(plain) >= MIN_CLAIM_CHARS:
            claims.append((plain, cited_labels(sentence)))
    return claims
