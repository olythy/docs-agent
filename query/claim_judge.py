"""Does a quoted passage state the fact of the sentence it is cited for?

The answerer asks the model for a passage per citation and checks by code that the passage
is in the chunk. That makes the quote real, not relevant: a model that cites four excerpts
after one sentence can usually find *some* true passage in each. This asks one more,
separate call about the short pair (sentence, passage), which is much easier to judge than
the whole chunk, and the answerer drops the citations whose passage does not state the
sentence.

The evaluation judges the same citations from outside, against the whole chunk, with its own
prompt (``corpus/citation_support.py``): the runtime judge must not be the only judge.

Key exports:
    ClaimJudge     -- The contract.
    LLMClaimJudge  -- The model-backed implementation.
"""

import json
import logging
from collections.abc import Sequence
from typing import Protocol

from drivers.llm import AnswerDriver
from llm_json import extract_json

logger = logging.getLogger(__name__)


class ClaimJudge(Protocol):
    """Says, for ``(sentence, passage)`` pairs, whether the passage states the sentence."""

    def supports(self, pairs: Sequence[tuple[str, str]]) -> list[bool] | None:
        """One verdict per pair, or ``None`` if no usable answer could be had."""
        ...


class LLMClaimJudge:
    """Asks the answer driver, in one call for all pairs of an answer.

    Args:
        llm: The driver used for the call.
        temperature: Sampling temperature (0: a judgement, not a draft).
    """

    def __init__(self, llm: AnswerDriver, temperature: float = 0.0) -> None:
        self._llm = llm
        self._temperature = temperature

    def supports(self, pairs: Sequence[tuple[str, str]]) -> list[bool] | None:
        """See :class:`ClaimJudge`."""
        if not pairs:
            return []
        items = [
            {"id": i, "claim": claim, "passage": passage}
            for i, (claim, passage) in enumerate(pairs, start=1)
        ]
        prompt = (
            "You check the citations of a question-answering system. For each item, "
            "decide whether the PASSAGE itself states the fact(s) the CLAIM says. "
            "'supported' is true only if the passage contains those facts; it is false if "
            "the passage is only about the same case or topic, lacks the fact, or "
            "contradicts it. A claim often opens by naming the court, the case number or "
            "the parties, which the question already stated and the passage need not "
            "repeat: judge the fact the claim reports (an amount, a ruling, a reason, a "
            "date), not those words. Judge each item on its own passage only.\n\n"
            f"{json.dumps(items, ensure_ascii=False)}\n\n"
            'Output ONLY this JSON, one entry per item in the same order: {"supported": '
            "[true, false, ...]}"
        )
        reply = self._llm.run_tool_calling_turn(
            [{"role": "user", "content": prompt}], temperature=self._temperature
        ).content
        try:
            data = extract_json(reply or "")
        except (ValueError, json.JSONDecodeError):
            return None
        verdicts = data.get("supported") if isinstance(data, dict) else None
        if (
            not isinstance(verdicts, list)
            or len(verdicts) != len(pairs)
            or not all(isinstance(v, bool) for v in verdicts)
        ):
            logger.warning("[cite] The claim judge gave no usable answer.")
            return None
        return verdicts
