"""Grading an adversarial answer: did the system decline, without asserting the missing fact?

The keyword check in ``query.decline_detection`` only looks for decline phrases. It counts
an answer that says "not mentioned" while it goes on to state the fact as a decline, and
misses an honest answer that says what the documents do contain and flags what they do not
(the partial answer a ``missing_fact`` trap rightly gets). A separate model call reads the
question and the answer and decides the two things apart.

Key exports:
    DeclineGrade   -- The verdict on one answer.
    grade_decline  -- Runs the check with a given judge.
    llm_decline_judge -- The judge that asks the answer driver.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass

from llm_json import extract_json

Judge = Callable[[str, str], dict | None]


@dataclass(frozen=True)
class DeclineGrade:
    """The verdict on the answer to an unanswerable question.

    Attributes:
        declined: The answer refuses, or says explicitly that the asked detail is missing.
        asserted: The answer states the asked detail as if it were known.
        correct: Declined and did not assert it.
    """

    declined: bool
    asserted: bool

    @property
    def correct(self) -> bool:
        """The right behaviour: say it is not there, and do not make it up."""
        return self.declined and not self.asserted


def grade_decline(question: str, answer: str, judge: Judge) -> DeclineGrade | None:
    """Grade ``answer`` to an unanswerable ``question``.

    Args:
        question: The unanswerable question.
        answer: The system's answer text.
        judge: Takes ``(question, answer)`` and returns ``{"declined": bool, "asserted":
            bool}``, or ``None`` if it cannot.

    Returns:
        The grade, or ``None`` when the judge gave no usable reply.
    """
    verdict = judge(question, answer)
    if not isinstance(verdict, dict):
        return None
    declined, asserted = verdict.get("declined"), verdict.get("asserted")
    if not isinstance(declined, bool) or not isinstance(asserted, bool):
        return None
    return DeclineGrade(declined, asserted)


def llm_decline_judge(question: str, answer: str) -> dict | None:
    """Ask the answer driver (a call apart from the one that wrote the answer)."""
    from drivers.llm import get_answer_driver

    prompt = (
        "The question below cannot be answered from the documents: it asks for a detail "
        "or a case that does not exist in them. Read the system's ANSWER and decide two "
        "things separately.\n"
        "- declined: true if the answer refuses, or says explicitly that the detail "
        "asked for is not available or not stated (it may add what the documents do "
        "contain).\n"
        "- asserted: true if the answer states the detail asked for (a date, an amount, "
        "an outcome, a fact about the asked case) as if it were known, whatever else it "
        "says. Stating a different, true fact from the documents is not asserting.\n\n"
        f"QUESTION\n{question}\n\nANSWER\n{answer}\n\n"
        'Output ONLY this JSON: {"declined": true | false, "asserted": true | false}'
    )
    reply = (
        get_answer_driver()
        .run_tool_calling_turn([{"role": "user", "content": prompt}], temperature=0.0)
        .content
    )
    try:
        data = extract_json(reply or "")
    except (ValueError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None
