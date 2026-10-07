"""Heuristic detection of an honest "I don't know" reply, in English or Hungarian.

Shared between scripts/eval_cli.py's LLM generation benchmark and
corpus/commands/eval.py's golden-set evaluation -- both need to tell an
honest decline apart from a hallucinated answer for unanswerable questions.
"""

_DECLINE_PHRASES = [
    "could not find",
    "does not mention",
    "cannot find",
    "not mentioned",
    "not provided",
    "no information",
    "unable to find",
    "nem találtam",
    "nem található",
    "nem tartalmaz",
    "nem szerepel",
    "nincs információ",
    "nem tér ki",
    "nem derül ki",
    "nem állapítható meg",
    # the router's own plain refusals: the named thing matches no document, the kind of
    # request is not supported yet, the question could not be interpreted. All are honest
    # declines, so for an unanswerable question they are right, and for an answerable one
    # they are a plain failure rather than an answer.
    "no documents match",
    "not supported yet",
    "could not interpret",
]


def looks_like_a_decline(answer: str) -> bool:
    """Return True if ``answer`` reads as an honest "not found" reply.

    A simple keyword heuristic, not an LLM judgment call -- deliberately
    cheap and deterministic, since this only needs to catch the standard
    decline phrasing this project's own system prompts (drivers/llm.py's
    _build_prompt()) instruct the model to use, not arbitrary rephrasing.

    Args:
        answer: The model's generated answer text.

    Returns:
        True if any known decline phrase (English or Hungarian) appears.
    """
    lower = answer.lower()
    return any(phrase in lower for phrase in _DECLINE_PHRASES)
