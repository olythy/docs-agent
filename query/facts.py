"""What can be read from a question without asking a model: identifiers and years.

The retrieval pipeline needs two plain facts about the question text, the identifier
tokens it names (case numbers, invoice numbers, ...) and the years it mentions. They
used to be detected where they were needed (in the router, in the retrieval, in a
strategy). :class:`QueryFactsReader` reads them **once**, and the result travels with
the question; a profile decides whether a fact is used (years are always read).

Key exports:
    QueryFacts       -- The facts of one question.
    QueryFactsReader -- Reads them.
"""

from dataclasses import dataclass

from query.inflection import strip_case_ending
from query.time_filter import extract_years
from store import extract_identifier_tokens


@dataclass(frozen=True)
class QueryFacts:
    """The plain facts of a question.

    Attributes:
        question: The question as asked.
        identifiers: Identifier-like tokens in it, in the order they appear, without a Hungarian
            case ending glued on ("4.P.20.409/2023/4-es" -> "4.P.20.409/2023/4").
        years: The distinct years it refers to, ascending (a range is expanded; years
            inside an identifier do not count).
    """

    question: str
    identifiers: tuple[str, ...] = ()
    years: tuple[int, ...] = ()


class QueryFactsReader:
    """Reads :class:`QueryFacts` from a question (pure, deterministic, no model call)."""

    def read(self, question: str) -> QueryFacts:
        """Read the facts of ``question``.

        Args:
            question: The user's question.
        """
        identifiers = dict.fromkeys(
            strip_case_ending(token) for token in extract_identifier_tokens(question)
        )
        return QueryFacts(
            question=question,
            identifiers=tuple(identifiers),
            years=tuple(extract_years(question)),
        )
