"""Can the document a question describes be told apart from the others?

A practical question names no case number: a court, a year and what the lawsuit is about
have to identify the decision. When the description fits several decisions equally well,
no system can know which one is meant, and the right behaviour is to ask back or to list
them, not to answer from one: such a question measures nothing about retrieval or answers.
The generation checks that a description fits only its own document were made by a model
reading summaries and let too many through (23% of the practical questions turned out to
have the right document ranked first by the system's own search in only 77% of the cases).

This module ranks the documents of a retrieval's candidate list and says where the golden
document stands. The ranking is the system's own, so the gate is circular on purpose in one
way: "identifiable" means "the best document for this description is the golden one",
which is what a person could also be expected to reach.

Key exports:
    document_ranking -- The same without the scores.
    gold_position    -- Where the golden document stands (0 is first), or ``None``.
"""

from collections.abc import Sequence

from models import RetrievedChunk
from query.retrieval.documents import document_scores


def document_ranking(ranked: Sequence[RetrievedChunk]) -> list[str]:
    """The ``source_file`` of every document in ``ranked``, best first."""
    return [source for source, _ in document_scores(ranked)]


def gold_position(ranked: Sequence[RetrievedChunk], gold: str) -> int | None:
    """Where ``gold`` stands in :func:`document_ranking` (0 is first), or ``None``."""
    ranking = document_ranking(ranked)
    return ranking.index(gold) if gold in ranking else None
