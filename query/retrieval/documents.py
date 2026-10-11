"""Ranking the documents of a ranked list of chunks.

Several chunks of one decision rank high together; this turns a chunk ranking into a
document ranking, which the document-reading step and the evaluation's identifiability
check both use.

Key exports:
    document_scores -- The documents with their scores, best first.
"""

from collections import defaultdict
from collections.abc import Sequence

from models import RetrievedChunk

#: Dampens the first ranks so that one lucky chunk does not outweigh several good ones.
_RANK_OFFSET = 10


def document_scores(ranked: Sequence[RetrievedChunk]) -> list[tuple[str, float]]:
    """``(source_file, score)`` of every document in ``ranked``, best first.

    A document scores the sum of ``1 / (10 + rank)`` over its chunks: a document with
    several chunks near the top beats one with a single chunk at the very top.
    """
    scores: defaultdict[str, float] = defaultdict(float)
    for rank, chunk in enumerate(ranked):
        scores[chunk.metadata.source_file] += 1 / (_RANK_OFFSET + rank)
    return sorted(scores.items(), key=lambda item: -item[1])
