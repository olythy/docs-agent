"""Tests for query.hybrid.reciprocal_rank_fusion (pure logic, no DB needed)."""

from query.hybrid import reciprocal_rank_fusion


def _chunk(chunk_id, content="c", score=0.0, page=1):
    return {"id": chunk_id, "content": content, "metadata": {"page_number": page}, "score": score}


def test_chunk_in_both_lists_combines_scores():
    vector_results = [_chunk(1, score=0.9)]
    fulltext_results = [_chunk(1, score=0.42)]

    fused = reciprocal_rank_fusion(vector_results, fulltext_results, k=60)

    assert len(fused) == 1
    expected = 1 / (1 + 60) + 1 / (1 + 60)
    assert fused[0]["score"] == expected


def test_chunk_in_only_one_list_still_included():
    vector_results = [_chunk(1, score=0.9)]
    fulltext_results = []

    fused = reciprocal_rank_fusion(vector_results, fulltext_results, k=60)

    assert len(fused) == 1
    assert fused[0]["id"] == 1
    assert fused[0]["score"] == 1 / (1 + 60)


def test_result_is_sorted_by_descending_fused_score():
    # id 2 appears in both lists (should win); id 1 only in vector; id 3 only in fulltext.
    vector_results = [_chunk(2, score=0.8), _chunk(1, score=0.5)]
    fulltext_results = [_chunk(2, score=0.9), _chunk(3, score=0.3)]

    fused = reciprocal_rank_fusion(vector_results, fulltext_results, k=60)

    assert [c["id"] for c in fused] == [2, 1, 3]


def test_k_parameter_changes_relative_weighting():
    vector_results = [_chunk(1, score=0.9)]
    fulltext_results = []

    fused_small_k = reciprocal_rank_fusion(vector_results, fulltext_results, k=0)
    fused_large_k = reciprocal_rank_fusion(vector_results, fulltext_results, k=1000)

    assert fused_small_k[0]["score"] == 1 / (1 + 0)
    assert fused_large_k[0]["score"] == 1 / (1 + 1000)
    assert fused_small_k[0]["score"] > fused_large_k[0]["score"]


def test_preserves_content_and_metadata():
    vector_results = [_chunk(1, content="hello world", page=3, score=0.9)]

    fused = reciprocal_rank_fusion(vector_results, [])

    assert fused[0]["content"] == "hello world"
    assert fused[0]["metadata"] == {"page_number": 3}
