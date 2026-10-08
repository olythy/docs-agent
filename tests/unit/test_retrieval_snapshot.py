"""Tests for the pure part of corpus.commands.retrieval_snapshot: comparing two snapshots."""

import copy

from corpus.commands.retrieval_snapshot import diff_snapshots

BASE = {
    "settings": {"RETRIEVAL_TOP_K": 4, "RERANKER_DRIVER": "vertex"},
    "questions": {
        "q1": {
            "question": "?",
            "stages": {"vector": [1, 2, 3], "fused": [2, 1, 3], "final": [2, 1]},
            "final": [2, 1],
            "gate_passed": True,
        },
        "q2": {
            "question": "?",
            "stages": {"vector": [9]},
            "final": [],
            "gate_passed": False,
        },
    },
}


def test_identical_snapshots_have_no_differences():
    assert diff_snapshots(BASE, copy.deepcopy(BASE)) == []


def test_a_changed_stage_and_final_are_each_reported_with_both_values():
    other = copy.deepcopy(BASE)
    other["questions"]["q1"]["stages"]["fused"] = [1, 2, 3]
    other["questions"]["q1"]["final"] = [1, 2]

    problems = diff_snapshots(BASE, other)

    assert problems == [
        "q1 stage 'fused': [2, 1, 3] -> [1, 2, 3]",
        "q1 final: [2, 1] -> [1, 2]",
    ]


def test_a_stage_that_appears_or_disappears_is_a_difference():
    other = copy.deepcopy(BASE)
    other["questions"]["q1"]["stages"]["listwise"] = [2]
    del other["questions"]["q1"]["stages"]["vector"]

    problems = diff_snapshots(BASE, other)

    assert "q1 stage 'listwise': None -> [2]" in problems
    assert "q1 stage 'vector': [1, 2, 3] -> None" in problems


def test_a_gate_that_now_fails_or_passes_is_reported():
    other = copy.deepcopy(BASE)
    other["questions"]["q2"]["gate_passed"] = True

    assert diff_snapshots(BASE, other) == ["q2 relevance gate: False -> True"]


def test_a_question_missing_from_one_side_is_reported():
    other = copy.deepcopy(BASE)
    del other["questions"]["q2"]

    assert diff_snapshots(BASE, other) == ["q2: only in the old snapshot"]


def test_different_settings_make_the_comparison_meaningless_and_say_so():
    other = copy.deepcopy(BASE)
    other["settings"]["RETRIEVAL_TOP_K"] = 8

    problems = diff_snapshots(BASE, other)

    assert problems == ["settings differ (RETRIEVAL_TOP_K: 4 -> 8): not comparable"]


def test_the_stage_ids_name_each_list_by_its_step_and_keep_side_results():
    from models import ChunkMetadata, RetrievedChunk
    from query.context import Slot
    from query.runner import StageRecord

    def chunk(i):
        return RetrievedChunk(
            id=i,
            content="c",
            metadata=ChunkMetadata(source_file="a", page_number=1, chunk_index=i),
            score=0.1,
        )

    records = [
        StageRecord(
            "dense_search",
            {},
            {Slot.DENSE_POOL: (chunk(1), chunk(2))},
            {},
            {},
            0.0,
        ),
        StageRecord(
            "year_dense_widening",
            {},
            {Slot.DENSE_POOL: (chunk(3),)},
            {"year_pool": (chunk(4),)},
            {},
            0.0,
        ),
        StageRecord("relevance_gate", {}, {}, {}, {"gate_passed": True}, 0.0),
    ]

    from corpus.commands.retrieval_snapshot import stage_ids

    assert stage_ids(records) == {
        "dense_search.dense_pool": [1, 2],
        "year_dense_widening.dense_pool": [3],
        "year_dense_widening.year_pool": [4],
    }


def test_the_models_are_part_of_what_makes_two_snapshots_comparable():
    from corpus.commands.retrieval_snapshot import _SETTINGS

    assert {"EMBEDDING_MODEL", "RERANKER_MODEL"} <= set(_SETTINGS)
