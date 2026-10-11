"""A long evaluation saves each run and is resumed instead of started again."""

from corpus.commands.eval import load_saved_runs, save_run


def test_saved_runs_are_read_back_by_question_and_run_number(tmp_path):
    path = tmp_path / "runs.jsonl"
    save_run(path, "q1", 1, {"answer": "a", "grades": {"exact_match": {"x": True}}})
    save_run(path, "q1", 2, {"answer": "b"})
    save_run(path, "q2", 1, {"answer": "c"})

    saved = load_saved_runs(path)

    assert saved[("q1", 1)]["answer"] == "a" and saved[("q1", 2)]["answer"] == "b"
    assert saved[("q2", 1)]["answer"] == "c" and len(saved) == 3


def test_without_a_file_nothing_is_saved_or_loaded(tmp_path):
    save_run(None, "q1", 1, {"answer": "a"})

    assert load_saved_runs(None) == {}
    assert load_saved_runs(tmp_path / "missing.jsonl") == {}


def test_a_value_json_cannot_hold_is_saved_as_text(tmp_path):
    path = tmp_path / "runs.jsonl"

    save_run(path, "q1", 1, {"when": {1, 2}})

    assert "when" in load_saved_runs(path)[("q1", 1)]
