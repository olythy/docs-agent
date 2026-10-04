

def test_extract_json_tolerates_a_raw_newline_inside_a_string():
    """Regression: a grader reply with a literal newline in "reason" crashed
    a whole eval run (json.loads strict mode)."""
    from corpus.verification import extract_json

    result = extract_json('{"verdict": "SUPPORTED", "reason": "line one\nline two"}')

    assert result == {"verdict": "SUPPORTED", "reason": "line one\nline two"}
