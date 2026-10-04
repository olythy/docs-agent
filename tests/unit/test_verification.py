

def test_extract_json_tolerates_a_raw_newline_inside_a_string():
    """Regression: a grader reply with a literal newline in "reason" crashed
    a whole eval run (json.loads strict mode)."""
    from corpus.verification import extract_json

    result = extract_json('{"verdict": "SUPPORTED", "reason": "line one\nline two"}')

    assert result == {"verdict": "SUPPORTED", "reason": "line one\nline two"}


def test_extract_json_repairs_a_backslash_that_is_not_a_valid_escape():
    """Regression: a grader reply with '\\§' or '\\ ' in a string crashed a whole
    eval run ('Invalid \\escape') even after strict=False."""
    from corpus.verification import extract_json

    result = extract_json(r'{"verdict": "SUPPORTED", "reason": "az 5:84. \§ (1) szerint"}')

    assert result["verdict"] == "SUPPORTED"
    assert "§" in result["reason"]


def test_extract_json_keeps_valid_escapes_intact():
    from corpus.verification import extract_json

    assert extract_json(r'{"a": "line\nbreak \"quoted\" \\ back"}') == {
        "a": 'line\nbreak "quoted" \\ back'
    }
