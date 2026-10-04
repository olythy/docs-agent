"""Parsing JSON out of an LLM reply, tolerantly.

Models wrap JSON in code fences, put a raw newline inside a string, or emit a
backslash that is not a valid JSON escape; each of those has aborted a real
run of this project. One tolerant parser, shared by everything that reads a
model's JSON (the corpus tooling and the metadata extractor alike).

Key exports:
    extract_json -- Parse the first JSON object out of an LLM response.
"""

import json
import re

#: A backslash that does not start a valid JSON escape (\", \\, \/, \b, \f, \n,
#: \r, \t, \uXXXX); doubling it turns it into a literal backslash.
_INVALID_ESCAPE = re.compile(r'\\(?!["\\/bfnrtu])')


def extract_json(text: str) -> dict:
    """Parse the first JSON object found in an LLM response.

    Models sometimes wrap JSON in ```` ```json ... ``` ```` fences despite
    being asked not to -- strip those before parsing rather than failing.
    Also tolerates raw control characters (e.g. a literal newline) inside
    JSON strings, which ``json.loads``' default strict mode rejects, and
    backslashes that are not a valid JSON escape (``"\\ "``, ``"\\§"``):
    confirmed live, each of those grader replies aborted a whole 33-question
    eval run.

    Raises:
        json.JSONDecodeError: If the text is still not JSON after the repairs.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped
        stripped = stripped.rsplit("```", 1)[0]
    try:
        return json.loads(stripped, strict=False)
    except json.JSONDecodeError:
        return json.loads(_INVALID_ESCAPE.sub(r"\\\\", stripped), strict=False)
