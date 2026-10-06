"""Parsing JSON out of an LLM reply, tolerantly.

Models wrap JSON in code fences, put a raw newline inside a string, or emit a
backslash that is not a valid JSON escape; each of those has aborted a real
run of this project. One tolerant parser, shared by everything that reads a
model's JSON (the corpus tooling and the metadata extractor alike).

Key exports:
    extract_json   -- Parse the first JSON object out of an LLM response.
    DuplicateKeyError -- An object that has the same key twice (see ``extract_json``).
"""

import json
import re

#: A backslash that does not start a valid JSON escape (\", \\, \/, \b, \f, \n,
#: \r, \t, \uXXXX); doubling it turns it into a literal backslash.
_INVALID_ESCAPE = re.compile(r'\\(?!["\\/bfnrtu])')


class DuplicateKeyError(ValueError):
    """A JSON object that has the same key twice.

    Python's parser silently keeps the *last* value, so a model that fuses two
    list items into one object (``{"key": "a", ..., "key": "b", ...}``) loses the
    first item without any error: for a query plan that is a filter that silently
    disappears, and a wrong count is then presented as exact.
    """


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    """``object_pairs_hook`` that refuses an object with a repeated key."""
    seen: set[str] = set()
    for key, _ in pairs:
        if key in seen:
            raise DuplicateKeyError(
                f"the JSON object has the key {key!r} twice (two items fused into one "
                "object? each list item needs its own braces)"
            )
        seen.add(key)
    return dict(pairs)


def extract_json(text: str, reject_duplicate_keys: bool = False) -> dict:
    """Parse the first JSON object found in an LLM response.

    Models sometimes wrap JSON in ```` ```json ... ``` ```` fences despite
    being asked not to -- strip those before parsing rather than failing.
    Also tolerates raw control characters (e.g. a literal newline) inside
    JSON strings, which ``json.loads``' default strict mode rejects, and
    backslashes that are not a valid JSON escape (``"\\ "``, ``"\\§"``):
    confirmed live, each of those grader replies aborted a whole 33-question
    eval run.

    Args:
        text: The model's reply.
        reject_duplicate_keys: Raise :class:`DuplicateKeyError` for an object that
            repeats a key, instead of keeping the last value. Use it where a lost
            item would silently change the result (a query plan). Off by default.

    Raises:
        json.JSONDecodeError: If the text is still not JSON after the repairs.
        DuplicateKeyError: If ``reject_duplicate_keys`` and a key is repeated.
    """
    hook = _no_duplicate_keys if reject_duplicate_keys else None
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped
        stripped = stripped.rsplit("```", 1)[0]
    try:
        return json.loads(stripped, strict=False, object_pairs_hook=hook)
    except json.JSONDecodeError:
        return json.loads(
            _INVALID_ESCAPE.sub(r"\\\\", stripped),
            strict=False,
            object_pairs_hook=hook,
        )
