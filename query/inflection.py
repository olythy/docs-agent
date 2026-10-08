"""Hungarian word endings that stick to an identifier in a question.

In Hungarian a case ending is attached to a number with a hyphen: "a 4.P.20.409/2023/4-es
ügyben", "a BH2019.45-re hivatkozik". The ending is not part of the identifier, and left on it
the identifier matches nothing. This is a property of the **language**, not of any corpus,
so it lives here and not in the generic identifier rules (``metadata/identifiers.py``),
which stay language-free: another language would have its own module.

Key exports:
    strip_case_ending -- An identifier without a Hungarian case ending.
"""

import re

#: A hyphen and one to three lower-case letters at the very end, after a digit: "-es", "-re",
#: "-ban", "-nak". Longer or upper-case tails are not endings but parts of the identifier
#: ("-ítélet", "-III").
_CASE_ENDING = re.compile(r"^(?P<stem>.*\d)-[a-záéíóöőúüű]{1,3}$")


def strip_case_ending(token: str) -> str:
    """The token without a Hungarian case ending glued on with a hyphen.

    Args:
        token: An identifier-like token from a question.

    Returns:
        The token without the ending, or the token itself when it has none.
    """
    match = _CASE_ENDING.match(token)
    return match.group("stem") if match else token
