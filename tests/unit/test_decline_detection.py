"""query.decline_detection: does an answer read as an honest \"not found\"?"""

from query.decline_detection import looks_like_a_decline


def test_looks_like_a_decline_english():
    assert (
        looks_like_a_decline("I could not find information about that in the document.")
        is True
    )
    assert (
        looks_like_a_decline("The provided context does not mention the fee.") is True
    )
    assert looks_like_a_decline("Unable to find the requested details.") is True


def test_looks_like_a_decline_hungarian():
    assert (
        looks_like_a_decline("A megadott szövegben nem találtam információt erről.")
        is True
    )
    assert looks_like_a_decline("A dokumentumban nem szerepel az adószám.") is True
    assert looks_like_a_decline("Nincs információ a nyitvatartásról.") is True


def test_looks_like_a_decline_false_for_valid_answers():
    assert looks_like_a_decline("The company was founded in 2018 in Budapest.") is False
    assert looks_like_a_decline("A projekt költségvetése 5 millió forint.") is False


def test_the_systems_own_plain_refusal_counts_as_a_decline():
    assert looks_like_a_decline(
        "No documents match the filter (court_decision documents: document_identifier "
        "contains 'PK-987654')."
    )
    assert not looks_like_a_decline("A bíróság a keresetet elutasította.")


def test_the_systems_other_plain_refusals_are_declines_too():
    """Every refusal the system itself words must read as one to the eval."""
    assert looks_like_a_decline(
        "This kind of question is not supported yet, so I will not guess at an answer. "
        "(five cases similar to the 27.P.20.339/2021/37 case)"
    )
    assert looks_like_a_decline(
        "I could not interpret this question well enough to answer it from the "
        "structured data, and I did not want to guess."
    )
    assert not looks_like_a_decline(
        "The supported claim was granted."
    )  # not just any "support"
