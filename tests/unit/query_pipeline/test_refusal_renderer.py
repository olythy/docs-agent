"""The sentence a person is told for a refusal: the original wording, word for word."""

import pytest

from query.decline_detection import looks_like_a_decline
from query.outcome import (
    COULD_NOT_INTERPRET_MESSAGE,
    NO_RESULTS_MESSAGE,
    NOT_SUPPORTED_MESSAGE,
    Declined,
    DeclineReason,
    RefusalRenderer,
    no_matching_documents_message,
)

render = RefusalRenderer().render


class TestTheWords:
    def test_a_question_that_could_not_be_interpreted(self):
        declined = Declined(DeclineReason.COULD_NOT_INTERPRET, "planning", "bad key")

        assert (
            render(declined) == COULD_NOT_INTERPRET_MESSAGE
        )  # the detail is for the log

    def test_a_request_that_is_not_supported_yet_carries_the_planners_reason(self):
        declined = Declined(
            DeclineReason.NOT_SUPPORTED, "planning", "five similar cases"
        )

        assert render(declined) == f"{NOT_SUPPORTED_MESSAGE} (five similar cases)"

    def test_without_a_reason_it_is_just_the_message(self):
        assert (
            render(Declined(DeclineReason.NOT_SUPPORTED, "planning"))
            == NOT_SUPPORTED_MESSAGE
        )

    def test_a_filter_that_selects_nothing_names_the_filter(self):
        declined = Declined(
            DeclineReason.NO_MATCHING_DOCUMENTS,
            "scope",
            "court_decision documents: x = 'y'",
        )

        assert render(declined) == (
            "No documents match the filter (court_decision documents: x = 'y')."
        )

    def test_the_note_follows_on_the_same_line_after_a_space(self):
        declined = Declined(
            DeclineReason.NO_MATCHING_DOCUMENTS,
            "scope",
            "f",
            note="Note: 2 could not be checked.",
        )

        assert (
            render(declined)
            == "No documents match the filter (f). Note: 2 could not be checked."
        )

    @pytest.mark.parametrize(
        "reason", [DeclineReason.NOT_RELEVANT, DeclineReason.RERANK_REJECTED]
    )
    def test_both_retrieval_gates_say_the_same_to_the_person(self, reason):
        """They differ only in the explain record: which gate refused."""
        assert render(Declined(reason, "some_gate")) == NO_RESULTS_MESSAGE


class TestTheWordsAreFixed:
    """The eval's decline detection and the adversarial questions depend on these exact
    sentences, so they are written out here, not read back from the constants."""

    def test_the_fixed_sentences(self):
        assert NO_RESULTS_MESSAGE == (
            "I could not find relevant information about this in the provided documents."
        )
        assert NOT_SUPPORTED_MESSAGE == (
            "This kind of question is not supported yet, so I will not guess at an answer."
        )
        assert COULD_NOT_INTERPRET_MESSAGE == (
            "I could not interpret this question well enough to answer it from the "
            "structured data, and I did not want to guess."
        )

    def test_the_empty_filter_message_is_built_in_one_place(self):
        assert (
            no_matching_documents_message("f", "n")
            == "No documents match the filter (f). n"
        )
        assert (
            no_matching_documents_message("f") == "No documents match the filter (f)."
        )


class TestTheEvalCanTellItIsARefusal:
    """query.decline_detection reads these sentences; none of them may stop being one."""

    @pytest.mark.parametrize(
        "declined",
        [
            Declined(DeclineReason.COULD_NOT_INTERPRET, "planning", "x"),
            Declined(DeclineReason.NOT_SUPPORTED, "planning", "similar cases"),
            Declined(DeclineReason.NO_MATCHING_DOCUMENTS, "scope", "f"),
            Declined(DeclineReason.NOT_RELEVANT, "relevance_gate"),
            Declined(DeclineReason.RERANK_REJECTED, "rerank_score_gate"),
        ],
        ids=lambda d: d.reason.value,
    )
    def test_every_refusal_reads_as_a_decline(self, declined):
        assert looks_like_a_decline(render(declined))
