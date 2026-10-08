"""Tests for corpus.commands.eval's question selection and repeat summary."""

import pytest
import typer

from corpus.commands.eval import print_repeat_summary, select_questions

QUESTIONS = [
    {"id": "q0001"},
    {"id": "q0010"},
    {"id": "q0012"},
]


def test_select_questions_returns_everything_without_ids():
    assert select_questions(QUESTIONS, None) == QUESTIONS
    assert select_questions(QUESTIONS, []) == QUESTIONS


def test_select_questions_keeps_only_the_requested_ids_in_the_order_given():
    result = select_questions(QUESTIONS, ["q0012", "q0001"])

    assert [q["id"] for q in result] == ["q0012", "q0001"]


def test_select_questions_ignores_a_repeated_id():
    assert [q["id"] for q in select_questions(QUESTIONS, ["q0010", "q0010"])] == [
        "q0010"
    ]


def test_select_questions_rejects_an_unknown_id_and_lists_the_valid_ones():
    with pytest.raises(typer.BadParameter, match="q9999"):
        select_questions(QUESTIONS, ["q0010", "q9999"])


def _run(qid, answer_ok, retrieval_ok, answer="valasz"):
    return {
        "question_id": qid,
        "persona_id": "synthesizer",
        "answer": answer,
        "grades": {
            "independent_fact": {
                "answer_correct": answer_ok,
                "retrieval_hit": retrieval_ok,
            }
        },
    }


def test_repeat_summary_reports_pass_counts_and_flags_a_question_below_target(capsys):
    results = [
        _run("q0010", True, True),
        _run("q0010", False, True, answer="I could not find this information"),
        _run("q0012", True, True),
        _run("q0012", True, True),
    ]

    print_repeat_summary(results, target=0.95)

    out = capsys.readouterr().out
    q0010 = out.split("q0010")[1].split("q0012")[0]
    q0012 = out.split("q0012")[1]
    assert "answer correct 1/2" in q0010
    assert "refused 1/2" in q0010
    assert "below target" in q0010
    assert "answer correct 2/2" in q0012
    assert "below target" not in q0012


class _FakeStore:
    """Stands in for store.VectorStore: only the ingested file names matter here."""

    def get_all_source_files(self):
        return {"A_P_1_2020_1.docx", "B_P_2_2021_2.docx"}

    def search_by_identifier(self, tokens, top_k, per_token=False):
        raise AssertionError(
            "must not fall back to identifier search when files are named"
        )


def test_cited_documents_are_the_file_names_the_answer_names(monkeypatch):
    """Regression: a fraction ('15/100-ad') and a statute fragment ('1)-(2') in
    the answer were matched literally against document content, 'citing'
    unrelated courts' documents the answer never mentioned."""
    import store
    from corpus.commands.eval import _resolve_cited_source_files

    monkeypatch.setattr(store, "VectorStore", _FakeStore)
    answer = (
        "A felperes 15/100-ad tulajdoni illetősége (Ptk. 5:84. § (1)-(2)) "
        "(A_P_1_2020_1.docx, 3. oldal) és (B_P_2_2021_2.docx, 2. oldal); "
        "lásd még Nincs_Ilyen.docx."
    )

    assert _resolve_cited_source_files(answer) == [
        "A_P_1_2020_1.docx",
        "B_P_2_2021_2.docx",
    ]


def test_the_grader_reads_only_a_few_cited_documents_and_says_so(monkeypatch):
    """An answer listing ~50 documents must not send them all in one request."""
    from types import SimpleNamespace

    import corpus.commands.eval as eval_module

    sent: list[str] = []

    class Driver:
        def run_tool_calling_turn(self, messages):
            sent.append(messages[0]["content"])
            return SimpleNamespace(content='{"verdict": "SUPPORTED", "reason": "fine"}')

    monkeypatch.setattr(
        eval_module, "fetch_full_content", lambda f: f + ":" + "x" * 200_000
    )
    monkeypatch.setattr("drivers.llm.get_answer_driver", lambda: Driver())
    files = [f"doc{i}.docx" for i in range(50)]

    verdict, reason = eval_module._verify_answer_claim_support("q", "a", files)

    prompt = sent[0]
    assert verdict == "SUPPORTED"
    assert reason == "[graded on the first 5 of 50 cited documents] fine"
    assert "doc4.docx" in prompt and "doc5.docx" not in prompt
    assert len(prompt) < 5 * 61_000 + 5_000  # each document is cut, too


class TestAnswerAndRetrieved:
    """The eval asks the query service once, and grades the chunks its explanation holds."""

    def test_v2_asks_the_service_once_with_the_strategy_as_the_profile(
        self, monkeypatch
    ):
        import query.composition as composition_module
        from corpus.commands.eval import _answer_and_retrieved
        from models import ChunkMetadata, RetrievedChunk
        from query.decision import ReadDocuments, Scope
        from query.facts import QueryFacts
        from query.outcome import Answerable
        from query.query_service import Answer, Explain
        from query.service import RetrievalResult

        chunk = RetrievedChunk(
            id=1,
            content="c",
            metadata=ChunkMetadata(source_file="a.pdf", page_number=1, chunk_index=0),
            score=0.5,
        )
        calls = []

        class FakeService:
            def answer(self, question, **kwargs):
                calls.append((question, kwargs))
                decision = ReadDocuments(QueryFacts(question), None, "vector", Scope())
                result = RetrievalResult(Answerable((chunk,)), ())
                return Answer("TEXT", Explain(decision, result))

        monkeypatch.setattr(
            composition_module, "build_query_service", lambda settings: FakeService()
        )

        answer, chunks = _answer_and_retrieved("Q?", "vector")

        assert (answer, chunks) == ("TEXT", [chunk])
        assert calls == [("Q?", {"profile": "vector"})]


class TestCitationRanks:
    """The diagnostic ranks each cited document in a wide pool, over all documents."""

    def test_documents_are_ranked_by_first_appearance_and_missing_ones_are_none(
        self, monkeypatch
    ):
        import query.composition as composition_module
        from corpus.commands.eval import DIAGNOSTIC_POOL_SIZE, _citation_ranks
        from models import ChunkMetadata, RetrievedChunk
        from query.outcome import Answerable
        from query.service import RetrievalResult

        def chunk(i, source):
            return RetrievedChunk(
                id=i,
                content="c",
                metadata=ChunkMetadata(
                    source_file=source, page_number=1, chunk_index=i
                ),
                score=0.5,
            )

        requests = []

        class FakeRetrieval:
            def retrieve(self, request, store):
                requests.append((request, store))
                chunks = (chunk(1, "a.pdf"), chunk(2, "a.pdf"), chunk(3, "b.pdf"))
                return RetrievalResult(Answerable(chunks), ())

        monkeypatch.setattr(
            composition_module, "build_retrieval_service", lambda s: FakeRetrieval()
        )

        ranks = _citation_ranks(
            "q", [{"source_file": "b.pdf"}, {"source_file": "z.pdf"}], "hybrid"
        )

        assert ranks == {"b.pdf": 2, "z.pdf": None}
        request, _ = requests[0]
        assert (request.profile, request.top_k) == ("hybrid", DIAGNOSTIC_POOL_SIZE)
        assert request.scope is None  # over all documents
