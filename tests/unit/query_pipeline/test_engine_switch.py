"""How production reaches the new pipeline: ``retrieve_chunks`` with ``QUERY_ENGINE=v2``.

The scenarios already prove that the result is the same; these cover what they cannot:
the guards on the switch, that the caller's overrides arrive, and that the composition
root wires the observers and the listwise reranker as the settings say.
"""

import json

import pytest
from test_retrieval_characterization import (  # type: ignore[import-not-found]
    FakeCrossEncoder,
    FakeEmbedding,
    FakeListwise,
    FakeStore,
)

import query.composition as composition_module
import query.retrieval as retrieval_module
from logger import EventLogger
from query.retrieval import (
    HybridRetrievalStrategy,
    RetrievalStrategy,
    VectorRetrievalStrategy,
    retrieve_chunks,
)

QUESTION = "What about the costs of the proceedings?"


@pytest.fixture
def wired(monkeypatch, settings_override, tmp_path):
    """Patch the drivers the way the scenarios do; returns a function that sets the
    settings and the audit log path."""

    def wire(**settings):
        config = settings_override(
            **{
                "RETRIEVAL_STRATEGY": "hybrid",
                "RERANKER_DRIVER": "cross_encoder",
                "RERANKER_MODEL": "model-x",
                "RETRIEVAL_TOP_K": 4,
                "RETRIEVAL_CANDIDATE_POOL_SIZE": 6,
                "RETRIEVAL_MIN_SCORE": 0.25,
                "RERANKER_MIN_SCORE": 0.5,
                "QUERY_ENGINE": "v2",
                **settings,
            }
        )
        monkeypatch.setattr(retrieval_module, "settings", config)
        monkeypatch.setattr(composition_module, "get_embedding_driver", FakeEmbedding)
        monkeypatch.setattr(
            composition_module, "get_reranker_driver", lambda: FakeCrossEncoder()
        )
        monkeypatch.setattr(composition_module, "get_answer_driver", lambda: "llm")
        monkeypatch.setattr(composition_module, "listwise_rerank", FakeListwise())
        path = tmp_path / "log.jsonl"
        monkeypatch.setattr(composition_module, "get_logger", lambda: EventLogger(path))
        return path

    return wire


class TestGuards:
    def test_an_unknown_engine_is_an_error_not_a_silent_default(self, wired):
        wired(QUERY_ENGINE="v3")

        with pytest.raises(ValueError, match="Unknown QUERY_ENGINE"):
            retrieve_chunks(QUESTION, store=FakeStore())  # type: ignore[arg-type]

    def test_a_plain_vector_strategy_runs_the_vector_profile(self, wired):
        wired()  # the settings say hybrid; the strategy object says vector

        chunks = retrieve_chunks(
            QUESTION,
            strategy=VectorRetrievalStrategy(),
            store=FakeStore(),  # type: ignore[arg-type]
        )

        assert [c.id for c in chunks] == [
            1,
            3,
            5,
            7,
        ]  # the cosine cut, no keyword/rerank

    def test_a_plain_hybrid_strategy_runs_the_hybrid_profile(self, wired):
        wired(
            RETRIEVAL_STRATEGY="vector"
        )  # the settings say vector; the object says hybrid

        chunks = retrieve_chunks(
            QUESTION,
            strategy=HybridRetrievalStrategy(),
            store=FakeStore(),  # type: ignore[arg-type]
        )

        assert [c.id for c in chunks] == [
            1,
            3,
            5,
        ]  # the cross-encoder threshold dropped one

    @pytest.mark.parametrize(
        "overrides",
        [
            {"reranker_driver_name": "none"},
            {"diversify_guarantees": False},
            {"period_filter": True},
        ],
    )
    def test_a_strategy_with_its_own_overrides_is_refused_loudly(
        self, wired, overrides
    ):
        wired()

        with pytest.raises(ValueError, match="set them in the settings"):
            retrieve_chunks(
                QUESTION,
                strategy=HybridRetrievalStrategy(**overrides),
                store=FakeStore(),  # type: ignore[arg-type]
            )

    def test_an_override_equal_to_the_settings_is_not_an_override(self, wired):
        wired(RETRIEVAL_PERIOD_FILTER=True, RETRIEVAL_DIVERSIFY_GUARANTEES=False)

        chunks = retrieve_chunks(
            QUESTION,
            strategy=HybridRetrievalStrategy(
                period_filter=True, diversify_guarantees=False
            ),
            store=FakeStore(),  # type: ignore[arg-type]
        )

        assert chunks  # accepted

    def test_any_other_kind_of_strategy_is_refused(self, wired):
        wired()

        class Custom(RetrievalStrategy):
            def select_chunks(self, *args, **kwargs):
                return []

        with pytest.raises(ValueError, match="cannot run a Custom"):
            retrieve_chunks(
                QUESTION,
                strategy=Custom(),
                store=FakeStore(),  # type: ignore[arg-type]
            )

    def test_an_unknown_profile_is_an_error(self, wired):
        wired(RETRIEVAL_STRATEGY="nope")

        with pytest.raises(ValueError, match="Unknown retrieval profile"):
            retrieve_chunks(QUESTION, store=FakeStore())  # type: ignore[arg-type]


class TestOverrides:
    def test_top_k_and_min_score_reach_the_pipeline(self, wired):
        wired()

        two = retrieve_chunks(QUESTION, top_k=2, store=FakeStore())  # type: ignore[arg-type]
        none = retrieve_chunks(QUESTION, min_score=0.95, store=FakeStore())  # type: ignore[arg-type]

        assert [c.id for c in two] == [1, 3]
        assert none == []

    def test_a_precomputed_query_vector_means_no_embedding_call(
        self, wired, monkeypatch
    ):
        wired()

        class Embedding(FakeEmbedding):
            def embed_query(self, question):
                raise AssertionError("must not embed when a vector is supplied")

        monkeypatch.setattr(composition_module, "get_embedding_driver", Embedding)

        chunks = retrieve_chunks(
            QUESTION,
            query_vector=[0.0] * 384,
            store=FakeStore(),  # type: ignore[arg-type]
        )

        assert [c.id for c in chunks] == [1, 3, 5]


class TestWiring:
    def test_the_audit_log_gets_the_gate_and_the_rerank_events(self, wired):
        path = wired()

        retrieve_chunks(QUESTION, store=FakeStore())  # type: ignore[arg-type]

        events = {
            e["action"]: e["data"]
            for e in map(json.loads, path.read_text().splitlines())
        }
        assert events["relevance_gate_checked"]["passed"] is True
        assert events["rerank_applied"]["reranker_model"] == "model-x"
        assert events["rerank_applied"]["accepted_count"] == 3

    def test_the_listwise_step_gets_the_model_and_the_limit_from_the_settings(
        self, wired, monkeypatch
    ):
        wired(LISTWISE_RERANK_ENABLED=True, LISTWISE_RERANK_MAX_CANDIDATES=7)
        seen: dict = {}

        def spy(question, chunks, driver, **kwargs):
            seen.update(driver=driver, **kwargs)
            return list(chunks)

        monkeypatch.setattr(composition_module, "listwise_rerank", spy)

        retrieve_chunks(QUESTION, store=FakeStore())  # type: ignore[arg-type]

        assert seen == {"driver": "llm", "max_candidates": 7}

    def test_the_listwise_model_is_not_even_looked_up_when_the_step_is_off(
        self, wired, monkeypatch
    ):
        wired(LISTWISE_RERANK_ENABLED=False)

        def forbidden():
            raise AssertionError("the language model must not be built")

        monkeypatch.setattr(composition_module, "get_answer_driver", forbidden)

        retrieve_chunks(QUESTION, store=FakeStore())  # type: ignore[arg-type]

    def test_the_caller_s_trace_is_filled_like_the_original(self, wired):
        wired()
        from models import RetrievalTrace

        trace = RetrievalTrace()
        retrieve_chunks(QUESTION, store=FakeStore(), trace=trace)  # type: ignore[arg-type]

        assert list(trace.stages) == [
            "vector",
            "vector_csls",
            "fulltext",
            "fused",
            "reranked",
            "listwise",
            "final",
        ]
        assert trace.notes["gate_passed"] is True


class FakeAnswerDriver:
    """Stands for the language model: records the prompt, replies with a fixed text."""

    def __init__(self, reply="THE ANSWER"):
        self.reply, self.calls = reply, []

    def generate(self, system_prompt, user_message, max_tokens=1024):
        self.calls.append((system_prompt, user_message, max_tokens))
        return self.reply


class TestWholeAnswer:
    """``query_knowledge_base`` with ``QUERY_ENGINE=v2`` answers through the QueryService."""

    def test_a_question_is_read_and_answered_by_the_new_pipeline(
        self, wired, monkeypatch
    ):
        wired()
        driver = FakeAnswerDriver()
        monkeypatch.setattr(composition_module, "get_answer_driver", lambda: driver)

        answer = retrieval_module.query_knowledge_base(
            QUESTION,
            store=FakeStore(),  # type: ignore[arg-type]
        )

        assert answer == "THE ANSWER"
        (_, user_message, _) = driver.calls[0]
        assert user_message.endswith(f"Question: {QUESTION}")

    def test_nothing_relevant_is_the_original_refusal_and_no_model_is_asked(
        self, wired, monkeypatch
    ):
        wired(RETRIEVAL_MIN_SCORE=0.99)
        driver = FakeAnswerDriver()
        monkeypatch.setattr(composition_module, "get_answer_driver", lambda: driver)

        answer = retrieval_module.query_knowledge_base(
            QUESTION,
            store=FakeStore(),  # type: ignore[arg-type]
        )

        assert answer == retrieval_module.NO_RESULTS_MESSAGE
        assert driver.calls == []

    def test_a_metadata_filter_is_refused_not_ignored(self, wired):
        wired()

        with pytest.raises(ValueError, match="QUERY_ENGINE=v2"):
            retrieval_module.query_knowledge_base(
                QUESTION,
                metadata_filter={"a": "b"},
                store=FakeStore(),  # type: ignore[arg-type]
            )

    def test_a_plain_strategy_names_the_profile_the_answer_is_read_with(
        self, wired, monkeypatch
    ):
        wired()  # the settings say hybrid
        driver = FakeAnswerDriver()
        monkeypatch.setattr(composition_module, "get_answer_driver", lambda: driver)

        retrieval_module.query_knowledge_base(
            QUESTION,
            strategy=VectorRetrievalStrategy(),
            store=FakeStore(),  # type: ignore[arg-type]
        )

        # the cosine cut (ids 1,3,5,7), no keyword/rerank: the vector profile was run
        user_message = driver.calls[0][1]
        assert user_message.count("Source:") == 4

    def test_a_strategy_with_overrides_is_still_refused(self, wired):
        wired()

        with pytest.raises(ValueError, match="QUERY_ENGINE=v2"):
            retrieval_module.query_knowledge_base(
                QUESTION,
                strategy=HybridRetrievalStrategy(period_filter=True),
                store=FakeStore(),  # type: ignore[arg-type]
            )


class TestComposition:
    """Which parts ``build_query_service`` puts together for the settings."""

    def test_with_the_router_off_every_question_is_read_and_none_is_answered_exactly(
        self, wired
    ):
        from query.decision import UnplannedDecider

        wired(QUERY_ROUTER=False)

        service = composition_module.build_query_service(retrieval_module.settings)

        assert isinstance(service._decider, UnplannedDecider)
        assert service._exact is None

    def test_with_the_router_on_a_planner_decides_and_exact_answers_are_possible(
        self, wired
    ):
        from query.answering import ExactAnswerer
        from query.decision import PlanningDecider

        wired(QUERY_ROUTER=True)

        service = composition_module.build_query_service(retrieval_module.settings)

        assert isinstance(service._decider, PlanningDecider)
        assert isinstance(service._exact, ExactAnswerer)

    @pytest.mark.parametrize("partial", [False, True])
    def test_the_answer_policy_comes_from_the_settings(self, wired, partial):
        wired(ANSWER_PARTIAL_COVERAGE=partial, EXPOSE_DOCUMENT_DATE=not partial)

        service = composition_module.build_query_service(retrieval_module.settings)

        policy = service._grounded._policy  # type: ignore[attr-defined]
        assert (policy.partial_coverage, policy.expose_document_date) == (
            partial,
            not partial,
        )
