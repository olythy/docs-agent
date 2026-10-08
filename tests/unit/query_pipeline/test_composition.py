"""How the services are put together from the settings and the drivers.

The scenarios (``test_retrieval_characterization.py``) prove what the chain does; these
cover what they cannot: that the caller's overrides arrive, that the composition root wires
the observers, the listwise reranker, the planner and the answer policy as the settings say.
"""

import json
from types import SimpleNamespace

import pytest
from test_retrieval_characterization import (  # type: ignore[import-not-found]
    FakeCrossEncoder,
    FakeEmbedding,
    FakeListwise,
    FakeStore,
)

import query.composition as composition_module
from logger import EventLogger
from query.answering import ExactAnswerer
from query.composition import build_query_service, build_retrieval_service
from query.decision import PlanningDecider, Scope
from query.outcome import NO_RESULTS_MESSAGE, Answerable
from query.profiles import DEFAULT_PROFILE
from query.service import RetrievalRequest

QUESTION = "What about the costs of the proceedings?"


@pytest.fixture
def wired(monkeypatch, settings_override, tmp_path):
    """Patch the drivers the way the scenarios do; returns a function that sets the
    settings and gives back them and the audit log path."""

    def wire(**settings):
        config = settings_override(
            **{
                "RERANKER_DRIVER": "cross_encoder",
                "RERANKER_MODEL": "model-x",
                "RETRIEVAL_TOP_K": 4,
                "RETRIEVAL_CANDIDATE_POOL_SIZE": 6,
                "RETRIEVAL_MIN_SCORE": 0.25,
                "RERANKER_MIN_SCORE": 0.5,
                **settings,
            }
        )
        monkeypatch.setattr(composition_module, "get_embedding_driver", FakeEmbedding)
        monkeypatch.setattr(
            composition_module,
            "get_reranker_driver",
            lambda *a, **k: FakeCrossEncoder(),
        )
        monkeypatch.setattr(composition_module, "get_answer_driver", lambda: "llm")
        monkeypatch.setattr(composition_module, "listwise_rerank", FakeListwise())
        path = tmp_path / "log.jsonl"
        monkeypatch.setattr(composition_module, "get_logger", lambda: EventLogger(path))
        return SimpleNamespace(settings=config, log=path)

    return wire


def retrieve(env, question=QUESTION, profile=None, **request):
    """Run the retrieval service built from the wired settings; the chunks, or ``[]``."""
    result = build_retrieval_service(env.settings).retrieve(
        RetrievalRequest(question, profile=profile or DEFAULT_PROFILE, **request),
        FakeStore(),  # type: ignore[arg-type]
    )
    return list(result.outcome.chunks) if isinstance(result.outcome, Answerable) else []


class TestProfiles:
    def test_an_unknown_profile_is_an_error(self, wired):
        env = wired()

        with pytest.raises(ValueError, match="Unknown retrieval profile"):
            retrieve(env, profile="nope")

    def test_the_default_profile_applies_the_cross_encoder_threshold(self, wired):
        env = wired()

        assert [c.id for c in retrieve(env)] == [1, 3, 5]


class TestOverrides:
    def test_top_k_and_min_score_reach_the_pipeline(self, wired):
        env = wired()

        assert [c.id for c in retrieve(env, top_k=2)] == [1, 3]
        assert retrieve(env, min_score=0.95) == []

    def test_a_precomputed_query_vector_means_no_embedding_call(
        self, wired, monkeypatch
    ):
        env = wired()

        class Embedding(FakeEmbedding):
            def embed_query(self, question):
                raise AssertionError("must not embed when a vector is supplied")

        monkeypatch.setattr(composition_module, "get_embedding_driver", Embedding)

        assert [c.id for c in retrieve(env, query_vector=[0.0] * 384)] == [1, 3, 5]


class TestWiring:
    def test_the_audit_log_gets_the_gate_and_the_rerank_events(self, wired):
        env = wired()

        retrieve(env)

        events = {
            e["action"]: e["data"]
            for e in map(json.loads, env.log.read_text().splitlines())
        }
        assert events["relevance_gate_checked"]["passed"] is True
        assert events["rerank_applied"]["reranker_model"] == "model-x"
        assert events["rerank_applied"]["accepted_count"] == 3

    def test_the_listwise_step_gets_the_model_and_the_limit_from_the_settings(
        self, wired, monkeypatch
    ):
        env = wired(LISTWISE_RERANK_ENABLED=True, LISTWISE_RERANK_MAX_CANDIDATES=7)
        seen: dict = {}

        def spy(question, chunks, driver, **kwargs):
            seen.update(driver=driver, **kwargs)
            return list(chunks)

        monkeypatch.setattr(composition_module, "listwise_rerank", spy)

        retrieve(env)

        assert seen == {"driver": "llm", "max_candidates": 7}

    def test_the_listwise_model_is_not_even_looked_up_when_the_step_is_off(
        self, wired, monkeypatch
    ):
        env = wired(LISTWISE_RERANK_ENABLED=False)

        def forbidden():
            raise AssertionError("the language model must not be built")

        monkeypatch.setattr(composition_module, "get_answer_driver", forbidden)

        retrieve(env)

    def test_the_reranker_is_the_one_the_settings_name(self, wired, monkeypatch):
        env = wired(RERANKER_DRIVER="none")
        asked = []
        monkeypatch.setattr(
            composition_module,
            "get_reranker_driver",
            lambda name=None: asked.append(name) or FakeCrossEncoder(),
        )

        build_retrieval_service(env.settings)

        assert asked == ["none"]


class FakeAnswerDriver:
    """Stands for the language model: records the prompt, replies with a fixed text."""

    def __init__(self, reply="THE ANSWER"):
        self.reply, self.calls = reply, []

    def generate(self, system_prompt, user_message, max_tokens=1024):
        self.calls.append((system_prompt, user_message, max_tokens))
        return self.reply


class TestQueryService:
    def test_it_has_a_planner_and_an_exact_answerer(self, wired):
        env = wired()

        service = build_query_service(env.settings)

        assert isinstance(service._decider, PlanningDecider)
        assert isinstance(service._exact, ExactAnswerer)

    @pytest.mark.parametrize("expose", [False, True])
    def test_the_answer_policy_comes_from_the_settings(self, wired, expose):
        env = wired(EXPOSE_DOCUMENT_DATE=expose)

        service = build_query_service(env.settings)

        assert service._grounded._policy.expose_document_date is expose  # type: ignore[attr-defined]

    def test_a_question_in_a_fixed_scope_is_read_and_answered(self, wired, monkeypatch):
        env = wired()
        driver = FakeAnswerDriver()
        monkeypatch.setattr(composition_module, "get_answer_driver", lambda: driver)

        answer = build_query_service(env.settings).answer(
            QUESTION,
            scope=Scope(),
            store=FakeStore(),  # type: ignore[arg-type]
        )

        assert answer.text == "THE ANSWER"
        assert driver.calls[0][1].endswith(f"Question: {QUESTION}")

    def test_nothing_relevant_is_the_refusal_and_no_model_is_asked(
        self, wired, monkeypatch
    ):
        env = wired(RETRIEVAL_MIN_SCORE=0.99)
        driver = FakeAnswerDriver()
        monkeypatch.setattr(composition_module, "get_answer_driver", lambda: driver)

        answer = build_query_service(env.settings).answer(
            QUESTION,
            scope=Scope(),
            store=FakeStore(),  # type: ignore[arg-type]
        )

        assert answer.text == NO_RESULTS_MESSAGE
        assert driver.calls == []
