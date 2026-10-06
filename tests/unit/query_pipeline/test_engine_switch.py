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
from query.retrieval import VectorRetrievalStrategy, retrieve_chunks

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

    def test_a_strategy_object_has_no_meaning_for_the_new_engine(self, wired):
        wired()

        with pytest.raises(ValueError, match="strategy object"):
            retrieve_chunks(
                QUESTION,
                strategy=VectorRetrievalStrategy(),
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
