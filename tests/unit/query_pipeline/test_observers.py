"""The observers: what is logged for a step, and that the old trace stays unchanged."""

import json
import logging

from logger import EventLogger
from models import ChunkMetadata, RetrievedChunk
from query.answering import GroundedAnswer
from query.facts import QueryFacts
from query.observers import (
    AnswerAuditObserver,
    AuditLogObserver,
    CompositeObserver,
    ProgressLogObserver,
)
from query.retrieval.context import RetrievalContext
from query.retrieval.step import Continue, RetrievalStep
from query.retrieval.steps.gates import RelevanceGateStep


def chunk(chunk_id: int, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        content="",
        metadata=ChunkMetadata(source_file="a.docx", page_number=None, chunk_index=0),
        score=score,
    )


def context(*scores: float) -> RetrievalContext:
    return RetrievalContext(
        facts=QueryFacts("the question"),
        dense_pool=tuple(chunk(i, s) for i, s in enumerate(scores, start=1)),
    )


def events(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


class TestAuditLog:
    def test_a_passed_gate_writes_the_same_event_as_before(self, tmp_path):
        path = tmp_path / "log.jsonl"
        gate = RelevanceGateStep(depth=2, min_score=0.5)
        before = context(0.9, 0.1, 0.1)

        AuditLogObserver(EventLogger(path)).on_step(gate, before, gate.run(before), 0.0)

        (event,) = events(path)
        assert event["action"] == "relevance_gate_checked"
        assert event["data"] == {
            "question": "the question",
            "passed": True,
            "top_score": 0.9,
            "top_k": 2,
            "min_score": 0.5,
            "candidate_count": 3,
        }

    def test_a_failed_gate_is_logged_too(self, tmp_path):
        path = tmp_path / "log.jsonl"
        gate = RelevanceGateStep(depth=2, min_score=0.5)
        before = context(0.2)

        AuditLogObserver(EventLogger(path)).on_step(gate, before, gate.run(before), 0.0)

        assert events(path)[0]["data"]["passed"] is False

    def test_a_step_without_an_event_writes_nothing(self, tmp_path):
        path = tmp_path / "log.jsonl"

        class Plain(RetrievalStep):
            name = "dense_search"

            def run(self, context):
                return Continue(context)

        before = context(0.9)
        AuditLogObserver(EventLogger(path)).on_step(
            Plain(), before, Continue(before), 0.0
        )

        assert not path.exists() or path.read_text() == ""


class TestProgress:
    def test_it_logs_a_line_per_step_with_the_sizes(self, caplog):
        gate = RelevanceGateStep(depth=2, min_score=0.5)
        before = context(0.9)

        with caplog.at_level(logging.INFO, logger="query.progress"):
            ProgressLogObserver().on_step(gate, before, gate.run(before), 0.012)

        assert "relevance_gate" in caplog.text

    def test_a_refusal_is_logged_with_its_reason(self, caplog):
        gate = RelevanceGateStep(depth=2, min_score=0.5)
        before = context(0.1)

        with caplog.at_level(logging.INFO, logger="query.progress"):
            ProgressLogObserver().on_step(gate, before, gate.run(before), 0.0)

        assert "declined (not_relevant)" in caplog.text


class TestComposite:
    def test_it_forwards_to_every_observer_in_order(self):
        seen: list[str] = []

        class Tag:
            def __init__(self, tag):
                self.tag = tag

            def on_step(self, step, before, result, seconds):
                seen.append(self.tag)

        gate = RelevanceGateStep(depth=1, min_score=0.5)
        before = context(0.9)

        CompositeObserver([Tag("a"), Tag("b")]).on_step(
            gate, before, gate.run(before), 0.0
        )

        assert seen == ["a", "b"]


class TestAnswerAudit:
    def test_one_event_per_answer_with_what_the_old_path_wrote(self, tmp_path):
        path = tmp_path / "log.jsonl"
        observer = AnswerAuditObserver(EventLogger(path), "vertex", "gemini-x")

        observer.on_answer(
            "Mi volt?",
            (chunk(1, 0.5), chunk(2, 0.4)),
            GroundedAnswer("An answer.", refused=False),
            2.34567,
        )

        (event,) = events(path)
        assert event["action"] == "answer_generated"
        assert event["data"] == {
            "question": "Mi volt?",
            "llm_driver": "vertex",
            "llm_model": "gemini-x",
            "chunk_count": 2,
            "latency_seconds": 2.346,
            "refused": False,
        }

    def test_the_models_own_refusal_is_marked(self, tmp_path):
        path = tmp_path / "log.jsonl"

        AnswerAuditObserver(EventLogger(path), "d", "m").on_answer(
            "q", (), GroundedAnswer("I could not find ...", refused=True), 0.1
        )

        assert events(path)[0]["data"]["refused"] is True
