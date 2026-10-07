"""The extraction runner with several workers: faster, and otherwise the same.

A worker count must change the speed and nothing else: the stored values, the report and
the progress are what the plain loop gives, the database is only touched from the main
thread, an unexpected error still stops the run, and memory does not grow with the corpus.
"""

import threading
from unittest.mock import MagicMock

import pytest
from test_metadata_runner import (  # type: ignore[import-not-found]
    FakeDocuments,
    _chunk,
    _key,
)

from metadata.evidence import EvidenceSelector
from metadata.runner import MetaExtractionRunner
from metadata.sources import Candidate, MetaSource, SourceResult
from models import Document
from models import MetaSource as Kind


def _documents(n: int) -> list[Document]:
    return [Document(f"{i:02d}" + "a" * 62, f"d{i}.docx") for i in range(n)]


def _marker(content_hash: str) -> str:
    return f"Court-{content_hash[:2]}"


class PerDocumentChunks:
    """Every document has one chunk that names it, so a result can be traced to it."""

    def __init__(self, on_read=None, skip=()):
        self._on_read = on_read
        self._skip = skip
        self.reads = 0

    def get_document_chunks(self, content_hash):
        self.reads += 1
        if self._on_read:
            self._on_read(self)
        if content_hash[:2] in self._skip:
            return []
        return [_chunk(0, f"Bíróság: {_marker(content_hash)} ítélete")]


class EchoSource(MetaSource):
    """Returns the court named in the chunk it is shown, quoting it."""

    kind = Kind.LLM
    needs_verification = True

    def __init__(self, before=None):
        self._before = before

    def supports(self, key):
        return True

    def extract(self, chunks, keys):
        if self._before:
            self._before(chunks[0].content)
        marker = chunks[0].content.split(": ")[1].split(" ")[0]
        return SourceResult(
            candidates=[
                Candidate("court", marker, evidence=marker, evidence_chunk_index=0)
            ]
        )


def _run(n, workers, source=None, chunks=None, on_progress=None, documents=None):
    docs = documents or FakeDocuments([_key("court")], _documents(n))
    runner = MetaExtractionRunner(
        docs,
        chunks or PerDocumentChunks(),
        EvidenceSelector(MagicMock(), per_key=2, max_chunks=6),
        [source or EchoSource()],
        on_progress=on_progress,
        workers=workers,
    )
    return docs, runner.run("court_decision")


class TestSameResult:
    @pytest.mark.parametrize("workers", [2, 4, 16])
    def test_the_values_and_the_report_do_not_depend_on_the_worker_count(self, workers):
        plain_docs, plain = _run(12, 1)
        docs, report = _run(12, workers)

        assert report == plain
        assert docs.values == plain_docs.values
        assert docs.statuses == plain_docs.statuses

    def test_every_value_belongs_to_the_document_it_was_found_in(self):
        """A mix-up between concurrent documents would show up here."""
        docs, _ = _run(20, 8)

        for (content_hash, key), values in docs.values.items():
            assert [v.value_text for v in values] == [_marker(content_hash)]

    def test_a_document_without_chunks_is_skipped_the_same_way(self):
        _, plain = _run(6, 1, chunks=PerDocumentChunks(skip=("02", "04")))
        _, report = _run(6, 3, chunks=PerDocumentChunks(skip=("02", "04")))

        assert report == plain and report.documents == 4

    def test_a_single_worker_is_the_plain_loop(self):
        threads = set()
        source = EchoSource(before=lambda _: threads.add(threading.current_thread()))

        _run(3, 1, source=source)

        assert threads == {threading.current_thread()}  # no pool was started


class TestConcurrency:
    def test_documents_really_are_processed_at_the_same_time(self):
        """Three extractions wait for each other: they only pass if three run at once."""
        barrier = threading.Barrier(3, timeout=5)
        source = EchoSource(before=lambda _: barrier.wait())

        _, report = _run(6, 3, source=source)

        assert report.documents == 6

    def test_the_database_is_only_touched_from_the_main_thread(self):
        main = threading.current_thread()
        touched: set[threading.Thread] = set()

        class Recording(FakeDocuments):
            def get_statuses(self, content_hash):
                touched.add(threading.current_thread())
                return super().get_statuses(content_hash)

            def replace_values(self, content_hash, key, values):
                touched.add(threading.current_thread())
                super().replace_values(content_hash, key, values)

            def set_status(self, status):
                touched.add(threading.current_thread())
                super().set_status(status)

        reads: set[threading.Thread] = set()
        chunks = PerDocumentChunks(
            on_read=lambda _: reads.add(threading.current_thread())
        )
        workers_seen: set[threading.Thread] = set()
        source = EchoSource(
            before=lambda _: workers_seen.add(threading.current_thread())
        )

        _run(
            10,
            4,
            source=source,
            chunks=chunks,
            documents=Recording([_key("court")], _documents(10)),
        )

        assert touched == {main} and reads == {main}
        assert (
            workers_seen and main not in workers_seen
        )  # the model work did go to workers

    def test_only_a_bounded_number_of_documents_is_in_flight(self):
        """Chunks are read lazily: reading all of the corpus up front would not scale."""
        progress: list[int] = []
        worst = 0

        def on_read(chunks):
            nonlocal worst
            worst = max(worst, chunks.reads - len(progress))

        _run(
            40,
            2,
            chunks=PerDocumentChunks(on_read=on_read),
            on_progress=lambda done, total: progress.append(done),
        )

        assert worst <= 2 * 2


class TestFailure:
    def test_an_unexpected_error_in_a_worker_stops_the_run_and_is_raised(self):
        def fail_on_document_three(content):
            if _marker("03") in content:
                raise RuntimeError("the API is down")

        docs = FakeDocuments([_key("court")], _documents(10))

        with pytest.raises(RuntimeError, match="the API is down"):
            _run(
                10, 3, source=EchoSource(before=fail_on_document_three), documents=docs
            )

        failed_hash = _documents(10)[3].content_hash
        assert (failed_hash, "court") not in docs.statuses  # nothing is recorded for it


class TestProgress:
    def test_progress_counts_every_document_once_up_to_the_total(self):
        calls: list[tuple[int, int]] = []

        _run(
            9,
            3,
            chunks=PerDocumentChunks(skip=("01",)),
            on_progress=lambda done, total: calls.append((done, total)),
        )

        assert [d for d, _ in calls] == list(range(1, 10))
        assert {t for _, t in calls} == {9}


def test_fewer_than_one_worker_is_refused():
    with pytest.raises(ValueError, match="at least 1"):
        MetaExtractionRunner(
            FakeDocuments([_key("court")]),
            PerDocumentChunks(),
            EvidenceSelector(MagicMock()),
            [EchoSource()],
            workers=0,
        )
