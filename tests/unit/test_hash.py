"""Unit tests for ingestion.hash module."""

import hashlib
from pathlib import Path

import pytest

from ingestion.hash import compute_file_hash


def test_compute_file_hash_returns_correct_sha256(tmp_path: Path):
    f = tmp_path / "hello.txt"
    content = b"Hello, Antigravity RAG!"
    f.write_bytes(content)

    expected = hashlib.sha256(content).hexdigest()
    assert compute_file_hash(f) == expected


def test_compute_file_hash_raises_on_missing_file():
    with pytest.raises(FileNotFoundError, match="File not found"):
        compute_file_hash("/path/to/nonexistent/file/12345.xyz")


def test_compute_file_hash_raises_on_directory(tmp_path: Path):
    d = tmp_path / "somedir"
    d.mkdir()

    with pytest.raises(IsADirectoryError, match="Cannot hash a directory"):
        compute_file_hash(d)
