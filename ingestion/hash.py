"""File hashing utilities for document deduplication and integrity tracking.

Provides streaming cryptographic hash calculation (SHA-256) so large files
(e.g. multi-hundred-page PDFs) can be hashed efficiently without loading their
entire raw byte content into memory at once.

Key exports:
    compute_file_hash -- Return the hex-encoded SHA-256 digest of a file.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

#: Chunk size for streaming file reads (64 KB).
_BUFFER_SIZE = 64 * 1024


def compute_file_hash(file_path: str | Path) -> str:
    """Calculate the SHA-256 checksum of a file on disk.

    Reads the target file in chunks of 64 KB to ensure minimal memory overhead
    even for very large documents.

    Args:
        file_path: Path to the target file.

    Returns:
        The hexadecimal SHA-256 digest string (64 characters).

    Raises:
        FileNotFoundError: If the file does not exist.
        IsADirectoryError: If the path points to a directory.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"Cannot hash a directory: {path}")

    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_BUFFER_SIZE):
            hasher.update(chunk)

    return hasher.hexdigest()
