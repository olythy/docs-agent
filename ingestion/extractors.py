"""Document extractor abstractions.

Defines the common interface (``Extractor``) for turning a source document
into the ``(full_text, word_page_map)`` pair :func:`ingestion.chunker.chunk_document`
expects, following the Strategy / Driver pattern described in AGENTS.md.

Unlike every other Strategy in this project (``EMBEDDING_DRIVER``,
``CHUNKING_STRATEGY``, ...), the active extractor is **not** chosen from
``settings`` — it's chosen by the file's extension, via :func:`get_extractor`:
    - ``.pdf``                → :class:`PDFExtractor`
    - ``.md`` / ``.markdown`` → :class:`MarkdownExtractor`

That's deliberate: which extractor applies is a fact about the file, not a
preference — there's nothing to configure.

Usage::

    from ingestion.extractors import get_extractor
    extractor = get_extractor(Path("notes.md"))
    extractor.validate(path)
    full_text, word_page_map = extractor.extract(path)
"""

from abc import ABC, abstractmethod
from pathlib import Path

from ingestion.pdf_loader import extract_document_text, extract_pages, is_scanned_pdf


class Extractor(ABC):
    """Abstract base class for all document-format extractors.

    Two-step contract, mirroring how ``ingestion.ingest.add_document()``
    already used PDF extraction before this abstraction existed: validate
    first (fail with a clear error before any embedding-driver/DB work
    happens), then extract the real content.
    """

    @abstractmethod
    def validate(self, file_path: Path) -> None:
        """Raise if ``file_path`` has no usable content.

        Args:
            file_path: Path to the source document.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file exists but has no extractable content
                (e.g. empty, or a scanned PDF with no text layer).
        """

    @abstractmethod
    def extract(self, file_path: Path, mode: str = "flat") -> tuple[str, list[int]]:
        """Return the whole document as one string, plus a per-word page/section map.

        Same contract :func:`ingestion.chunker.chunk_document` expects:
        ``word_page_map[i]`` identifies where ``full_text.split()[i]`` came
        from (1-based page number for PDFs; a header-based section index
        for Markdown, where there are no real pages) — always the same
        length as ``full_text.split()``.

        Args:
            file_path: Path to the source document.
            mode: Forwarded to :func:`ingestion.pdf_loader.extract_document_text`
                for :class:`PDFExtractor` (``"flat"`` or ``"blocks"``).
                Ignored by extractors whose format doesn't need it (e.g.
                :class:`MarkdownExtractor`, which already has its paragraph
                structure natively, with no coordinate-based mode to choose).

        Returns:
            A ``(full_text, word_page_map)`` tuple.
        """


class PDFExtractor(Extractor):
    """Wraps the existing, unchanged ``ingestion.pdf_loader`` functions.

    No PDF-parsing logic lives here — this class only adapts
    ``pdf_loader``'s functions to the :class:`Extractor` interface, reusing
    them exactly as ``ingestion.ingest.add_document()`` used to call them
    directly.
    """

    def validate(self, file_path: Path) -> None:
        pages = extract_pages(file_path)
        if not pages:
            raise ValueError(
                f"'{file_path.name}' has no pages. The file may be empty or corrupted."
            )
        if is_scanned_pdf(pages):
            raise ValueError(
                f"'{file_path.name}' appears to be a scanned PDF with no text layer. "
                "OCR support is not implemented in this version."
            )

    def extract(self, file_path: Path, mode: str = "flat") -> tuple[str, list[int]]:
        return extract_document_text(file_path, mode=mode)


class MarkdownExtractor(Extractor):
    """Reads a Markdown file directly — no coordinate-based heuristics needed.

    Unlike a PDF, Markdown already marks its own paragraph breaks (blank
    lines) and structure (ATX headers, ``# `` through ``###### ``) directly
    in the text — there's nothing to infer, so ``extract()`` just reads the
    file as-is. The ``mode`` parameter (PDF's flat-vs-blocks choice) doesn't
    apply here and is ignored, the same way :class:`ingestion.chunker.WordChunkingStrategy`
    ignores the ``driver`` parameter it's still handed for interface
    uniformity.
    """

    def validate(self, file_path: Path) -> None:
        if not file_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {file_path}")
        if not file_path.read_text(encoding="utf-8").strip():
            raise ValueError(f"'{file_path.name}' is empty.")

    def extract(self, file_path: Path, mode: str = "flat") -> tuple[str, list[int]]:
        full_text = file_path.read_text(encoding="utf-8")
        word_page_map = _markdown_section_map(full_text)
        return full_text, word_page_map


def _markdown_section_map(full_text: str) -> list[int]:
    """Return a 1-based "section index" per word in ``full_text.split()``.

    There's no real "page" in a Markdown file, so this reuses the same
    word_page_map mechanism chunk_document() already relies on for PDFs,
    with each ATX header (``#`` through ``######``) starting a new section
    — a location marker meaningful enough for citations ("this came from
    section 3") without needing a whole separate metadata shape.

    Lines inside a fenced code block (delimited by a line starting with
    ` ``` `) are never treated as headers — without this, a documentation
    file with a shell/Python example containing a ``#`` comment would be
    miscounted as starting a new section every time the example does.

    Args:
        full_text: The whole Markdown document.

    Returns:
        One section index per word — same length as ``full_text.split()``.
    """
    word_page_map: list[int] = []
    section = 1
    in_code_fence = False

    for line in full_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
        elif not in_code_fence and stripped.startswith("#"):
            section += 1
        word_page_map.extend([section] * len(line.split()))

    return word_page_map


def get_extractor(file_path: Path) -> Extractor:
    """Factory function: return the extractor matching ``file_path``'s extension.

    Args:
        file_path: Path to the source document.

    Returns:
        An :class:`Extractor` instance for that file's format.

    Raises:
        ValueError: If the file's extension isn't a supported format.
    """
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return PDFExtractor()
    if suffix in (".md", ".markdown"):
        return MarkdownExtractor()

    raise ValueError(
        f"Unsupported document type: '{suffix}'. Supported extensions are: "
        ".pdf, .md, .markdown."
    )
