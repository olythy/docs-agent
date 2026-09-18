# Docs Agent Sample Notes

This is a small, non-sensitive Markdown fixture used to exercise the
document-level chunking pipeline for Markdown files (the same way
`sample_1.pdf`/`sample_2.pdf` exercise it for PDFs) — not a real document.

## Project Overview

Docs Agent is a personal RAG pipeline: PDF and Markdown documents are
extracted, split into overlapping chunks, embedded, and stored in a
Postgres database with the pgvector extension. Questions are answered by
retrieving the most similar chunks and asking a language model to answer
strictly from that retrieved context.

## Example Configuration

Below is a shell example showing how to run the test suite. The comment on
the second line starts with a `#`, just like a Markdown header — it must
not be mistaken for one, since it sits inside a fenced code block.

```bash
# Run the full test suite
AGENT_ENV=test uv run pytest -v
```

## Chunking Strategies

Two chunking strategies are available: a simple word-count sliding window,
and a paragraph-and-sentence-aware splitter built on
`langchain_text_splitters`. The latter tends to keep related sentences
together more often than a fixed-size window does, at the cost of a small
additional dependency.
