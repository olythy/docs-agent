# Engineering Decisions & Bug Log

A dated, reverse-chronological log of what was tried, what broke, and why the current defaults in `README.md` are what they are. `README.md` describes the system as it stands today; this file is the running "why," kept out of the README so reference and history don't keep drifting into one document. Newest entries first. Each entry names the commit(s) it came from.

## 2026-09-27 — Pipeline refinements & idempotency (`73a1e03`)

- Added word overlap in `SplitOverflowStrategy` when dividing oversized chunks, preventing sentence mutilation at boundary lines.
- Added `VectorStore.delete_chunks_from_source()`, called during `add_document(..., force=True)`, so re-indexing cleanly replaces existing chunks instead of accumulating duplicates.
- Enhanced `VectorStore(conn=...)` with context-manager connection lifecycle, letting callers reuse a single Postgres connection across sequential vector and keyword searches.

## 2026-09-27 — Multilingual evaluation suite & gold labels (`d6fd649`)

- Added a Hungarian enterprise IT policy corpus (`tests/data/sample_hu.md`).
- Expanded the eval benchmark to 25 bilingual queries (19 answerable, 6 unanswerable) with passage-level `expected_text_contains` gold labels.
- Evaluator tracks Passage Hit@1, Recall@k, MRR, and fallback accuracy, broken down by language (EN/HU).

## 2026-09-27 — Hierarchical heading enrichment & metadata filtering (`e88687c`)

- Markdown extractor now traces `# H1 > ## H2` heading paths outside code fences and prepends contextual breadcrumbs to chunks, storing `header_path` in chunk JSONB metadata.
- Added Postgres JSONB containment filtering (`metadata_filter` via `@>`) across the whole pipeline: `VectorStore`, `retrieve_chunks`, `agent.py`, and `mcp_server.py`.

## 2026-09-27 — Cross-encoder logit calibration & early rejection (`c4ef97c`)

Calibrated `RERANKER_MIN_SCORE=-2.0` on the cross-encoder logit scale (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`): relevant chunks empirically score +2 to +6, clearly irrelevant ones -3.5 to -9, giving a natural threshold in between.

**Caveat, worth being explicit about:** `-2.0` was calibrated on the same 6 unanswerable eval questions that the resulting `Fallback = 1.00` number (README, "Retrieval") is then measured against — that's training-set accuracy, not a demonstrated generalization to unseen questions. Validating it properly needs a larger, independent eval set that wasn't used for tuning.

## 2026-09-27 — Asymmetric embedding retrieval (`14a0d85`)

Upgraded the default embedding model to `intfloat/multilingual-e5-small` and implemented asymmetric embedding prefixes: `"passage: "` for indexed chunks (`embed_documents()`), `"query: "` for questions (`embed_query()`) — matches the model's retrieval-optimized training, measurably improves ranking vs. symmetric embedding.

## 2026-09-26 — Structured observability & FTS query sanitization (`fb9c4c1`)

- Added a dedicated JSONL structured audit logger (`logger.py` → `logs/log.jsonl`) tracking query lifecycles, RRF fusions, reranking thresholds, and metadata filtering.
- Added bilingual (Hungarian & English) stopword removal for full-text queries, while preserving short technical acronyms (`AI`, `UI`, `DB`, `CI`, `CD`, `RAG`, `SQL`, `LLM`) that a generic stopword list would otherwise strip.

## 2026-09-26 — Chunking & token budgeting defaults (`a1e6857`)

Made `CHUNK_OVERFLOW_STRATEGY=split` the default (previously `warn`) and tuned `CHUNK_SIZE=250`/`CHUNK_OVERLAP=30` words to fit comfortably within the 512-token limit of multilingual embedding models even under dense tokenization. See the 2026-09-17 entry below for the empirical finding that motivated this.

## 2026-09-26 — Model instantiation & factory caching (`fa01c75`)

Added `@lru_cache` across all driver factories (`get_embedding_driver`, `get_reranker_driver`, `get_answer_driver`) to prevent redundant, heavy Transformer model re-initialization and GPU/RAM churn during high-frequency queries and tests.

*(The six entries above were all part of the same pass, prompted by a structured Information Retrieval / ML-focused code review.)*

## 2026-09-19 — MCP server: five real bugs found integrating with a real host (`fb5af0a`)

Found by testing `mcp_server.py` against a real MCP client (Claude Desktop), not by inspection or reading the SDK's docs alone:

1. `ingestion/ingest.py`/`query/retrieval.py` used to report progress via `print()`. Over MCP's stdio transport, stdout is reserved for the JSON-RPC protocol — a stray `print()` line landed on that channel mid-call and broke a real client's message parsing (`Failed to parse JSONRPC message from server`), confirmed by spawning `mcp_server.py` as a real subprocess and calling both tools over the actual protocol. Fixed by switching both modules to Python's `logging` module (stderr by default, invisible to the protocol stream) — `agent.py`/`scripts/eval_cli.py`/the `make add-document`/`query` targets configure a bare `logging.basicConfig(format="%(message)s")` so their own CLI output looks exactly as before.
2. `uv run mcp install`'s own generated launch command (`uv run --with "mcp[cli]==X.Y.Z" mcp run <path>`) is built for a standalone, dependency-free single-file script — the SDK's own docs say so explicitly. `mcp_server.py` isn't that: it imports the whole project (`psycopg2`, `openai`, `sentence-transformers`, ...). Registering it as-is and launching from Claude Desktop failed immediately with **"Server disconnected"**; reproduced directly by spawning the exact generated command from an unrelated directory with a clean environment (no inherited venv, matching how Claude Desktop actually spawns it): `ModuleNotFoundError: No module named 'psycopg2'`. Fixed with `uv run --project <docs-agent dir> mcp_server.py` instead, which resolves against this project's own environment regardless of the caller's working directory. `make mcp-install` automates this registration and rewrites the generated config entry with the `--project` flag and required environment variables automatically.
3. **`add_document()` had no protection against re-ingesting the same file** — asking Claude Desktop to add a document already in the knowledge base silently duplicated its chunks. With a small corpus and a fixed `RETRIEVAL_TOP_K`, duplicate copies of the same chunks crowded out other, genuinely relevant unique chunks from a real query's top-k results — directly degrading answer quality, not just cleanliness. `add_document()` now raises `ValueError` if `VectorStore.has_chunks_from_source()` says the file's already present, with an explicit `force=True` escape hatch for genuinely wanting a duplicate.
4. When forced to answer from the knowledge base alone (via `/my-docs`), the model correctly identified that a "Technology Stack" list was retrieved incompletely — a bullet point split across two adjacent chunks, and the query only surfaced one of them. Root cause, confirmed by inspecting `retrieve_chunks()`'s actual output: it was bug 3 above — duplicate chunks were occupying the other top-k slots that should have gone to the chunk with the rest of the list.
5. Re-adding an already-present document (bug 3's guard, working as designed) surfaced with a **completely blank error message** in Claude Desktop. Cause: `mcp_server.py`'s `add_document` let the plain `ValueError` propagate, and the MCP SDK treats any exception besides its own `ToolError`/`MCPError` as an unexpected crash — deliberately hiding the exception's text from the client. Fixed by catching `ValueError`/`FileNotFoundError` (the specific, "the model could retry with different arguments" cases) and re-raising as `ToolError`, whose message does reach the client.

## 2026-09-18 — Hybrid search: full-text query was silently ANDing every word of the question (`b6b3d34`)

Found while building the retrieval eval script, not by inspection: `search_fulltext()` originally passed the raw question straight into `websearch_to_tsquery('simple', question)`. The `simple` text-search configuration has no stopword list (chosen deliberately, since the corpus mixes Hungarian and English), so every word of the question — including grammar words — became a **mandatory** term (`websearch_to_tsquery` ANDs bare words together). A real chunk almost never contains a question's grammar words verbatim, so keyword search was silently returning **zero results for nearly every natural-language question**, undetected until the eval script's real numbers showed `0 keyword result(s)` on every single run.

Fixed by OR-joining the question's words (`" or ".join(query_text.split())`) before passing them to `websearch_to_tsquery`, so a chunk matching *any* of the question's content words now contributes to RRF fusion. Covered by a unit test (asserts the OR-joined string reaches the query) and a DB test (a real sentence full of grammar words that would have failed pre-fix).

## 2026-09-18 — Chunking: paragraphs spanning a page boundary were silently truncated (`fabbabe`)

Chunking used to run per page (`chunk_pages()`), with no overlap between pages — a paragraph that happened to span a page break was silently split into two truncated, unrelated chunks. `ingest.py` now concatenates the whole document first (`pdf_loader.extract_document_text()`) and chunks that (`chunker.chunk_document()`), so there's no page loop left to truncate anything at a page boundary. `chunk_pages()` itself is unchanged and still used by `eval_cli.py extract`-adjacent tooling and its own tests.

Since a chunk's words can now come from more than one page, `page_number` in its metadata is assigned by majority vote (whichever page contributed the most words) rather than a page range.

## 2026-09-17 — Chunking: `split` shown to catch what `warn` misses (`2404c3e`)

Verified against a real document, with an earlier embedding model whose sequence limit was 128 tokens (not the current default model's 512): with `CHUNK_SIZE=50` words and `WORDS_PER_TOKEN=0.4`, the `warn` heuristic estimated `50 / 0.4 = 125` tokens — just under the 128-token limit, so it **didn't warn at all**. Yet 8 of the 21 actual chunks (38%) really did exceed 128 tokens (up to 198), because real token count varies a lot per chunk's content, not just per configured size. `split` caught and corrected all 8 (21 → 29 chunks), each landing between 53 and 127 tokens — no near-empty stragglers, and afterward zero chunks exceeded the limit.

This is the finding that later motivated making `split` the default (see the 2026-09-26 chunking-defaults entry above).
