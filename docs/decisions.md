# Engineering Decisions & Bug Log

A dated, reverse-chronological log of what was tried, what broke, and why the current defaults in `README.md` are what they are. `README.md` describes the system as it stands today; this file is the running "why," kept out of the README so reference and history don't keep drifting into one document. Newest entries first. Each entry names the commit(s) it came from.

## 2026-09-29 — Switched to `EMBEDDING_DRIVER=openrouter`; made `RERANKER_DRIVER=cross_encoder` the default (`ca9e225`)

Ingesting the real-estate-law corpus with the local embedding driver measured ~17.8s/document (10 real documents, 177.76s total) — projected to **~2 days** of continuous CPU-bound embedding for the full ~10,000-document corpus. Looked for a faster alternative before committing to that.

**`EMBEDDING_DRIVER=openrouter` was added** rather than a native Gemini SDK integration, after discovering (live, via `GET /api/v1/embeddings/models`) that OpenRouter already exposes `google/gemini-embedding-001` — reusable with the *same* OpenRouter account/key this project's `LLM_DRIVER=openrouter` already uses, no new provider account needed. Verified against the real endpoint, not assumed:
- Native output is 3072d, but OpenRouter's `dimensions` request parameter truncates it — confirmed working down to 384, this project's existing `document_chunks` column width, so switching needed **no schema migration** (helped by the table having just been flushed for this restart anyway).
- Real cost: `1.65e-06` per 11-token call ≈ **$0.15/1M tokens** ≈ **~$5 for the full corpus** — cheap enough not to be a real constraint.
- Real speed on 10 real documents (word chunking, see below): **55.7s total ≈ 5.6s/doc**, ~3.2× faster than local, projecting to **~15-16 hours** for the full corpus instead of ~2 days.

**Two real bugs found via the same live re-ingestion test, not by inspection:**
1. The API hard-rejects a batch over 250 items: `HTTP 400`, *"batchSize value of 300 but the supported range is from 1 (inclusive) to 251 (exclusive)"*. `OpenRouterEmbeddingDriver.embed_batch()` now splits into sub-batches of at most 250 automatically. Without this, any document producing >250 chunks in one `embed_documents()` call failed outright (3 of the first 10 real documents did).
2. This driver has no real tokenizer (`count_tokens()`/`max_sequence_length()` stay `None`, same as `openai`), and `.env` still had `CHUNKING_STRATEGY=langchain` from earlier local-driver tuning. Without a real tokenizer, `langchain` silently measures `CHUNK_SIZE` in raw *characters*, not words — one real document went from an expected ~15 chunks to **407**. Fixed by setting `CHUNKING_STRATEGY=word` for this driver, which needs no token counting at all.

**A more consequential finding, discovered while sanity-checking the relevance gate against the new model**: a real, obviously relevant Hungarian query scored **0.73–0.76** raw cosine similarity against real ingested content, which sounds fine against `RETRIEVAL_MIN_SCORE=0.25` — until a deliberately *irrelevant* Hungarian query (about frying chicken) also scored **0.54–0.55**, comfortably above the same 0.25 threshold. The `_passes_relevance_gate()` check (`query/retrieval.py`) is vector-only by design (see the README's "What happens when there's no reliable source" section) — with this embedding model, that gate alone can no longer reliably distinguish relevant from irrelevant queries. The threshold was tuned for the local e5 model's similarity scale; nothing guarantees it transfers to a different model's scale, and this is the first time this project actually swapped `EMBEDDING_DRIVER` on a populated index to find out.

**Decided against raising `RETRIEVAL_MIN_SCORE` to compensate.** The vector gate is a cheap, coarse, single-number threshold — tightening it risks rejecting genuinely relevant chunks that simply don't score well under raw cosine similarity (different phrasing, topically-adjacent-but-correct content), trading false negatives for false positives on too little data (two anecdotal queries) to calibrate responsibly. Real recalibration is deferred to the golden-set/eval work already planned, where there'll be enough labeled examples to tune against instead of guessing.

**Instead, re-tested with `RERANKER_DRIVER=cross_encoder` enabled** (previously not the default — see the 2026-09-26 "Cross-encoder logit calibration" entry). The reranker correctly rejected the irrelevant chicken-frying query (all candidates scored below `RERANKER_MIN_SCORE=-2.0`, correctly returning no results) while keeping the relevant query's chunks (logits 3.6–4.2). This works because the cross-encoder scores `(question, chunk)` *jointly*, not via embedding distance — its relevance judgment isn't tied to any particular embedding model's similarity scale the way the vector gate's threshold is.

**This made `RERANKER_DRIVER=cross_encoder` the new default** (`config.py`, `.env.example`), reversing the original `none` default from the 2026-09-26 entry. That original reasoning ("hybrid fusion is already a reasonable ranking on its own, reranking is optional") held for a single, fixed local embedding model — it doesn't hold once the embedding driver is something users are expected to actually swap, since nothing else in the pipeline then reliably catches an irrelevant query. `none` remains a real, selectable option for comparison/eval (`make eval` vs. the old bare-RRF baseline), just no longer the production default.

## 2026-09-29 — `RtfExtractor`: plain text only, header detection deferred; a real striprtf whitespace quirk (`0f00f1d`)

Second extractor for the real-estate-law corpus, after `DocxExtractor`. Unlike DOCX, RTF has no equally cheap way to get both clean text *and* paragraph-level formatting (alignment) in one library call: `striprtf` (the standard choice for RTF-to-text) discards all formatting, so replicating DOCX's centered-paragraph header signal would require a hand-rolled RTF parser reading `\qc`/`\pard` control words directly. Deferred deliberately rather than built speculatively — as of this decision, 0 of 5743 downloaded corpus documents are RTF, all are DOCX. `RtfExtractor` therefore matches `PDFExtractor`'s level of detail (plain text, no `header_path`), not `DocxExtractor`'s.

**A real library quirk, found via a real (unmocked) test, not assumed**: `striprtf` sometimes swallows the whitespace immediately following a hex-escaped character (`\'e9`), merging it with the next word (`t\'e9 l` → `"té l"`, not `"t é l"`). This only matters for letter-spaced text — which happens to be exactly the convention real corpus documents use for centered section titles ("í t é l e t e t :", see the `DocxExtractor` entry above) — but since `RtfExtractor` doesn't do header detection at all yet, it's a cosmetic spacing slip in body text today, not something anything currently depends on. Documented in the extractor's own docstring and a dedicated regression test, in case it matters once header detection is eventually added for RTF.

Also resolved along the way: RTF files declare their own character encoding via a `\ansicpg<N>` control word in the header — the raw file bytes are decoded as latin-1 first (a 1:1 byte-to-codepoint mapping that never raises, since RTF's control words are pure ASCII and non-ASCII characters are hex-escaped rather than embedded as raw high bytes), then `striprtf` interprets the hex escapes using whichever codec `\ansicpg` declares (defaulting to `cp1250` — Central European — since this corpus is Hungarian, for the rare case it's missing).

## 2026-09-29 — `DocxExtractor`: alignment, not heading styles, is the section-title signal (`977bb00`)

Building the ingestion side for the real-estate-law corpus (`corpus/download_court_decisions.py`), starting with `.docx` since 5555/5555 downloaded so far are DOCX, none RTF. The obvious first guess — use Word's "Heading 1"/"Heading 2" paragraph styles, mirroring how `MarkdownExtractor` uses `#`/`##` — turned out to be wrong: inspecting several real corpus documents' raw XML during the earlier corpus-download work already showed no `w:pStyle` references at all in the document body, just a template's unused style *definitions*. Every paragraph is styled "Normal".

What the documents actually use for section titles ("ítélete", "Indokolás", "A Kúria mint felülvizsgálati bíróság") is **centered paragraph alignment** — confirmed empirically across 5 real documents (`python-docx`, checking `paragraph.alignment == WD_ALIGN_PARAGRAPH.CENTER`): centered paragraphs are rare (2-4 out of 100-400 per document) and were a section title in every single occurrence checked, never body text. A second candidate signal — letter-spacing ("Í T É L E T") — was rejected after the same check: some titles are plain, unspaced caps ("INDOKOLÁS", "ÍTÉLETET:"), so it would have missed real titles rather than just risking false positives.

Since these titles are flat (no "H1 > H2" nesting the way Markdown headings can nest), `DocxExtractor` reuses `MarkdownExtractor`'s section-index/header-breadcrumb *output shape* (`ChunkMetadata.header_path`, `page_number`-as-section-index) but with a much simpler builder — just "which centered title came most recently before this word," no header stack to maintain.

## 2026-09-29 — Chunk packaging: rejected `ProcessedChunk`/`ChunkPacker`, extracted a smaller fix (`64a3c75`)

Károly asked for a `ProcessedChunk` type (distinct from `Chunk`) and a `ChunkPacker`/`ChunkEnricher` component to own header-breadcrumb prefixing *and* token-budget enforcement, motivated by a real observation: `SplitOverflowStrategy.apply()` was reverse-engineering `chunk_document()`'s header-embedding format — stripping `enrich_chunk_content()`'s output back off via a string-prefix check — to re-split an already-packaged chunk without duplicating or double-counting its header. That coupling is real and worth fixing.

The two-part proposal was scoped down before implementing, for reasons that mirror the 2026-09-29 Clean Architecture entry below:
- **No `ProcessedChunk` type.** `Chunk` already *is* the packaged, embedding-ready shape — in `chunk_document()`, `Chunk.content` is already header-enriched. A second type with no new fields would just be `Chunk` under another name, working against the same day's minimal-model-set decision (see the Clean Architecture entry below, which introduced `models.py`).
- **No `ChunkPacker` owning token-budget enforcement.** That's already `ChunkOverflowStrategy`'s explicit, `.env`-selected job (`WarnOverflowStrategy` vs. `SplitOverflowStrategy`). Giving a second component that same authority would blur `WarnOverflowStrategy`'s whole point (warn, don't correct) and create two places that could disagree about whether a chunk fits.

What actually landed instead: `_strip_header_prefix()` in `ingestion/chunker.py`, written as `enrich_chunk_content()`'s explicit inverse and kept immediately next to it, plus `_package_chunk(content, metadata) -> Chunk` as the single place both `chunk_document()` and `SplitOverflowStrategy.apply()` go through to build a chunk — removing the implicit, unenforced format contract without adding a new type or a competing authority over the token budget.

## 2026-09-29 — Clean Architecture re-discussion: targeted fixes instead of a layered rewrite (`c91b35c`)

Károly proposed migrating to Clean Architecture (Entities/Use Cases/Interface Adapters/Frameworks, with a strict Dependency Rule) ahead of scaling up to the 10,000+ document legal corpus, explicitly as both a real concern and a hands-on learning goal. `AGENTS.md` already had a dated, considered rejection of DDD/layered ceremony for this project (2026-09-17) — the honest way to answer "should we revisit that" was to actually check the current code against the three concrete complaints (SOLID/SRP, OCP, DIP) rather than debate architecture styles in the abstract.

**What the code review actually found:**
- **SRP**: `agent.py`, `query/retrieval.py`, and `store.py` were already thin/focused — `store.py` in particular already *is* what the proposal called an "Interface Adapter/Repository," just not filed under that name. The one real hotspot: `ingestion/ingest.py`'s `add_document()` mixed real dedup/versioning business logic (skip-unchanged / alias-duplicate / replace-previous-version / insert-new, ~30 lines of branching) directly into what's supposed to be a thin orchestrator, and `add_directory()` had a near-duplicate of the same branching.
- **OCP**: already well served — the Strategy/Driver pattern runs across embedding, LLM, reranker, *and* retrieval strategy (4 independent hierarchies), plus the deliberately-different extractor dispatch. Nothing to extend here; the proposal's "can we extend this to extractors/vector-DBs too" question turned out to already be "yes, extractors already do."
- **DIP**: `ingest.py`/`retrieval.py` depend on a concrete `VectorStore` class, not an ABC. But there is exactly one backend (Postgres+pgvector) and no concrete plan for a second — a `VectorStoreRepository` ABC with one implementation would be the textbook premature abstraction this project's own top-level guidance warns against.

**The agreed compromise** (all three items delivered in commit `c91b35c`):
1. Extracted the dedup/versioning decision into a pure, unit-tested `_resolve_ingest_action()` (`ingestion/ingest.py`) — fixes the one real SRP hotspot, and as a bonus removes the near-duplicated branching between `add_document`/`add_directory`. See `AGENTS.md`'s "Orchestrator functions must stay thin" bullet.
2. Introduced typed `dataclass`es (`models.py`: `ChunkMetadata`, `Chunk`, `RetrievedChunk`) in place of the untyped chunk dict, motivated by the corpus's upcoming richer metadata (header paths, section breadcrumbs) making typo'd dict keys a real risk. Not a full Entity layer — no validation logic, no framework independence beyond what a plain dataclass already gives; Postgres JSONB storage unchanged (`to_dict()`/`from_dict()` are the only serialization boundary).
3. Deliberately did **not** introduce a `VectorStoreRepository` ABC — no second backend to justify it (see DIP finding above).

Explicitly NOT done, and why: no Entities/Use-Case/Interface-Adapter folder restructuring, no dependency-injection framework, no bounded contexts. The project's domain is still a linear pipeline with no complex business invariants to protect — the two real problems named above had small, targeted fixes, and manufacturing a bigger architectural exercise around them would have been solving a problem the codebase didn't actually have.

## 2026-09-29 — Corpus acquisition: reverse-engineered eakta.birosag.hu's search/download endpoints (`b157435`)

Built `corpus/download_court_decisions.py` to collect a large (10,000+ document), realistic real-estate-law corpus for the upcoming large-scale retrieval/answer-accuracy eval (profiled user questions, a golden set — see the RAG-accuracy-at-scale direction). No public API exists for the site's "Bírósági Határozatok Gyűjteménye" (anonymized court decisions), so the actual request shape was reverse-engineered from the page's embedded JavaScript (the "ENCO.Grid" component) and verified against live responses rather than assumed.

**Findings that shaped the script:**
- The search (`POST /AnonimizaltHatarozat/Search?Area=`) is a plain JSON API — no headless browser or HTML scraping needed, simpler than expected. Getting the request shape right took actual trial and error: the site's own JS builds `KeresoSzavak[]`/`KeresoSzoOperatorok[]` array params, and a wrong operator enum value (guessed as `"AND"`/`"OR"` before finding the real `KeresoSzoOperatorDropdown` options: `Osszes`/`Kifejezes`/`Pontos`) produces a generic, misleading "Hiba történt..." error with no indication of which field is wrong.
- The `HatarozatFajta` filter (Ítélet/Végzés) accepts exactly one value per search — getting both types requires two separate searches, deduplicated by `(court, case_number)` across all of them.
- **The "native format" download endpoint doesn't always return RTF** — it returns whichever format the source court originally filed in, RTF *or* DOCX, with the real format only knowable from the response's `Content-Type`/`Content-Disposition` headers, not assumable from the request. The script reads this back per-download rather than hardcoding an extension.
- Inspecting a real DOCX/RTF pair showed neither uses Word "Heading" styles for section titles (e.g. "ÍTÉLET", "INDOKOLÁS") — the actual convention is a **centered, letter-spaced paragraph** (`w:jc="center"` in DOCX, `\qc` in RTF), confirmed identically in both formats for the same section marker. This is a cheap, reliable, markup-level signal — unlike PDF, which would need coordinate/font-size heuristics for the same job — and is the planned basis for a future header-enrichment pass on this corpus, analogous to the Markdown extractor's `header_path` (see the 2026-09-27 entry above), once ingestion-side RTF/DOCX extractors exist.

**Resume-safety, added after being asked directly "what happens if this gets interrupted":**
- File writes are atomic (write to a `.part` temp file, then rename) — a crash/power-loss mid-write can never leave a truncated file that a later run would mistake for a complete download.
- `meta.csv` is append-only and a re-run only treats a `(court, case_number)` as done if its *latest* recorded status is `downloaded`/`already_downloaded` — a row left at `error` (e.g. from a transient network outage) is automatically retried on the next run instead of being skipped forever.

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
