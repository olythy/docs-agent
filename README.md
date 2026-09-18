# docs-agent

A learning project demonstrating a **RAG-based AI Agent** with tool-calling, built on top of:

- 🗄️ **Supabase** (PostgreSQL + pgvector) — vector storage
- 🤖 **OpenRouter / OpenAI** — pluggable LLM driver for grounded answer generation
- 🔤 **Sentence-Transformers** — free, offline, multilingual embeddings (default)
- 📄 **pdfplumber** — PDF text extraction

## What Makes It an Agent (Not Just a RAG Pipeline)?

Unlike a static RAG pipeline (query → embed → retrieve → answer), this project is designed around **OpenAI-style function-calling**, where an LLM decides — on its own — which of two tools to invoke based on user intent:

| Tool | Triggered when... |
|---|---|
| `add_document(file_path)` | User wants to ingest a new document |
| `query_knowledge_base(question)` | User wants to ask a question |

`agent.py` wires this up: both tools are described to the LLM as OpenAI-style `tools=[...]` function schemas, and a single free-form message runs the standard tool-calling loop — the model decides whether to call `add_document`, `query_knowledge_base`, both, or neither. Try it interactively with `uv run python agent.py`, or call `agent.run_agent("...")` directly. `AnswerDriver` (the same driver `query_knowledge_base` uses for answer generation) exposes a public `get_client()`/`model` for this — the agent loop needs the raw, tools-capable chat client, not the RAG-specific `answer()` method with its fixed prompt shape.

**Caveat:** tool-calling support is model-dependent, and `LLM_MODEL`'s default (`openrouter/free`, which auto-routes to *some* available free model) isn't guaranteed to support it — pick a model explicitly known to support tools if `agent.py` doesn't behave as expected.

## Architecture

```
.
├── agent.py                 # Function-calling loop: LLM picks add_document vs query_knowledge_base
├── mcp_server.py            # MCP server (stdio): search_knowledge_base + add_document, for Claude Desktop etc.
├── config.py               # Centralized Settings (env + defaults)
├── db.py                   # Postgres connection factory — nothing else
├── store.py                # VectorStore: all document_chunks persistence (save/search)
├── drivers/
│   ├── embedding.py         # EmbeddingDriver strategy: local (sentence-transformers) vs openai
│   ├── llm.py                # AnswerDriver strategy: openrouter vs openai
│   └── reranker.py           # RerankerDriver strategy: none vs cross_encoder
├── ingestion/
│   ├── extractors.py         # Extractor strategy: PDF vs Markdown, chosen by file extension
│   ├── pdf_loader.py         # PDF text extraction (pdfplumber), flat/blocks modes
│   ├── chunker.py            # Chunking strategies (word/langchain) + overflow correction
│   └── ingest.py             # add_document orchestration
├── query/
│   ├── retrieval.py          # query_knowledge_base: hybrid retrieval + answer generation
│   └── hybrid.py             # reciprocal_rank_fusion: pure RRF fusion logic
├── migrations/              # Python migrations (Laravel-artisan-style runner)
│   ├── base.py                # Migration ABC: up()/down() run raw SQL, no ORM
│   ├── 0001_create_document_chunks_table.py
│   └── 0002_add_fulltext_search.py
├── scripts/
│   ├── migrate.py            # Migration runner: uv run python scripts/migrate.py [subcommand]
│   ├── make_migration.py     # Scaffold a new migration file
│   ├── db_flush.py           # Truncate document_chunks
│   ├── extract_text.py       # PDF extraction diagnostic CLI
│   ├── inspect_chunks.py     # Chunking diagnostic CLI: full strategy comparison matrix, with bars
│   ├── evaluate_retrieval.py # Retrieval-quality eval: vector-only vs hybrid+rerank
│   └── fix_mcp_install.py    # Patches `mcp install`'s generated Claude Desktop config — see "MCP Server"
├── docker-compose.yml       # Local Postgres+pgvector (dev + test databases)
├── docker/
│   └── init-test-db.sql      # Creates the "docs_agent_test" database on first startup
├── pyproject.toml           # Project metadata, dependencies, pytest config
├── uv.lock                  # Locked, reproducible dependency versions
├── .env.example             # Environment variable template
└── .env.test.example        # .env.test template — see AGENT_ENV below
```

## Setup

### 1. Clone and sync the environment

```bash
git clone <repo-url>
cd docs-agent
uv sync
```

[`uv`](https://docs.astral.sh/uv/) reads `.python-version` (3.12) and `pyproject.toml`/`uv.lock`, installs the pinned Python if needed, creates `.venv`, and installs every dependency (prod + dev) in one step — no separate `pip install` needed.

### 2. Configure environment

```bash
cp .env.example .env
cp .env.test.example .env.test
# Edit .env and fill in your EMBEDDING_API_KEY (if using openai driver)
# and LLM_API_KEY (required). DATABASE_URL already matches the local
# Docker setup below — no edit needed for that one. .env.test only
# overrides DATABASE_URL for tests/db/ — see AGENT_ENV below.
```

### 3. Start the local database and run migrations

```bash
make setup
```

This one command: starts a local Postgres+pgvector via `docker-compose.yml` (no account/signup needed — see [Local Database (Docker)](#local-database-docker) below), waits for it to be healthy, and runs migrations against **both** `docs_agent` (dev) and `docs_agent_test` (test, a genuinely separate database — see [AGENT_ENV](#agent_env-and-envtest) below).

Equivalent manual steps, if you'd rather run them one at a time:

```bash
make docker-up          # starts Postgres, waits until healthy
make db-migrate         # migrates docs_agent (DATABASE_URL from .env)
make db-migrate-test    # migrates docs_agent_test (AGENT_ENV=test -> .env.test)
```

Migration output should look like:

```
Running migrations (Batch 1):
  Migrating: 0001_create_document_chunks_table ... DONE (0.36s)
Migration complete!
```

Running it again is safe — already-applied migrations are skipped:

```
Nothing to migrate. Database schema is up to date.
```

## Local Database (Docker)

`docker-compose.yml` runs a single local Postgres+pgvector server (`pgvector/pgvector:pg16`) that hosts **two separate databases**:

| Database | Used by | Created by |
|---|---|---|
| `docs_agent` | `DATABASE_URL` in `.env` — the app itself | `POSTGRES_DB` in `docker-compose.yml` |
| `docs_agent_test` | `DATABASE_URL` in `.env.test` — `tests/db/` | `docker/init-test-db.sql`, run once on first startup |

They're genuinely separate databases, not the same one reused — `tests/db/` runs real `INSERT`/`DELETE`/`TRUNCATE`, and migration tests even `CREATE`/`DROP TABLE`, so running the test suite must never touch data you're actually using. Neither the init script nor `docker-compose.yml` creates any tables or the `vector` extension — that stays owned by `migrations/`, so there's one canonical source of truth for the schema regardless of which database it's applied to.

Data persists in a named Docker volume (`docs_agent_pgdata`) across restarts. If you ever want a completely clean slate (re-runs the test-database init script too):

```bash
make docker-down-clean   # stops the container AND deletes the data volume
make setup                # starts fresh and re-migrates both databases
```

A managed Postgres (e.g. Supabase) still works too — see the commented-out alternative in `.env.example`. `db.py`'s connection logic doesn't care whether `DATABASE_URL` points at Docker or a managed instance.

## AGENT_ENV and `.env.test`

`config.py` always loads `.env` first. If `AGENT_ENV=test` (a real shell/CI variable — `make test` sets it automatically), it also loads `.env.test` **on top**, overriding just the keys that differ (in practice, only `DATABASE_URL`) — everything else (`LLM_API_KEY`, `EMBEDDING_DRIVER`, ...) is inherited unchanged from `.env`.

`AGENT_ENV` itself must **never** be set inside `.env`/`.env.test` — deciding which file(s) to load requires already knowing `AGENT_ENV`, so putting it inside a conditionally-loaded file is circular. It only ever comes from the real process environment, with `local` as the default when unset. You shouldn't normally need to set it by hand; `make test`/`make db-migrate-test` already do.

If `AGENT_ENV=test` but no `.env.test` exists, `config.py` raises immediately with a clear error rather than silently falling back to `.env`'s `DATABASE_URL` — otherwise `tests/db/` could truncate whichever database `DATABASE_URL` happens to point at.

## Makefile Shortcuts

Run `make` or `make help` any time for this same list straight from the terminal — it's generated from each target's own `## ` comment, so it can't drift out of sync the way a hand-maintained table can.

| Command | Equivalent |
|---|---|
| `make setup` | `docker-up` + `db-migrate` + `db-migrate-test` — one-shot onboarding |
| `make docker-up` | `docker compose up -d --wait` — start local Postgres, wait until healthy |
| `make docker-down` | `docker compose down` — stop the container, keep its data |
| `make docker-down-clean` | `docker compose down -v` — stop the container **and delete its data** |
| `make db-migrate` | `uv run python scripts/migrate.py up` — migrates `DATABASE_URL` (`.env`) |
| `make db-migrate-test` | `AGENT_ENV=test uv run python scripts/migrate.py up` — migrates `DATABASE_URL` from `.env.test` instead |
| `make db-flush` | `uv run python scripts/db_flush.py` — truncates `document_chunks` (rows only, keeps the schema) |
| `make db-refresh` | `db-flush` then `db-migrate` — empty the table and re-apply any pending migrations in one command |
| `make migrate-status` | `uv run python scripts/migrate.py status` — applied vs. pending migrations |
| `make migrate-install` | `uv run python scripts/migrate.py install` — create the `schema_migrations` table only |
| `make migrate-fresh` | `uv run python scripts/migrate.py fresh` — revert every migration, drop tracking, re-apply everything from scratch |
| `make migrate-rollback` | `uv run python scripts/migrate.py rollback` — revert the most recently applied *batch* |
| `make migrate-reset` | `uv run python scripts/migrate.py reset` — revert every applied migration |
| `make migrate-refresh` | `uv run python scripts/migrate.py refresh` — `reset` then `up` |
| `make make-migration name=<snake_case_name>` | `uv run python scripts/make_migration.py <snake_case_name>` — scaffold a new migration file |
| `make add-document path=<file>` | One-shot ingestion — `ingestion.ingest.add_document()` on `<file>` |
| `make query q="<question>"` | One-shot question — full pipeline (`query.retrieval.query_knowledge_base()`), real LLM call |
| `make mcp-dev` | `uv run mcp dev mcp_server.py` — runs `mcp_server.py` under the MCP Inspector for local testing |
| `make mcp-install` | `uv run mcp install mcp_server.py --name "docs-agent" -f .env` — registers it with Claude Desktop |
| `make test` | `AGENT_ENV=test uv run pytest -v` — a session-scoped pytest fixture (`tests/db/conftest.py`) migrates and truncates the test DB itself, so this works regardless of how pytest gets invoked |
| `make lint` | `uv run ruff check .` |

Every `db-*` and `migrate-*` command (except `db-migrate-test`, and `test`) acts on whatever `DATABASE_URL` is currently set to in `.env` — with the local Docker setup that's the separate `docs_agent` database, so this is safe by default; if you point `DATABASE_URL` at a shared/managed database, double-check `.env` before running them.

## Embedding Drivers

The project uses the **Strategy / Driver pattern** so the embedding backend is swappable via config:

| `EMBEDDING_DRIVER` | Model | Cost | Language support |
|---|---|---|---|
| `local` (default) | `paraphrase-multilingual-MiniLM-L12-v2` | Free, offline | 50+ languages incl. Hungarian |
| `openai` | `text-embedding-3-small` | Paid API | Primarily English |

Set `EMBEDDING_DRIVER=openai` in `.env` to switch — no code changes needed.

## Chunking & Token Limits

Embedding models don't read arbitrarily long text — each one has a maximum input length in *tokens* (not words), and text beyond that limit is **silently truncated** during embedding, not rejected. The truncated tail becomes invisible to retrieval, which can badly hurt answer quality without ever raising an error. Confirmed empirically for the default local model: `paraphrase-multilingual-MiniLM-L12-v2` truncates at only **128 tokens** — far below the 512 an earlier version of this project's docs assumed.

Since `CHUNK_SIZE` (`chunker.py`) is configured in *words*, not tokens, what happens next depends on `CHUNK_OVERFLOW_STRATEGY` (`.env`, default `warn`) — a Strategy pattern in `ingestion/chunker.py`, same shape as the embedding/LLM drivers:

- **`warn`** (default) estimates the token count using an approximate `WORDS_PER_TOKEN` ratio (default `0.75`, an English average) and **warns** — via `warnings.warn`, not an error — when a chunk is *likely* to get truncated. This ratio is only an approximation: subword tokenizers typically produce *more* tokens than words, and morphologically rich languages like Hungarian tend to tokenize *worse* (fewer words per token) than the English-based default. If the warning under-fires for your content, lower `WORDS_PER_TOKEN` in `.env`. It still doesn't correct anything — the chunk is stored and silently truncated at embed time regardless.
- **`split`** measures each chunk's *real* token count with the active driver's own tokenizer (`EmbeddingDriver.count_tokens()` — only `LocalSentenceTransformerDriver` implements this, via its raw HuggingFace tokenizer, not `model.tokenize()`, which was confirmed empirically to already truncate) and, for any chunk that actually overflows, splits it into **balanced** pieces — a hard guarantee, not an estimate. It first estimates how many pieces are needed (`ceil(total_tokens / max_seq_length)`) and aims each one at an even share of that, binary-searching the real token count per piece to stay correct. Greedily maxing out each piece up to the hard limit instead (the first version of this) sounds optimal but isn't: a chunk only slightly over the limit (e.g. 132 tokens vs. a 128 limit) would produce one full 128-token piece and a near-empty ~4-token straggler — almost useless as its own embedding, too little semantic content for retrieval to ever match it well. Falls back to `warn` (with a warning explaining why) if the active driver can't report real token counts, e.g. `EMBEDDING_DRIVER=openai`.

Run `uv run python scripts/inspect_chunks.py` (uses `TEST_DOC_PATH` by default, or pass a path — PDF or Markdown) to see all of this at once for your own documents: one table, every `PDF_EXTRACTION_MODE` x `CHUNKING_STRATEGY` x `CHUNK_OVERFLOW_STRATEGY` combination as its own row, each with a bar showing that row's *worst* chunk against the model's real token limit, plus real processing time in milliseconds (model-load and other one-time import costs are explicitly warmed up beforehand so they don't unfairly inflate whichever row happens to run first) — so which combination actually needs the least correction, and whether that correction is worth its cost, is a glance, not six separate reports to compare by hand.

Verified against a real document (`CHUNK_SIZE=50`, `WORDS_PER_TOKEN=0.4`): the `warn` heuristic estimates `50 / 0.4 = 125` tokens, just under the 128-token limit, so it **didn't warn at all** — yet 8 of the 21 actual chunks (38%) really did exceed 128 tokens (up to 198), because real token count varies a lot per chunk's content, not just per configured size. `split` caught and corrected all 8 (21 → 29 chunks), each landing between 53 and 127 tokens — no near-empty stragglers, and afterward zero chunks exceed the limit.

**Known limitation:** the `CHUNK_OVERLAP` (an absolute word count) isn't reconsidered by either strategy. If `CHUNK_SIZE` were drastically lowered to match a tight token limit, a fixed `CHUNK_OVERLAP` could become a disproportionately large fraction of it. Not addressed yet.

## Supported Formats, Extraction Mode & Chunking Strategy

`add_document()` accepts PDF (`.pdf`) and Markdown (`.md`/`.markdown`) files — the format is detected from the extension via `ingestion/extractors.py`'s `get_extractor()`, a Strategy pattern like the embedding/LLM drivers, but selected by file extension rather than an `.env` setting (there's nothing to prefer — the file's format is a fact, not a choice). `PDFExtractor` wraps `pdf_loader.py`'s pdfplumber-based extraction; `MarkdownExtractor` just reads the file directly — Markdown already marks its own paragraph breaks (blank lines) and structure (`#` headers), so there's no coordinate-based heuristic to run, unlike PDF.

Markdown files have no real "pages", so chunk metadata's `page_number` is instead a **header-based section index** for them (every `#`...`######` line starts a new section) — the same field, same purpose (citing roughly where in the document a chunk came from), just a different unit depending on the source format. A `#` inside a fenced code block (e.g. a Python/shell comment in a documentation example) is correctly not treated as a header.

Two independent settings control how a PDF becomes chunks, both in `.env` (Markdown ignores both — see above):

- **`PDF_EXTRACTION_MODE`** (`flat` default, or `blocks`) — how page text is turned into one document-level string. `flat` just joins pages with a space; `blocks` additionally detects paragraph breaks from word coordinates (`pdfplumber`'s `extract_words()`, a per-page median line-spacing × 1.8 threshold) and preserves them, so a structure-aware chunker can split on them. `blocks` can't detect a break that happens to fall exactly at a page boundary — coordinates reset per page, so there's nothing to compare the gap against across it. Accepted v1 limitation, since `blocks` is opt-in (not the default) — worth revisiting if it ever matters in practice.
- **`CHUNKING_STRATEGY`** (`word` default, or `langchain`) — how the resulting text becomes chunks. `word` is the original sliding word-count window (`CHUNK_SIZE`/`CHUNK_OVERLAP`), unchanged in behavior. `langchain` uses `langchain_text_splitters.RecursiveCharacterTextSplitter` to try paragraph, then line, then sentence, then word boundaries in order — keeping whole paragraphs/sentences together whenever they fit, instead of cutting at a fixed word count regardless of structure.

Both dimensions are independent and combinable (e.g. `blocks` + `langchain` for the most structure-aware result on well-formatted PDFs) and default to the original behavior for backward compatibility.

**A design tradeoff worth knowing about `langchain`**: each piece it returns needs to be mapped back to a word-index (for the `page_number` metadata), via character-offset tracking rather than plain word-counting — `RecursiveCharacterTextSplitter` defaults to leaving a bare leftover separator at the start of the *next* piece when it splits at `". "`/`", "` (e.g. `". word6 word7"` instead of `"word6 word7"`), which breaks simple word-counting. This driver sets `keep_separator=False`, confirmed empirically to make every such split land cleanly on a word boundary — except one remaining edge case: a single "word" longer than the entire chunk budget forces the splitter's last-resort empty-string separator, which can still land mid-word. A stricter fix (an `assert` on the word-boundary assumption, raised in a code review) was considered and rejected: it would turn a rare, low-impact inaccuracy — the affected chunk's `page_number` could be off by one — into a hard crash during ingestion. The accepted tradeoff is the same one `_split_oversized_text` already documents for the `word` strategy: word-level granularity can't split a single oversized "word" any finer, so it's left as-is rather than crashing or guessing.

**The real fix underneath both**: chunking used to run **per page** (`chunk_pages()`), with no overlap between pages — a paragraph that happened to span a page break was silently split into two truncated, unrelated chunks. `ingest.py` now concatenates the whole document first (`pdf_loader.extract_document_text()`) and chunks that (`chunker.chunk_document()`) — there's no page loop left to truncate anything at a page boundary. `chunk_pages()` itself is unchanged and still used by `scripts/extract_text.py`-adjacent tooling and its own tests.

Since a chunk's words can now come from more than one page, `page_number` in its metadata is assigned by **majority vote** (whichever page contributed the most words) rather than a page range — simpler, backward-compatible with the existing single-number metadata shape, and the rare off-by-one-page citation is a negligible cost next to not having to change the prompt template and retrieval display for a page *range*.

## Retrieval: Hybrid Search, Reranking & Quality Evaluation

The retrieval side went through the same "own numbers, not just intuition" treatment as the chunking side above. This section is a direct answer to four things worth knowing about it: how search combines vector and keyword matching, how (and whether) results get reranked, how quality is actually measured, and what happens when nothing relevant exists.

### Hybrid search (vector + keyword, fused by rank)

Pure cosine-similarity search (the original design) misses one common case: an exact name, number, or code-like token can score poorly on embedding similarity even when it's a perfect keyword match — the embedding "smooths over" exact tokens that a keyword search finds trivially. `query/retrieval.py`'s `retrieve_chunks()` runs **both**, by default:

- `VectorStore.search()` — pgvector cosine similarity (unchanged).
- `VectorStore.search_fulltext()` — Postgres full-text search over a generated `tsvector` column (`migrations/0002_add_fulltext_search.py`), using the `simple` text-search configuration deliberately, not `english`/`hungarian` — the corpus mixes both languages, and a single language-specific configuration (with its stemming and stopword list) would only serve one of them well.

The two ranked lists are combined with **Reciprocal Rank Fusion** (`query/hybrid.py`'s `reciprocal_rank_fusion()`): every chunk's fused score is `Σ 1/(rank + k)` across whichever list(s) it appears in (`k=60`, the standard default). RRF fuses by **rank position**, not raw score — cosine similarity (0–1) and `ts_rank` (unbounded) live on incompatible scales, so averaging or weighting the raw numbers directly would be comparing apples to oranges. This is the same technique Elasticsearch/OpenSearch's built-in hybrid search uses: simple, no training, no extra model.

**A real bug this surfaced**, found while building the eval script below, not by inspection: `search_fulltext()` originally passed the raw question straight into `websearch_to_tsquery('simple', question)`. Because `simple` has no stopword list (that's exactly why it was chosen — see above), every word of the question — including grammar words like "milyen"/"used"/"is" — became a **mandatory** term (`websearch_to_tsquery` ANDs bare words together). A real chunk almost never contains a question's grammar words verbatim, so keyword search was silently returning **zero results for nearly every natural-language question**, undetected until the eval script's real numbers showed `0 keyword result(s)` on every single run. The fix: the question's words are OR-joined (`" or ".join(query_text.split())`) before being passed to `websearch_to_tsquery`, so a chunk matching *any* of the question's content words now contributes to the fusion, ranked by how many/how prominently they matched. Covered by both a unit test (asserts the OR-joined string reaches the query) and a DB test (a real sentence full of grammar words that would have failed pre-fix).

Which retrieval path runs is itself a Strategy (`query/retrieval.py`'s `RetrievalStrategy` ABC, same shape as every other driver/strategy in this project), controlled by `RETRIEVAL_STRATEGY` (`.env`, default `hybrid`): `hybrid` (`HybridRetrievalStrategy`) is everything described above; `vector` (`VectorRetrievalStrategy`) skips keyword search and fusion entirely, reproducing the pre-hybrid-search behavior exactly (same `min_score` filtering, same ordering). Kept as a real, selectable strategy rather than a one-off comparison hack specifically so `scripts/evaluate_retrieval.py` measures the actual production code path, not a hand-rolled stand-in that could quietly drift out of sync with it. In practice there's little reason to prefer `vector` day-to-day — hybrid search only ever adds recall on top of it, at negligible extra cost (one more indexed Postgres query and a pure fusion function, no model involved) — its main use is exactly that eval/debug comparison.

### Reranking (optional, off by default)

Hybrid search produces a wide, cheap candidate pool (`RETRIEVAL_CANDIDATE_POOL_SIZE`, default 20). `drivers/reranker.py` adds an optional second stage: a **cross-encoder** scores each `(question, chunk)` pair *jointly* (not independently, like an embedding) — more accurate, but too expensive to run over a whole corpus, so it only ever reranks that already-small candidate pool. This is the standard "retrieve-then-rerank" architecture.

Model choice: **`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`**, a multilingual cross-encoder (MS MARCO machine-translated into 14 languages), instead of the far more common English-only `cross-encoder/ms-marco-MiniLM-L-6-v2` — this project's content and embedding model are both multilingual (Hungarian + English), and an English-only reranker would be a regression on Hungarian content specifically. Verified empirically, not just by model-card description: given the Hungarian sentence *"Magyarországon a személyi jövedelemadó (SZJA) mértéke egységesen 15 százalék"* against the question *"Mennyi az SZJA kulcsa Magyarországon?"*, it scored **+6.39** — clearly separated from an unrelated Hungarian sentence (**-6.13**) and an unrelated English one (**-9.24**).

Controlled by `RERANKER_DRIVER` (`.env`, default `none`) — a Strategy pattern like every other driver in this project. `none` (`NoopRerankerDriver`) passes the RRF-fused order through unchanged; `cross_encoder` (`CrossEncoderRerankerDriver`) reorders by the model's score. Defaulting to `none` was a deliberate choice, not a placeholder: the cross-encoder model must be downloaded and loaded (real latency, real memory) on every process that queries the knowledge base, and hybrid search's own fusion is already a reasonable ranking on its own — reranking is an *optional* quality lever, not a required part of the pipeline.

### How quality is measured

`scripts/evaluate_retrieval.py` + `tests/data/eval_questions.json` — 13 hand-written questions (10 answerable, 3 deliberately unanswerable) against the two real, committed fixtures (`tests/data/sample.md`, `tests/data/sample.pdf`). It runs every question through `retrieve_chunks()` with each `RetrievalStrategy` swapped in explicitly — **vector-only** (`VectorRetrievalStrategy`) and **hybrid+rerank** (`HybridRetrievalStrategy`, with whatever `RERANKER_DRIVER` is currently configured) — through the exact same production code path, not a hand-rolled duplicate, and reports:

- **Recall@k**: did a chunk from the expected source file appear anywhere in the top-k?
- **MRR** (Mean Reciprocal Rank): how high up was the first correct chunk?
- **Fallback rate**: for the unanswerable questions, did the retrieval-layer relevance gate correctly return nothing?

Run it with `uv run python scripts/evaluate_retrieval.py`. It's safe to run against any configured `DATABASE_URL`, including a populated dev database: it never deletes anything, only adds the two fixtures if they're not already present (`VectorStore.has_chunks_from_source`), and prints which database it's about to touch. Run with `AGENT_ENV=test` instead for a fully controlled corpus (no other documents mixed in).

**Honesty about what these numbers mean**: this corpus is two documents. Recall@k and MRR come out identical (**1.00**) for both configurations here, and that's an honest, expected result of the corpus being this small — with only two possible source documents, hybrid fusion and reranking don't have much room to change an already-easy ranking. These are *our own, repeatable* numbers for comparing configurations against each other as the corpus grows, not a statistically meaningful benchmark.

**A more interesting, honest finding** came from the fallback-rate metric: it measured **0.33** — 1 of the 3 deliberately unanswerable questions was correctly caught by the retrieval-layer gate alone, the other 2 weren't. Inspecting the real scores explained why: a question that stays *topically* on-subject but asks for a fact the document never states can still score above `RETRIEVAL_MIN_SCORE` (0.25) — e.g. 0.438 for "what is the founder's phone number?" and 0.370 for "what is the project's annual revenue?" — because cosine similarity reflects "is this about the same topic", not "does this contain the specific fact asked for". The one that *was* caught ("what programming language is the backend written in?", scoring 0.249) shows how close this can run either way — a hair's width under the 0.25 threshold, not a clean rejection. That's a genuine limitation of similarity-threshold gating, not a bug, and it's exactly why the system doesn't rely on that gate alone (see below).

### What happens when there's no reliable source

Two independent layers, not one:

1. **Retrieval-layer gate** (`query/retrieval.py`'s `_passes_relevance_gate`): if *nothing* in the vector-search candidate pool clears `RETRIEVAL_MIN_SCORE`, `query_knowledge_base()` returns `NO_RESULTS_MESSAGE` immediately — no LLM call at all. This deliberately checks pure vector similarity only, never the fused RRF/reranker score, since those live on scales with no equivalent calibrated "irrelevant" cutoff. This catches questions genuinely unrelated to anything in the knowledge base.
2. **Prompt-level grounding instruction** (`drivers/llm.py`'s `_build_prompt`): the system prompt explicitly instructs the model to answer strictly from the provided excerpts and respond with an exact "I could not find this information in the provided documents" if the excerpts don't answer the question. This catches the case the eval script's fallback-rate finding demonstrated above — a topically-relevant chunk that still doesn't contain the specific fact asked for — which a similarity threshold structurally cannot distinguish from a genuine answer.

**Verifying layer 2 actually works — and a real caveat found doing it**: `scripts/evaluate_retrieval.py --with-llm` (see below) generates a real answer for the 3 unanswerable questions specifically to check this. Running it surfaced something worth knowing before trusting any number out of it: with the default `LLM_MODEL=openrouter/free`, which auto-routes to *whichever* free model is available per call (not a fixed one), some calls — for **both** answerable and unanswerable questions, so it isn't tied to retrieval quality at all — came back with a broken, non-answer string (`"User Safety: safe"`/`"User Safety: unsafe"`) instead of a real completion, apparently a moderation-layer artifact from whichever free model got auto-selected that call. That's real, useful signal about `openrouter/free`'s reliability for this kind of measurement, but it also means a `--with-llm` run's raw decline-rate number can't be trusted as reproducible evidence by itself — it's confounded by which random free model happened to answer. Pin `LLM_MODEL` to a fixed, non-`:free` model before drawing any real conclusion from this layer.

## Local Diagnostic & Evaluation Scripts

Three hand-runnable scripts, no test framework involved — point them at a real document (or the committed fixtures) and read the output. Each one uses the exact same production code the real pipeline does (extractors, `chunk_document()`, `retrieve_chunks()`), never a reimplementation, so what they show is what `add_document()`/`query_knowledge_base()` would actually do.

| Script | What it's for | Run it |
|---|---|---|
| `scripts/extract_text.py` | Sanity-check a document *before* ingesting it — did text actually come out, grouped by page/section? Catches e.g. a scanned (image-only) PDF with no text layer early. | `uv run python scripts/extract_text.py /path/to/file.pdf` |
| `scripts/inspect_chunks.py` | Compare every `PDF_EXTRACTION_MODE` x `CHUNKING_STRATEGY` x `CHUNK_OVERFLOW_STRATEGY` combination for one document side by side — chunk count, token size vs. the embedding model's real limit (as a bar), processing time. See "Chunking & Token Limits" above. | `uv run python scripts/inspect_chunks.py /path/to/file.pdf` |
| `scripts/evaluate_retrieval.py` | Compare **vector-only vs. hybrid+rerank** retrieval quality — Recall@k, MRR, Fallback rate, plus a per-question breakdown and (with `--with-llm`) real generated answers. See "How quality is measured" above. | `uv run python scripts/evaluate_retrieval.py` |

All three fall back to `TEST_DOC_PATH` (`.env`) when no path is given, except `evaluate_retrieval.py`, which always runs against the two committed fixtures (`tests/data/sample.md`/`sample.pdf`) — see its own "Which database?" note above for why it's safe to run against a real, populated database.

**`evaluate_retrieval.py`'s per-question breakdown**, specifically: the aggregate Recall@k/MRR/Fallback rate numbers can land on identical values for both configurations purely because the corpus is small — that hides whether the two strategies actually behave differently on any *individual* question. Every run also prints a row per question, `vector` vs. `hybrid`, with a `<- differs` marker wherever the two disagree, so a difference is visible even when the averages coincide. A real example from this project's own fixtures: both configs score `1.00`/`1.00`/`0.33` in aggregate, and yet the breakdown shows the "what is the project's annual revenue?" question keeps 1 chunk under `vector` but 2 under `hybrid` — a real, if small, difference the averages alone completely hid.

**`--with-llm`**: none of the above touches the LLM — Recall@k/MRR/Fallback rate are all retrieval-layer-only, deliberately, to stay fast and free to run. Passing `--with-llm` additionally generates a real answer (real network call, `LLM_DRIVER`) for every question under both configurations, and for the 3 deliberately unanswerable ones, checks whether the answer actually reads like a decline — this is what closes the gap "What happens when there's no reliable source" describes above: Fallback rate alone only proves the *retrieval* gate didn't catch these three, not whether the *system* as a whole still ends up hallucinating. Run once with `--with-llm` and read the printed answers to find out — but see that same section's caveat about `LLM_MODEL=openrouter/free` before trusting the decline-rate number itself.

## MCP Server

`mcp_server.py` exposes `add_document`/a retrieval tool to any [MCP](https://modelcontextprotocol.io)-compatible host — most directly, Claude Desktop, so a document can be ingested and the knowledge base searched from inside a normal chat. Doesn't affect the interview-relevant retrieval design above; this is a separate, personal-use integration.

**Transport is stdio, not HTTP** — the host (Claude Desktop) spawns `mcp_server.py` itself and talks over stdin/stdout, which is what "add a local MCP server" means and needs no extra infrastructure. A network-reachable version (FastAPI + Docker) would only matter for a *remotely* accessible server and isn't built.

**The retrieval tool is named `search_knowledge_base`, not `query_knowledge_base`, and deliberately doesn't call this project's own `LLM_DRIVER`** — it wraps `query.retrieval.retrieve_chunks()` directly and returns the raw excerpts (`content`/`source_file`/`page_number`), not a generated answer. The point: the MCP *host's own model* (e.g. whatever Claude Desktop is already running) writes the final grounded answer from those excerpts, in its own conversation turn — no extra API call or cost to this project at all. The grounding instruction ("answer only from these excerpts, say so if they don't answer the question") is carried in the tool's `description`, the same mechanism `drivers/llm.py`'s own prompt uses, just addressed to the host model instead of `LLM_DRIVER`.

**`search_knowledge_base` alone wasn't reliable in practice — `/my-docs` is the fix.** Tested for real in Claude Desktop: asked about "the Player Central MVP's technology stack," and got back a detailed, entirely wrong answer (a Laravel/Nuxt-4/Stripe+Billingo stack) — nothing like the actual fixture content. The model hadn't used the tool's results at all; it answered from its own memory of an unrelated, same-named real project. A `tools` call is the *model's own judgment call*, and that judgment isn't trustworthy enough to rely on alone, even with a strongly-worded description. The fix is a `my-docs` **prompt** (a different MCP primitive from `tools`) — `/my-docs <question>` in Claude Desktop is *user*-invoked, so the instruction to call `search_knowledge_base` and answer only from its results becomes mandatory, not a suggestion the model can talk itself out of. Confirmed working afterward: forcing tool-only mode surfaced an honest, grounded, partial answer instead (see the chunking finding below) rather than a confident wrong one — though a second real test showed Claude Desktop's own memory feature *still* contributed alongside a correct, transparent tool call (labeled separately, not conflated — but still there). Tightened `/my-docs`'s wording further as a result ("no other tool", "every claim traceable to a specific excerpt") — a best-effort improvement, not a guarantee, since that memory feature is client-side and outside what any MCP prompt can necessarily override. For consistency, `drivers/llm.py`'s own system prompt (used by `query_knowledge_base()`'s `LLM_DRIVER` call, a separate code path from MCP entirely) got the same tightening: explicitly forbids outside/training knowledge and requires saying what's missing on a partial match, not just citing sources.

**Five real bugs found while verifying this against a real host, not by inspection or by reading the SDK's docs alone:**

1. `ingestion/ingest.py`/`query/retrieval.py` used to report progress via `print()`. Over MCP's stdio transport, stdout is reserved for the JSON-RPC protocol — a stray `print()` line landed on that channel mid-call and broke a real client's message parsing (`Failed to parse JSONRPC message from server`), confirmed by spawning `mcp_server.py` as a real subprocess and calling both tools over the actual protocol (not just a direct Python-level call, which would never have caught this). Fixed by switching both modules to Python's `logging` module (stderr by default, invisible to the protocol stream) — `agent.py`/`scripts/evaluate_retrieval.py`/the `make add-document`/`query` targets configure a bare `logging.basicConfig(format="%(message)s")` so their own CLI output looks exactly as before.
2. `uv run mcp install`'s own generated launch command (`uv run --with "mcp[cli]==X.Y.Z" mcp run <path>`) is built for a standalone, dependency-free single-file script — the SDK's own docs say so explicitly ("works from any directory... no project needed"). `mcp_server.py` isn't that: it imports the whole project (`psycopg2`, `openai`, `sentence-transformers`, ...). Registering it as-is and actually launching it from Claude Desktop failed immediately with **"Server disconnected"**; reproduced directly by spawning the exact generated command from an unrelated directory with a clean environment (no inherited venv — matching how Claude Desktop actually spawns it): `ModuleNotFoundError: No module named 'psycopg2'`. Fixed with `uv run --project <docs-agent dir> mcp_server.py` instead, which resolves against this project's own environment regardless of the caller's working directory — confirmed working the same way (a real MCP client, unrelated cwd, no inherited venv). `make mcp-install` now runs `scripts/fix_mcp_install.py` right after `mcp install` to rewrite the generated config entry automatically, since `mcp install` itself has no flag for this.
3. **`add_document()` had no protection against re-ingesting the same file** — asking Claude Desktop to add a document already in the knowledge base silently duplicated its chunks. This isn't just wasted storage: with a small corpus and a fixed `RETRIEVAL_TOP_K`, duplicate copies of the same 2 chunks crowded *out* other, genuinely relevant unique chunks from a real query's top-k results — directly degrading answer quality, not just cleanliness. `add_document()` now raises `ValueError` if `VectorStore.has_chunks_from_source()` says the file's already present (an existing method, previously only used by `scripts/evaluate_retrieval.py`'s own idempotent seeding), with an explicit `force=True` escape hatch for genuinely wanting a duplicate.
4. When forced to answer from the knowledge base alone (via `/my-docs`), the model correctly identified that the "Technology Stack" list was retrieved incompletely — the bullet `Nuxt 3 (mobile-first UI, ...)` splits across two adjacent chunks, and the query only surfaced one of them. Root cause, confirmed by inspecting `retrieve_chunks()`'s actual output directly: it was bug 3 above — duplicate chunks were occupying the other top-k slots that should have gone to the chunk with the rest of the list.
5. Re-adding an already-present document (bug 3's guard, working as designed) surfaced with a **completely blank error message** in Claude Desktop — the model couldn't say what had gone wrong. Cause: `mcp_server.py`'s `add_document` let the plain `ValueError` propagate, and the MCP SDK treats any exception besides its own `ToolError`/`MCPError` as an *unexpected crash* — deliberately hiding the exception's text from the client (only the server's own log gets it), on the reasoning that an unanticipated crash's internals aren't something the model could have avoided anyway. Confirmed via a real client call: `is_error=True` with no usable message. Fixed by catching `ValueError`/`FileNotFoundError` (the specific, "the model could retry with different arguments" cases — an infra failure like a DB being unreachable is deliberately left as a real crash instead) and re-raising as `ToolError`, whose message *does* reach the client — confirmed the same way, the real error text ("... is already in the knowledge base. Pass force=True ...") now comes through intact.

Register it with Claude Desktop: `make mcp-install` (reads `LLM_API_KEY`/`DATABASE_URL`/etc. from `.env` into the registered entry, since Claude Desktop spawns the server with none of the current shell's environment, and patches the launch command per bug 2 above), then fully quit and reopen Claude Desktop. Try `/my-docs <question>` for a forced, tool-only answer, or just ask normally and hope the model calls `search_knowledge_base` on its own (see the finding above for why that's not guaranteed). Try it standalone first with `make mcp-dev`, which opens the MCP Inspector for calling either tool (or the prompt) by hand.

**For maximum reliability, pair `/my-docs` with your own explicit reinforcement**, e.g. typing "Only use the docs-agent MCP server, nothing else" alongside it. Confirmed empirically: `/my-docs` alone (even with its current, already-strict wording) still let memory bleed into an answer on one real test; the same question with that extra, directly user-authored sentence came back fully tool-grounded, with no memory involved at all, an accurate partial answer, honest about exactly what it didn't know, and correctly cited to `sample.pdf`, page 1. Apparently a directly-user-authored instruction carries more weight for Claude than semantically-equivalent wording embedded in a prompt template — an honest, pragmatic workaround rather than something worth chasing further with more prompt-wording iteration (there's no reliable way to measure whether a different wording is actually better, versus just a different roll of inherent variance between similar runs).

## Database Schema

The `document_chunks` table stores chunked document text alongside its vector embedding:

| Column | Type | Description |
|---|---|---|
| `id` | `BIGSERIAL` | Primary key |
| `content` | `TEXT` | The raw text chunk |
| `metadata` | `JSONB` | File name, page number, chunk index, etc. |
| `embedding` | `vector(384)` | Embedding vector for similarity search |
| `content_tsv` | `tsvector` (generated) | Full-text search vector, derived automatically from `content` — see "Retrieval" above |
| `created_at` | `TIMESTAMPTZ` | Insertion timestamp |

An **HNSW index** (`vector_cosine_ops`) is created on `embedding` for fast approximate nearest-neighbour search, and a **GIN index** on `content_tsv` for full-text search.

## Code Conventions

- **Language:** All source code, comments, docstrings, and variable names are in **English**.
- **Architecture:** Driver / Strategy pattern for swappable backends.
- **Config:** Single `Settings` dataclass in `config.py` — no scattered `os.getenv()` calls.
- **Dependency management:** `uv` — `pyproject.toml` + `uv.lock` are the single source of truth (no `requirements.txt`).
- **Linting:** `ruff` for formatting and static analysis.
- **Tests:** `pytest`

## Roadmap (per PLAN.md)

- [x] Step 1 — Project setup, venv, packages, `config.py`
- [x] Step 2 — pgvector table + migration runner
- [x] Step 3 — PDF text extraction (`pdfplumber`)
- [x] Step 4 — Chunking + embedding + storage (`add_document` logic)
- [x] Step 5 — Query: embedding + top-k retrieval + answer generation
- [x] Hybrid search (vector + keyword, RRF-fused), optional cross-encoder reranking, and a retrieval-quality eval script — see "Retrieval" above (beyond the original steps, added in response to external evaluation criteria)
- [x] Step 6 — Function-calling agent (`add_document` vs `query_knowledge_base`) — see `agent.py`
- [x] Step 7 *(stretch)* — Wrap tools as an MCP server — see `mcp_server.py` and "MCP Server" below (stdio transport only; a network-reachable version via FastAPI/Docker is a possible future step, not built)
