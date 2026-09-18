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
| `add_document(file_path)` | User wants to ingest a new PDF |
| `query_knowledge_base(question)` | User wants to ask a question |

*(The function-calling dispatch layer above — the LLM actually choosing between the two tools — is Step 6 on the roadmap and isn't wired up yet. Today `add_document` and `query_knowledge_base` are plain Python functions you call directly.)*

## Architecture

```
.
├── config.py               # Centralized Settings (env + defaults)
├── db.py                   # Postgres connection factory — nothing else
├── store.py                # VectorStore: all document_chunks persistence (save/search)
├── drivers/
│   ├── embedding.py         # EmbeddingDriver strategy: local (sentence-transformers) vs openai
│   └── llm.py                # AnswerDriver strategy: openrouter vs openai
├── ingestion/
│   ├── extractors.py         # Extractor strategy: PDF vs Markdown, chosen by file extension
│   ├── pdf_loader.py         # PDF text extraction (pdfplumber), flat/blocks modes
│   ├── chunker.py            # Chunking strategies (word/langchain) + overflow correction
│   └── ingest.py             # add_document orchestration
├── query/
│   └── retrieval.py          # query_knowledge_base: retrieval + answer generation
├── migrations/              # Python migrations (Laravel-artisan-style runner)
│   ├── base.py                # Migration ABC: up()/down() run raw SQL, no ORM
│   └── 0001_create_document_chunks_table.py
├── scripts/
│   ├── migrate.py            # Migration runner: uv run python scripts/migrate.py [subcommand]
│   ├── make_migration.py     # Scaffold a new migration file
│   ├── db_flush.py           # Truncate document_chunks
│   ├── extract_text.py       # PDF extraction diagnostic CLI
│   └── inspect_chunks.py     # Chunking diagnostic CLI: full strategy comparison matrix, with bars
├── docker-compose.yml       # Local Postgres+pgvector (dev + test databases)
├── docker/
│   └── init-test-db.sql      # Creates the "docs_agent_test" database on first startup
├── pyproject.toml           # Project metadata, dependencies, pytest config
├── uv.lock                  # Locked, reproducible dependency versions
├── .env.example             # Environment variable template
└── .env.test.example        # .env.test template — see AGENT_ENV below
```

> **Coming soon:** `agent.py` (the function-calling dispatch layer, Step 6), `tools/`

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

## Database Schema

The `document_chunks` table stores chunked document text alongside its vector embedding:

| Column | Type | Description |
|---|---|---|
| `id` | `BIGSERIAL` | Primary key |
| `content` | `TEXT` | The raw text chunk |
| `metadata` | `JSONB` | File name, page number, chunk index, etc. |
| `embedding` | `vector(384)` | Embedding vector for similarity search |
| `created_at` | `TIMESTAMPTZ` | Insertion timestamp |

An **HNSW index** (`vector_cosine_ops`) is created on `embedding` for fast approximate nearest-neighbour search.

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
- [ ] Step 6 — Function-calling agent (`add_document` vs `query_knowledge_base`)
- [ ] Step 7 *(stretch)* — Wrap tools as an MCP server
