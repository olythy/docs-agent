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
│   ├── pdf_loader.py         # PDF text extraction (pdfplumber)
│   ├── chunker.py            # Word-based chunking with overlap
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
│   └── extract_text.py       # PDF extraction diagnostic CLI
├── pyproject.toml           # Project metadata, dependencies, pytest config
├── uv.lock                  # Locked, reproducible dependency versions
└── .env.example             # Environment variable template
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
# Edit .env and fill in your EMBEDDING_API_KEY (if using openai driver),
# LLM_API_KEY (required), and DATABASE_URL
```

### 3. Run database migrations

```bash
uv run python scripts/migrate.py up
# or: make db-migrate
```

Output should look like:

```
Running migrations (Batch 1):
  Migrating: 0001_create_document_chunks_table ... DONE (0.36s)
Migration complete!
```

Running it again is safe — already-applied migrations are skipped:

```
Nothing to migrate. Database schema is up to date.
```

## Makefile Shortcuts

| Command | Equivalent |
|---|---|
| `make db-migrate` | `uv run python scripts/migrate.py up` |
| `make db-flush` | `uv run python scripts/db_flush.py` — truncates `document_chunks` (rows only, keeps the schema) |
| `make db-refresh` | `db-flush` then `db-migrate` — empty the table and re-apply any pending migrations in one command |
| `make migrate-status` | `uv run python scripts/migrate.py status` — applied vs. pending migrations |
| `make migrate-install` | `uv run python scripts/migrate.py install` — create the `schema_migrations` table only |
| `make migrate-fresh` | `uv run python scripts/migrate.py fresh` — revert every migration, drop tracking, re-apply everything from scratch |
| `make migrate-rollback` | `uv run python scripts/migrate.py rollback` — revert the most recently applied *batch* |
| `make migrate-reset` | `uv run python scripts/migrate.py reset` — revert every applied migration |
| `make migrate-refresh` | `uv run python scripts/migrate.py refresh` — `reset` then `up` |
| `make make-migration name=<snake_case_name>` | `uv run python scripts/make_migration.py <snake_case_name>` — scaffold a new migration file |
| `make test` | `uv run pytest -v` |
| `make lint` | `uv run ruff check .` |

⚠️ Every `db-*` and `migrate-*` command acts on whatever `DATABASE_URL` is currently set to — there's no separate local database yet, so double-check `.env` before running them.

## Embedding Drivers

The project uses the **Strategy / Driver pattern** so the embedding backend is swappable via config:

| `EMBEDDING_DRIVER` | Model | Cost | Language support |
|---|---|---|---|
| `local` (default) | `paraphrase-multilingual-MiniLM-L12-v2` | Free, offline | 50+ languages incl. Hungarian |
| `openai` | `text-embedding-3-small` | Paid API | Primarily English |

Set `EMBEDDING_DRIVER=openai` in `.env` to switch — no code changes needed.

## Chunking & Token Limits

Embedding models don't read arbitrarily long text — each one has a maximum input length in *tokens* (not words), and text beyond that limit is **silently truncated** during embedding, not rejected. The truncated tail becomes invisible to retrieval, which can badly hurt answer quality without ever raising an error. Confirmed empirically for the default local model: `paraphrase-multilingual-MiniLM-L12-v2` truncates at only **128 tokens** — far below the 512 an earlier version of this project's docs assumed.

Since `CHUNK_SIZE` (`chunker.py`) is configured in *words*, not tokens, `add_document()` estimates the token count using an approximate `WORDS_PER_TOKEN` ratio (default `0.75`, an English average) and **warns** — via `warnings.warn`, not an error — when a chunk is likely to get truncated. This ratio is only an approximation: subword tokenizers typically produce *more* tokens than words, and morphologically rich languages like Hungarian tend to tokenize *worse* (fewer words per token) than the English-based default, since long inflected/compound words don't match a shared, multilingual tokenizer's vocabulary as cleanly. If the warning under-fires for your content, lower `WORDS_PER_TOKEN` in `.env`.

**Known limitation (deliberately not solved yet):** this warning is only a heuristic sanity check on the *configured* `CHUNK_SIZE` — it does not guarantee that no individual chunk is ever truncated, and it doesn't correct anything. A planned follow-up is per-chunk **corrective re-splitting**: after producing each actual chunk, tokenize it with the real model tokenizer and, if that specific chunk still exceeds the limit, split it further right there — a hard guarantee with no global state, instead of a global estimate. Not built yet; revisit when document completeness (e.g. important company documents) matters more than moving on to other fixes.

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
