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
| `add_document(file_path)` | User wants to ingest a single document |
| `add_directory(dir_path)` | User wants to batch-ingest an entire directory |
| `query_knowledge_base(question)` | User wants to ask a question |

`agent.py` wires this up: tools are described to the LLM as OpenAI-style `tools=[...]` function schemas, and a single free-form message runs the standard tool-calling loop — the model decides whether to call `add_document`, `add_directory`, `query_knowledge_base`, or neither. Try it interactively with `uv run python agent.py`, or call `agent.run_agent("...")` directly. `AnswerDriver` (the same driver `query_knowledge_base` uses for answer generation) exposes a public `get_client()`/`model` for this — the agent loop needs the raw, tools-capable chat client, not the RAG-specific `answer()` method with its fixed prompt shape.

**Caveat:** tool-calling support is model-dependent, and `LLM_MODEL`'s default (`openrouter/free`, which auto-routes to *some* available free model) isn't guaranteed to support it — pick a model explicitly known to support tools if `agent.py` doesn't behave as expected.

## Architecture

For a diagram of how the pipeline actually flows (ingestion, retrieval, which Strategy/Driver is selected by which setting) and a module responsibility map, see `docs/architecture.md`. This section just lists the directory structure.

```
.
├── agent.py                 # Function-calling loop: LLM picks add_document vs add_directory vs query_knowledge_base
├── mcp_server.py            # MCP server (stdio): search_knowledge_base + add_document + add_directory
├── config.py               # Centralized Settings (env + defaults)
├── models.py                # Core data shapes: ChunkMetadata, Chunk, RetrievedChunk
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
│   └── ingest.py             # add_document and add_directory orchestration
├── query/
│   ├── retrieval.py          # query_knowledge_base: hybrid retrieval + answer generation
│   └── hybrid.py             # reciprocal_rank_fusion: pure RRF fusion logic
├── migrations/              # Python migrations (Laravel-artisan-style runner)
│   ├── base.py                # Migration ABC: up()/down() run raw SQL, no ORM
│   ├── 0001_create_document_chunks_table.py
│   └── 0002_add_fulltext_search.py
├── scripts/
│   ├── dev_cli.py            # Development & infrastructure CLI: docker, setup, doctor, lint (uv run python scripts/dev_cli.py)
│   ├── db_cli.py             # Database CLI: migrations, flush, make-migration (uv run python scripts/db_cli.py)
│   ├── agent_cli.py          # Agent & runtime CLI: query, chat, ingest, mcp-dev, mcp-install, skills-install (uv run python scripts/agent_cli.py)
│   ├── eval_cli.py           # Evaluation & diagnostics CLI: eval, inspect, extract (uv run python scripts/eval_cli.py)
│   ├── log_cli.py            # Telemetry & logging CLI: watch/tail, stats, clear (uv run python scripts/log_cli.py)
│   └── utils.py              # Shared CLI utilities (subprocess runner, paths, terminal formatting)
├── docker-compose.yml       # Local Postgres+pgvector (dev + test databases)
├── docker/
│   └── init-test-db.sql      # Creates the "docs_agent_test" database on first startup
├── docs/
│   ├── decisions.md          # Engineering decision & bug-log history (the "why" behind this README)
│   └── architecture.md       # Pipeline diagrams + Strategy/Driver selection table (the "how it flows")
├── corpus/
│   └── download_court_decisions.py  # Downloads a real-estate-law eval corpus from eakta.birosag.hu (raw/ + meta.csv are gitignored)
├── skills/                  # Claude Code skills (canonical source — `make skills-install` symlinks into .claude/skills/)
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

### Environment & Infrastructure (`scripts/dev_cli.py`)

| Command | Equivalent / Description |
|---|---|
| `make help` *(or `make`)* | Print self-documenting list of all available Makefile targets |
| `make setup` | `uv run python scripts/dev_cli.py setup` — one-shot onboarding (Docker + dev & test DB migrations) |
| `make docker-up` | `uv run python scripts/dev_cli.py docker-up` — start local Postgres (dev+test), wait until healthy |
| `make docker-down` | `uv run python scripts/dev_cli.py docker-down` — stop the container, keep its data |
| `make docker-down-clean` | `uv run python scripts/dev_cli.py docker-clean` — stop the container **and delete its data** (full reset) |
| `make doctor` | `uv run python scripts/dev_cli.py doctor` — verify environment files, Docker status, and DB connections |

### Database & Migrations (`scripts/db_cli.py`)

| Command | Equivalent / Description |
|---|---|
| `make db-migrate` | `uv run python scripts/db_cli.py up` — migrate `DATABASE_URL` (`.env`, dev database) |
| `make db-migrate-test` | `AGENT_ENV=test uv run python scripts/db_cli.py up` — migrate test database (`.env.test`) |
| `make db-flush` | `uv run python scripts/db_cli.py flush` — truncate `document_chunks` (rows only, keeps schema) |
| `make db-refresh` | `uv run python scripts/db_cli.py flush && uv run python scripts/db_cli.py up` — empty table, re-apply pending migrations |
| `make migrate-status` | `uv run python scripts/db_cli.py status` — show applied vs. pending migrations |
| `make migrate-install` | `uv run python scripts/db_cli.py install` — create `schema_migrations` tracking table only |
| `make migrate-fresh` | `uv run python scripts/db_cli.py fresh` — revert everything, drop tracking, re-apply from scratch |
| `make migrate-rollback` | `uv run python scripts/db_cli.py rollback` — revert the most recently applied migration batch |
| `make migrate-reset` | `uv run python scripts/db_cli.py reset` — revert every applied migration |
| `make migrate-refresh` | `uv run python scripts/db_cli.py refresh` — `reset` then `up` |
| `make make-migration name=<snake_case>` | `uv run python scripts/db_cli.py make <snake_case>` — scaffold a new migration file |

### Agent & Runtime (`scripts/agent_cli.py`)

| Command | Equivalent / Description |
|---|---|
| `make add-document <file1> [file2...]`<br>*(or `path="..."`)* | `uv run python scripts/agent_cli.py ingest <file1> [file2...]` — ingest document(s) |
| `make add-directory <dir> [ext=.md]`<br>*(or `path=...`)* | `uv run python scripts/agent_cli.py ingest <dir> [--ext <ext>]` — batch-ingest a directory |
| `make delete-document <file>`<br>*(or `path="..."`)* | `uv run python scripts/agent_cli.py ingest --delete <file>` — delete chunks by path or hash |
| `make query "<question>"`<br>*(or `q="..."`)* | `uv run python scripts/agent_cli.py query "<question>"` — ask a question (full RAG pipeline, real LLM call) |
| `make chat` | `uv run python scripts/agent_cli.py chat` — interactive conversational terminal REPL |
| `make mcp-dev` | `uv run python scripts/agent_cli.py mcp-dev` — run under MCP Inspector |
| `make mcp-install` | `uv run python scripts/agent_cli.py mcp-install` — register with Claude Desktop and auto-patch launch config |
| `make skills-install` | `uv run python scripts/agent_cli.py skills-install` — symlink `skills/` into `.claude/skills/` so Claude Code discovers this project's skills |

### Evaluation & Diagnostics (`scripts/eval_cli.py`)

| Command | Equivalent / Description |
|---|---|
| `make inspect-chunks [<file>]`<br>*(or `path=...`)* | `uv run python scripts/eval_cli.py inspect [<path>]` — chunking strategy matrix against token limit |
| `make extract-text [<file>]`<br>*(or `path=...`)* | `uv run python scripts/eval_cli.py extract [<path>]` — preview text extraction grouped by page/section |
| `make eval` | `uv run python scripts/eval_cli.py eval` — retrieval quality evaluation (vector vs hybrid) |
| `make eval-rerank` | `uv run python scripts/eval_cli.py eval --with-rerank` — evaluation with cross_encoder reranking |
| `make eval-llm` | `uv run python scripts/eval_cli.py eval --with-llm` — evaluation with real LLM answer generation |
| `make eval-all` | `uv run python scripts/eval_cli.py eval --with-rerank --with-llm` — full benchmark (rerank + LLM) |

### Logging & Telemetry (`scripts/log_cli.py`)

| Command | Equivalent / Description |
|---|---|
| `make log [action=...]` | `uv run python scripts/log_cli.py watch [--action <action>]` — live-follow structured telemetry events |
| `make log-tail [n=20]` | `uv run python scripts/log_cli.py tail [-n <n>]` — print recent formatted events and exit |
| `make log-stats` | `uv run python scripts/log_cli.py stats` — summarize telemetry entries, top dropped stopwords, and rerank ratios |
| `make log-clear` | `uv run python scripts/log_cli.py clear` — clear the telemetry log file |

### Testing & Code Quality

| Command | Equivalent / Description |
|---|---|
| `make test` | `AGENT_ENV=test uv run pytest -v` — runs full test suite against test database |
| `make lint` | `uv run python scripts/dev_cli.py lint` — check code style and rules with ruff |
| `make lint-fix` | `uv run python scripts/dev_cli.py lint-fix` — auto-fix lint errors and reformat code |
| `make format` | `uv run python scripts/dev_cli.py format` — format code with ruff format |

Every `db-*` and `migrate-*` command (except `db-migrate-test`, and `test`) acts on whatever `DATABASE_URL` is currently set to in `.env` — with the local Docker setup that's the separate `docs_agent` database, so this is safe by default; if you point `DATABASE_URL` at a shared/managed database, double-check `.env` before running them.

## Embedding Drivers

The project uses the **Strategy / Driver pattern** so the embedding backend is swappable via config:

| `EMBEDDING_DRIVER` | Model | Cost | Language support |
|---|---|---|---|
| `local` (default) | `intfloat/multilingual-e5-small` | Free, offline | 100+ languages incl. Hungarian |
| `openai` | `text-embedding-3-small` | Paid API | Primarily English |

The `local` driver uses **asymmetric embedding**: document chunks are embedded with a `"passage: "` prefix (`embed_documents()`), while query strings use a `"query: "` prefix (`embed_query()`). This matches the retrieval-optimised training of the `e5` model family and measurably improves ranking compared to symmetric embedding (same prefix for both), particularly for cross-lingual queries.

Set `EMBEDDING_DRIVER=openai` in `.env` to switch — no code changes needed.

## Chunking & Token Limits

Embedding models don't read arbitrarily long text — each one has a maximum input length in *tokens* (not words), and text beyond that limit is **silently truncated** during embedding, not rejected. The truncated tail becomes invisible to retrieval, which can badly hurt answer quality without ever raising an error. The default local model (`intfloat/multilingual-e5-small`) has a **512-token** sequence limit.

Since `CHUNK_SIZE` (`chunker.py`) is configured in *words*, not tokens, what happens next depends on `CHUNK_OVERFLOW_STRATEGY` (`.env`, **default `split`**) — a Strategy pattern in `ingestion/chunker.py`, same shape as the embedding/LLM drivers:

- **`split`** (default) measures each chunk's *real* token count with the active driver's own tokenizer (`EmbeddingDriver.count_tokens()` — only `LocalSentenceTransformerDriver` implements this, via its raw HuggingFace tokenizer, not `model.tokenize()`, which was confirmed empirically to already truncate) and, for any chunk that actually overflows, splits it into **balanced** pieces — a hard guarantee, not an estimate. It first estimates how many pieces are needed (`ceil(total_tokens / max_seq_length)`) and aims each one at an even share of that, binary-searching the real token count per piece to stay correct. Greedily maxing out each piece up to the hard limit instead (the first version of this) sounds optimal but isn't: a chunk only slightly over the limit would produce one full-size piece and a near-empty straggler — almost useless as its own embedding, too little semantic content for retrieval to ever match it well. Falls back to `warn` (with a warning explaining why) if the active driver can't report real token counts, e.g. `EMBEDDING_DRIVER=openai`.
- **`warn`** (legacy fallback) estimates the token count using an approximate `WORDS_PER_TOKEN` ratio (default `0.75`, an English average) and **warns** — via `warnings.warn`, not an error — when a chunk is *likely* to get truncated. This ratio is only an approximation: subword tokenizers typically produce *more* tokens than words, and morphologically rich languages like Hungarian tend to tokenize *worse* (fewer words per token) than the English-based default. It doesn't correct anything — the chunk is stored and silently truncated at embed time regardless. See `docs/decisions.md` for the empirical comparison that led to `split` replacing this as the default.

Current defaults — `CHUNK_SIZE=250` words, `CHUNK_OVERLAP=30` words — were tuned to fit comfortably within the 512-token limit of the default embedding model even under dense tokenization.

Run `uv run python scripts/eval_cli.py inspect` (uses `TEST_DOC_PATH` by default, or pass a path — PDF or Markdown) to see all of this at once for your own documents: one table, every `PDF_EXTRACTION_MODE` x `CHUNKING_STRATEGY` x `CHUNK_OVERFLOW_STRATEGY` combination as its own row, each with a bar showing that row's *worst* chunk against the model's real token limit, plus real processing time in milliseconds (model-load and other one-time import costs are explicitly warmed up beforehand so they don't unfairly inflate whichever row happens to run first) — so which combination actually needs the least correction, and whether that correction is worth its cost, is a glance, not six separate reports to compare by hand.

**Known limitation:** the `CHUNK_OVERLAP` (an absolute word count) isn't reconsidered by either strategy. If `CHUNK_SIZE` were drastically lowered to match a tight token limit, a fixed `CHUNK_OVERLAP` could become a disproportionately large fraction of it. Not addressed yet.

## Supported Formats, Extraction Mode & Chunking Strategy

`add_document()` and `add_directory()` accept PDF (`.pdf`) and Markdown (`.md`/`.markdown`) files — the format is detected from the extension via `ingestion/extractors.py`'s `EXTRACTOR_REGISTRY` and `get_extractor()`. The system-wide list of allowed extensions can be configured in `.env` via `INGEST_EXTENSIONS` (default: `.pdf,.md,.markdown`), or overridden at runtime without restarting via the `allowed_extensions` parameter. `PDFExtractor` wraps `pdf_loader.py`'s pdfplumber-based extraction; `MarkdownExtractor` just reads the file directly — Markdown already marks its own paragraph breaks (blank lines) and structure (`#` headers), so there's no coordinate-based heuristic to run, unlike PDF.

Markdown files have no real "pages", so chunk metadata's `page_number` is instead a **header-based section index** for them (every `#`...`######` line starts a new section) — the same field, same purpose (citing roughly where in the document a chunk came from), just a different unit depending on the source format. A `#` inside a fenced code block (e.g. a Python/shell comment in a documentation example) is correctly not treated as a header.

Two independent settings control how a PDF becomes chunks, both in `.env` (Markdown ignores both — see above):

- **`PDF_EXTRACTION_MODE`** (`flat` default, or `blocks`) — how page text is turned into one document-level string. `flat` just joins pages with a space; `blocks` additionally detects paragraph breaks from word coordinates (`pdfplumber`'s `extract_words()`, a per-page median line-spacing × 1.8 threshold) and preserves them, so a structure-aware chunker can split on them. `blocks` can't detect a break that happens to fall exactly at a page boundary — coordinates reset per page, so there's nothing to compare the gap against across it. Accepted v1 limitation, since `blocks` is opt-in (not the default) — worth revisiting if it ever matters in practice.
- **`CHUNKING_STRATEGY`** (`word` default, or `langchain`) — how the resulting text becomes chunks. `word` is the original sliding word-count window (`CHUNK_SIZE`/`CHUNK_OVERLAP`), unchanged in behavior. `langchain` uses `langchain_text_splitters.RecursiveCharacterTextSplitter` to try paragraph, then line, then sentence, then word boundaries in order — keeping whole paragraphs/sentences together whenever they fit, instead of cutting at a fixed word count regardless of structure.

Both dimensions are independent and combinable (e.g. `blocks` + `langchain` for the most structure-aware result on well-formatted PDFs) and default to the original behavior for backward compatibility.

**A design tradeoff worth knowing about `langchain`**: each piece it returns needs to be mapped back to a word-index (for the `page_number` metadata), via character-offset tracking rather than plain word-counting — `RecursiveCharacterTextSplitter` defaults to leaving a bare leftover separator at the start of the *next* piece when it splits at `". "`/`", "` (e.g. `". word6 word7"` instead of `"word6 word7"`), which breaks simple word-counting. This driver sets `keep_separator=False`, confirmed empirically to make every such split land cleanly on a word boundary — except one remaining edge case: a single "word" longer than the entire chunk budget forces the splitter's last-resort empty-string separator, which can still land mid-word. A stricter fix (an `assert` on the word-boundary assumption, raised in a code review) was considered and rejected: it would turn a rare, low-impact inaccuracy — the affected chunk's `page_number` could be off by one — into a hard crash during ingestion. The accepted tradeoff is the same one `_split_oversized_text` already documents for the `word` strategy: word-level granularity can't split a single oversized "word" any finer, so it's left as-is rather than crashing or guessing.

**The real fix underneath both**: chunking used to run **per page** (`chunk_pages()`), with no overlap between pages — a paragraph that happened to span a page break was silently split into two truncated, unrelated chunks. `ingest.py` now concatenates the whole document first (`pdf_loader.extract_document_text()`) and chunks that (`chunker.chunk_document()`) — there's no page loop left to truncate anything at a page boundary. `chunk_pages()` itself is unchanged and still used by `eval_cli.py extract`-adjacent tooling and its own tests.

Since a chunk's words can now come from more than one page, `page_number` in its metadata is assigned by **majority vote** (whichever page contributed the most words) rather than a page range — simpler, backward-compatible with the existing single-number metadata shape, and the rare off-by-one-page citation is a negligible cost next to not having to change the prompt template and retrieval display for a page *range*.

## Retrieval: Hybrid Search, Reranking & Quality Evaluation

The retrieval side went through the same "own numbers, not just intuition" treatment as the chunking side above. This section is a direct answer to four things worth knowing about it: how search combines vector and keyword matching, how (and whether) results get reranked, how quality is actually measured, and what happens when nothing relevant exists.

### Hybrid search (vector + keyword, fused by rank)

Pure cosine-similarity search (the original design) misses one common case: an exact name, number, or code-like token can score poorly on embedding similarity even when it's a perfect keyword match — the embedding "smooths over" exact tokens that a keyword search finds trivially. `query/retrieval.py`'s `retrieve_chunks()` runs **both**, by default:

- `VectorStore.search()` — pgvector cosine similarity (unchanged).
- `VectorStore.search_fulltext()` — Postgres full-text search over a generated `tsvector` column (`migrations/0002_add_fulltext_search.py`), using the `simple` text-search configuration deliberately, not `english`/`hungarian` — the corpus mixes both languages, and a single language-specific configuration (with its stemming and stopword list) would only serve one of them well.

The two ranked lists are combined with **Reciprocal Rank Fusion** (`query/hybrid.py`'s `reciprocal_rank_fusion()`): every chunk's fused score is `Σ 1/(rank + k)` across whichever list(s) it appears in (`k=60`, the standard default). RRF fuses by **rank position**, not raw score — cosine similarity (0–1) and `ts_rank` (unbounded) live on incompatible scales, so averaging or weighting the raw numbers directly would be comparing apples to oranges. This is the same technique Elasticsearch/OpenSearch's built-in hybrid search uses: simple, no training, no extra model.

**A real bug this surfaced**, found while building the eval script below, not by inspection: `search_fulltext()` originally passed the raw question straight into `websearch_to_tsquery('simple', question)`. Because `simple` has no stopword list (that's exactly why it was chosen — see above), every word of the question — including grammar words like "milyen"/"used"/"is" — became a **mandatory** term (`websearch_to_tsquery` ANDs bare words together). A real chunk almost never contains a question's grammar words verbatim, so keyword search was silently returning **zero results for nearly every natural-language question**, undetected until the eval script's real numbers showed `0 keyword result(s)` on every single run. The fix: the question's words are OR-joined (`" or ".join(query_text.split())`) before being passed to `websearch_to_tsquery`, so a chunk matching *any* of the question's content words now contributes to the fusion, ranked by how many/how prominently they matched. Covered by both a unit test (asserts the OR-joined string reaches the query) and a DB test (a real sentence full of grammar words that would have failed pre-fix).

Which retrieval path runs is itself a Strategy (`query/retrieval.py`'s `RetrievalStrategy` ABC, same shape as every other driver/strategy in this project), controlled by `RETRIEVAL_STRATEGY` (`.env`, default `hybrid`): `hybrid` (`HybridRetrievalStrategy`) is everything described above; `vector` (`VectorRetrievalStrategy`) skips keyword search and fusion entirely, reproducing the pre-hybrid-search behavior exactly (same `min_score` filtering, same ordering). Kept as a real, selectable strategy rather than a one-off comparison hack specifically so `scripts/eval_cli.py eval` (`make eval`) measures the actual production code path, not a hand-rolled stand-in that could quietly drift out of sync with it. In practice there's little reason to prefer `vector` day-to-day — hybrid search only ever adds recall on top of it, at negligible extra cost (one more indexed Postgres query and a pure fusion function, no model involved) — its main use is exactly that eval/debug comparison.

### Reranking (optional, off by default)

Hybrid search produces a wide, cheap candidate pool (`RETRIEVAL_CANDIDATE_POOL_SIZE`, default 20). `drivers/reranker.py` adds an optional second stage: a **cross-encoder** scores each `(question, chunk)` pair *jointly* (not independently, like an embedding) — more accurate, but too expensive to run over a whole corpus, so it only ever reranks that already-small candidate pool. This is the standard "retrieve-then-rerank" architecture.

Model choice: **`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`**, a multilingual cross-encoder (MS MARCO machine-translated into 14 languages), instead of the far more common English-only `cross-encoder/ms-marco-MiniLM-L-6-v2` — this project's content and embedding model are both multilingual (Hungarian + English), and an English-only reranker would be a regression on Hungarian content specifically. Verified empirically, not just by model-card description: given the Hungarian sentence *"Magyarországon a személyi jövedelemadó (SZJA) mértéke egységesen 15 százalék"* against the question *"Mennyi az SZJA kulcsa Magyarországon?"*, it scored **+6.39** — clearly separated from an unrelated Hungarian sentence (**-6.13**) and an unrelated English one (**-9.24**).

Controlled by `RERANKER_DRIVER` (`.env`, default `none`) — a Strategy pattern like every other driver in this project. `none` (`NoopRerankerDriver`) passes the RRF-fused order through unchanged; `cross_encoder` (`CrossEncoderRerankerDriver`) reorders by the model's score. Defaulting to `none` was a deliberate choice, not a placeholder: the cross-encoder model must be downloaded and loaded (real latency, real memory) on every process that queries the knowledge base, and hybrid search's own fusion is already a reasonable ranking on its own — reranking is an *optional* quality lever, not a required part of the pipeline.

### How quality is measured

`scripts/eval_cli.py eval` (`make eval`) + `tests/data/eval_questions.json` — **25 bilingual questions** (19 answerable, 6 deliberately unanswerable) across three committed fixtures (`tests/data/sample.md`, `tests/data/sample.pdf`, `tests/data/sample_hu.md` — an Hungarian enterprise IT policy). It runs every question through `retrieve_chunks()` with each `RetrievalStrategy` swapped in explicitly — **vector-only** (`VectorRetrievalStrategy`) and **hybrid+rerank** (`HybridRetrievalStrategy`, with whatever `RERANKER_DRIVER` is currently configured) — through the exact same production code path, not a hand-rolled duplicate, and reports:

- **Passage Hit@1**: did the passage containing the expected gold fact land at rank 1?
- **Passage Recall@k**: did it land anywhere in the top-k?
- **Passage MRR** (Mean Reciprocal Rank): inverse rank of the first gold passage.
- **Fallback rate**: for the unanswerable questions, did the retrieval-layer relevance gate correctly return nothing?

Each question carries a `expected_text_contains` gold label (a specific phrase expected verbatim in the answer, e.g. `"pgvector"`, `"€49/month"`, `"150 000 Ft"`) — the eval script uses these for passage-level matching, not just file-level.

Run it with:

```bash
# Default (RERANKER_DRIVER=none — RRF fusion only, no cross-encoder):
make eval
# or: uv run python scripts/eval_cli.py eval

# With cross-encoder reranker:
make eval-rerank
# or: RERANKER_DRIVER=cross_encoder uv run python scripts/eval_cli.py eval

# Controlled corpus (only the 3 committed fixtures, no other documents):
AGENT_ENV=test RERANKER_DRIVER=cross_encoder uv run python scripts/eval_cli.py eval
```

It's safe to run against any configured `DATABASE_URL`, including a populated dev database: it never deletes anything, only adds the fixtures if they're not already present (`VectorStore.has_chunks_from_source`), and prints which database it's about to touch.

**Current benchmark numbers** (`AGENT_ENV=test`, controlled corpus):

| Config | Hit@1 | Recall@k | Passage MRR | Fallback |
|---|---|---|---|---|
| `vector-only` | 0.89 | 0.95 | 0.92 | 0.00 |
| `hybrid+rerank (cross_encoder)` | **0.95** | 0.95 | **0.95** | **1.00** |

Language breakdown for `hybrid+rerank (cross_encoder)`:
- **EN** (10 ans, 3 unans): Hit@1 = 0.90, Recall@k = 0.90, Fallback = 1.00
- **HU** (9 ans, 3 unans): Hit@1 = **1.00**, Recall@k = 1.00, Fallback = 1.00

**Honesty about what these numbers mean**: three documents, 25 questions — the corpus is tiny, so these compare our *configurations against each other*, not against a statistically meaningful external benchmark. What's meaningful here: the cross-encoder's `Fallback = 1.00` vs. vector-only's `Fallback = 0.00` (6/6 vs. 0/6 unanswerable queries correctly rejected at the retrieval layer) — this is a real, structural difference, not noise.

**A more interesting, honest finding** came from the fallback-rate metric: it measured **0.33** — 1 of the 3 deliberately unanswerable questions was correctly caught by the retrieval-layer gate alone, the other 2 weren't. Inspecting the real scores explained why: a question that stays *topically* on-subject but asks for a fact the document never states can still score above `RETRIEVAL_MIN_SCORE` (0.25) — e.g. 0.438 for "what is the founder's phone number?" and 0.370 for "what is the project's annual revenue?" — because cosine similarity reflects "is this about the same topic", not "does this contain the specific fact asked for". The one that *was* caught ("what programming language is the backend written in?", scoring 0.249) shows how close this can run either way — a hair's width under the 0.25 threshold, not a clean rejection. That's a genuine limitation of similarity-threshold gating, not a bug, and it's exactly why the system doesn't rely on that gate alone (see below).

### What happens when there's no reliable source

Two independent layers, not one:

1. **Retrieval-layer gate** (`query/retrieval.py`'s `_passes_relevance_gate`): if *nothing* in the vector-search candidate pool clears `RETRIEVAL_MIN_SCORE`, `query_knowledge_base()` returns `NO_RESULTS_MESSAGE` immediately — no LLM call at all. This deliberately checks pure vector cosine similarity only, never the fused RRF/reranker score: cosine similarity lives on a calibrated [0, 1] scale with a meaningful "too-low-to-be-relevant" interpretation (the original motivation for `RETRIEVAL_MIN_SCORE=0.25`). This catches questions genuinely unrelated to anything in the knowledge base.
2. **Cross-encoder reranking gate** (`RERANKER_DRIVER=cross_encoder`, optional): when enabled, `CrossEncoderRerankerDriver` scores each `(question, chunk)` pair jointly and discards any chunk scoring below `RERANKER_MIN_SCORE=-2.0` on the logit scale. This is a *second* gate that fires *after* the cosine gate — it operates on the already-filtered candidate pool, not the raw corpus. Its logit scale (unbounded, centered around 0) has a natural "irrelevant" region confirmed empirically: relevant chunks score +2 to +6, clearly irrelevant ones score -3.5 to -9. With `cross_encoder` enabled, the measured `Fallback = 1.00` (6/6 unanswerable queries correctly rejected at the retrieval layer, zero reaching the LLM), vs. `Fallback = 0.00` without it — a 6-chunk saving per irrelevant query with no LLM call at all.

  **Caveat:** `-2.0` was calibrated on the same 6 unanswerable questions that the `Fallback = 1.00` number above is then measured against — that's training-set accuracy, not a demonstrated generalization to unseen questions. Validating it properly needs a larger, independent eval set that wasn't used for tuning.
3. **Prompt-level grounding instruction** (`drivers/llm.py`'s `_build_prompt`): the system prompt explicitly instructs the model to answer strictly from the provided excerpts and respond with an exact "I could not find this information in the provided documents" if the excerpts don't answer the question. This catches the case the eval script's fallback-rate finding demonstrated above — a topically-relevant chunk that still doesn't contain the specific fact asked for — which a similarity threshold structurally cannot distinguish from a genuine answer.

**Verifying layers 2 and 3 work — `--with-llm`**: `scripts/eval_cli.py eval --with-llm` (`make eval-llm`) generates a real answer for every question under both configurations, side-by-side, with full (un-truncated) text. For answerable questions, it checks whether the expected gold fact (`expected_text_contains`) is verbatim in the answer (`✅ GOLD FACT MATCH` / `ℹ️ ANSWERED (FACT NOT FOUND)`). For unanswerable ones, it distinguishes between retrieval-layer rejection (`🛡️ RETRIEVAL REJECTED` — 0 chunks passed, 0 LLM calls) and prompt-layer decline (`🛡️ PROMPT DECLINED`), vs. a potential hallucination (`🚨 POTENTIAL HALLUCINATION`). A final **LLM Generation Benchmark Scorecard** summarises gold fact retention %, safe decline rate, API calls made, and average latency across both configurations.

The default LLM model is pinned to `google/gemini-3.1-flash-lite` (via OpenRouter) — a fixed, non-`:free` model — to ensure `--with-llm` results are reproducible. Using `openrouter/free` (auto-routed to a random available model) is explicitly not recommended for this kind of measurement: different models on different calls makes the decline-rate numbers meaningless as comparative evidence.

## Local Diagnostic & Evaluation Scripts

Three hand-runnable `eval_cli.py` subcommands, no test framework involved — point them at a real document (or the committed fixtures) and read the output. Each one uses the exact same production code the real pipeline does (extractors, `chunk_document()`, `retrieve_chunks()`), never a reimplementation, so what they show is what `add_document()`/`query_knowledge_base()` would actually do.

| Subcommand | What it's for | Run it |
|---|---|---|
| `eval_cli.py extract` | Sanity-check a document *before* ingesting it — did text actually come out, grouped by page/section? Catches e.g. a scanned (image-only) PDF with no text layer early. | `uv run python scripts/eval_cli.py extract /path/to/file.pdf` |
| `eval_cli.py inspect` | Compare every `PDF_EXTRACTION_MODE` x `CHUNKING_STRATEGY` x `CHUNK_OVERFLOW_STRATEGY` combination for one document side by side — chunk count, token size vs. the embedding model's real limit (as a bar), processing time. See "Chunking & Token Limits" above. | `uv run python scripts/eval_cli.py inspect /path/to/file.pdf` |
| `eval_cli.py eval` | Compare **vector-only vs. hybrid+rerank** retrieval quality — Recall@k, MRR, Fallback rate, plus a per-question breakdown and (with `--with-llm`) real generated answers. See "How quality is measured" above. | `uv run python scripts/eval_cli.py eval` |

All three fall back to `TEST_DOC_PATH` (`.env`) when no path is given, except `eval`, which always runs against the two committed fixtures (`tests/data/sample.md`/`sample.pdf`) — see its own "Which database?" note above for why it's safe to run against a real, populated database.

**`eval_cli.py eval`'s per-question breakdown**, specifically: the aggregate Recall@k/MRR/Fallback rate numbers can land on identical values for both configurations purely because the corpus is small — that hides whether the two strategies actually behave differently on any *individual* question. Every run also prints a row per question, `vector` vs. `hybrid`, with a `<- differs` marker wherever the two disagree, so a difference is visible even when the averages coincide.

**`--with-llm`**: none of the above touches the LLM — Recall@k/MRR/Fallback rate are all retrieval-layer-only, deliberately, to stay fast and free to run. Passing `--with-llm` additionally generates a real answer (real network call, `LLM_DRIVER`) for every question under both configurations side-by-side, with full un-truncated text. Each answer is tagged with its outcome: `✅ GOLD FACT MATCH` / `ℹ️ ANSWERED (FACT NOT FOUND)` for answerable questions (verified against `expected_text_contains` gold labels), and `🛡️ RETRIEVAL REJECTED` / `🛡️ PROMPT DECLINED` / `🚨 POTENTIAL HALLUCINATION` for unanswerable ones. A final **LLM Generation Benchmark Scorecard** at the end summarises gold fact retention rate, safe decline breakdown by layer, real API calls made (calls saved by retrieval-layer rejection), and average latency. See "What happens when there's no reliable source" above for the interpretation.

## MCP Server

`mcp_server.py` exposes `add_document`/a retrieval tool to any [MCP](https://modelcontextprotocol.io)-compatible host — most directly, Claude Desktop, so a document can be ingested and the knowledge base searched from inside a normal chat. Doesn't affect the retrieval design above; this is a separate, personal-use integration.

**Transport is stdio, not HTTP** — the host (Claude Desktop) spawns `mcp_server.py` itself and talks over stdin/stdout, which is what "add a local MCP server" means and needs no extra infrastructure. A network-reachable version (FastAPI + Docker) would only matter for a *remotely* accessible server and isn't built.

**The retrieval tool is named `search_knowledge_base`, not `query_knowledge_base`, and deliberately doesn't call this project's own `LLM_DRIVER`** — it wraps `query.retrieval.retrieve_chunks()` directly and returns the raw excerpts (`content`/`source_file`/`page_number`), not a generated answer. The point: the MCP *host's own model* (e.g. whatever Claude Desktop is already running) writes the final grounded answer from those excerpts, in its own conversation turn — no extra API call or cost to this project at all. The grounding instruction ("answer only from these excerpts, say so if they don't answer the question") is carried in the tool's `description`, the same mechanism `drivers/llm.py`'s own prompt uses, just addressed to the host model instead of `LLM_DRIVER`.

**`search_knowledge_base` alone wasn't reliable in practice — `/my-docs` is the fix.** Tested for real in Claude Desktop: asked about "the Player Central MVP's technology stack," and got back a detailed, entirely wrong answer (a Laravel/Nuxt-4/Stripe+Billingo stack) — nothing like the actual fixture content. The model hadn't used the tool's results at all; it answered from its own memory of an unrelated, same-named real project. A `tools` call is the *model's own judgment call*, and that judgment isn't trustworthy enough to rely on alone, even with a strongly-worded description. The fix is a `my-docs` **prompt** (a different MCP primitive from `tools`) — `/my-docs <question>` in Claude Desktop is *user*-invoked, so the instruction to call `search_knowledge_base` and answer only from its results becomes mandatory, not a suggestion the model can talk itself out of. Confirmed working afterward: forcing tool-only mode surfaced an honest, grounded, partial answer instead (see `docs/decisions.md`'s bug 4 for what that partial answer actually revealed) rather than a confident wrong one — though a second real test showed Claude Desktop's own memory feature *still* contributed alongside a correct, transparent tool call (labeled separately, not conflated — but still there). Tightened `/my-docs`'s wording further as a result ("no other tool", "every claim traceable to a specific excerpt") — a best-effort improvement, not a guarantee, since that memory feature is client-side and outside what any MCP prompt can necessarily override. For consistency, `drivers/llm.py`'s own system prompt (used by `query_knowledge_base()`'s `LLM_DRIVER` call, a separate code path from MCP entirely) got the same tightening: explicitly forbids outside/training knowledge and requires saying what's missing on a partial match, not just citing sources.

Five real bugs surfaced while verifying this against a real host (Claude Desktop) rather than by inspection alone — a stdio/stdout logging conflict, an `mcp install`-generated launch command that didn't resolve this project's dependencies, a missing re-ingestion guard that silently duplicated chunks and crowded out relevant results, the retrieval symptom that guard's absence caused, and a swallowed error message on the fix's own guard. See `docs/decisions.md` for the full writeup of each.

Register it with Claude Desktop: `make mcp-install` (reads `LLM_API_KEY`/`DATABASE_URL`/etc. from `.env` into the registered entry, since Claude Desktop spawns the server with none of the current shell's environment, and patches the launch command per `docs/decisions.md`'s bug 2), then fully quit and reopen Claude Desktop. Try `/my-docs <question>` for a forced, tool-only answer, or just ask normally and hope the model calls `search_knowledge_base` on its own (see the "`/my-docs` is the fix" note above for why that's not guaranteed). Try it standalone first with `make mcp-dev`, which opens the MCP Inspector for calling either tool (or the prompt) by hand.

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
- [x] Hybrid search (vector + keyword, RRF-fused), optional cross-encoder reranking, and a retrieval-quality eval script — see "Retrieval" above (beyond the original steps, added after a structured code review)
- [x] Step 6 — Function-calling agent (`add_document` vs `query_knowledge_base`) — see `agent.py`
- [x] Step 7 *(stretch)* — Wrap tools as an MCP server — see `mcp_server.py` and "MCP Server" below (stdio transport only; a network-reachable version via FastAPI/Docker is a possible future step, not built)

## Further Reading

`docs/decisions.md` has the full engineering-decision and bug-log history behind this README's current-state description — including the structured IR/ML code review that drove the caching, chunking, embedding, reranking, metadata-filtering, and eval-suite work reflected above, and the MCP integration bugs referenced in "MCP Server" above.

`docs/architecture.md` has a diagram view of the same system — the ingestion and retrieval pipelines end to end, and a table of every Strategy/Driver and which `.env` setting selects it — without the prose detail or specific numbers this README carries.
