# Project Instructions & Conventions

## Language Conventions
- **Codebase Language**: All source code, docstrings, inline comments, variable/class names, log messages, exception messages, and code-level documentation must be written strictly in **English**.
- **Chat Language**: Interaction and communication with the user in chat is conducted in **Hungarian**.

## Architecture & Code Guidelines
- Follow Driver / Strategy pattern for modularity (e.g. Embedding drivers: Local vs OpenAI).
- Centralized configuration via `Settings` class with sensible defaults and `.env` overrides.
- Use strict typing, type annotations, and Ruff for linting/formatting.

### Model selection: verify language/domain fit before adopting, not just "it's the default"

Before adopting *any* AI model (embedding, reranker, LLM) for a specific driver, verify via the provider's own current documentation that its documented capabilities actually match this project's real content — don't assume a provider's "default" or "recommended for RAG" choice is adequate without checking the fine print, and don't assume a benchmark number from one language transfers to another.

Confirmed costly on 2026-10-03: `EMBEDDING_DRIVER=vertex` adopted `text-embedding-005` because it's Vertex AI's default/recommended embedding model for RAG corpora — but Google's own docs state every text embedding model is *evaluated* on English text, and `text-multilingual-embedding-002` is the model actually recommended for multilingual use. This went undiagnosed through a full day of live measurement on this project's all-Hungarian legal corpus, during which a real brute-force similarity check (the *correct* document scoring *lower* than several wrong ones) was reasonably read as a fundamental "embeddings can't distinguish near-duplicate legal documents" limitation — a live A/B re-test with the multilingual model on the same two cases instead put the correct document(s) at rank #1 (and #1+#2, for a two-document question), cleanly separated from the rest. The real cause was never retrieval architecture; it was running an English-only model against Hungarian text.

**How to apply:** when choosing or changing `EMBEDDING_MODEL`/`RERANKER_MODEL`/`LLM_MODEL` for any driver, check the provider's current docs for that *specific* model's language/domain support before trusting it — via the Context7 MCP convention (see the global instructions), not assumption. For embedding/reranker changes specifically, run a small, cheap A/B similarity check against a few already-known test cases (see `docs/decisions.md`'s entries for examples of this pattern) *before* committing to a model switch that requires a full corpus re-ingestion.

### Design philosophy: SRP over DDD, flat layout over `src/`

Decided 2026-09-17 after explicit discussion — don't reintroduce these without discussing again.

- **The Single Responsibility Principle applies at the class/module level, not the file level.** Grouping a Strategy-pattern ABC, its concrete implementations, and its factory function in one file is normal and encouraged here, as long as they share one cohesive concept and one reason to change (e.g. `drivers/embedding.py`, `drivers/llm.py`, `ingestion/chunker.py`'s `ChunkOverflowStrategy`, and `migrations/base.py` alongside the migration files it supports). Splitting these into one-class-per-file would not improve SRP — it would just scatter one responsibility across more files.
- **No DDD tactical patterns** (aggregates, value objects, domain events, bounded contexts). This project's domain is a linear pipeline (extract → chunk → embed → store; embed → retrieve → answer) with no complex business invariants to protect — DDD ceremony would be disproportionate to the problem size.
- **No `src/` layout.** The project is intentionally not an installable/distributable package (`uv init --bare`, no build-system) — it runs in place via `uv run`. `src/` layout solves an import-shadowing problem specific to installed packages, which doesn't apply here.
- **Orchestrator functions must stay thin.** `add_document()` (`ingestion/ingest.py`) and `query_knowledge_base()` (`query/retrieval.py`) are the two agent-facing "tool" entry points (see `PLAN.md`) and must only coordinate: call the chunker, the embedding driver, the vector store, the LLM driver. They must never build or execute SQL directly — that belongs in the data-access layer:
  - `db.py` — only the Postgres connection factory (`get_connection()`). Nothing else.
  - `store.py` — the `VectorStore` class: owns all `document_chunks` persistence (save/search). Any change to how chunks are stored or queried belongs here, not in `ingest.py`/`retrieval.py`.
  - `migrations/base.py` — the `Migration` ABC stays inside `migrations/`, alongside the migrations that implement it (same one-cohesive-concept reasoning as the Strategy-pattern files above).
  - Non-I/O *decisions* an orchestrator makes (e.g. `ingest.py`'s dedup/versioning branching) still belong in a pure, separately-testable function in the same file — see `ingestion/ingest.py`'s `_resolve_ingest_action()` — rather than either inlining the branching in the orchestrator or promoting it to a whole new architectural layer (see the re-discussion note below).
- **`docker/init-test-db.sql` only creates the `docs_agent_test` database — nothing else.** No tables, no `CREATE EXTENSION`. Schema/extension setup stays owned exclusively by `migrations/`, so there's one canonical source of truth for the schema regardless of which database (local Docker or a managed Postgres) it's applied to.
- **`ingestion/extractors.py`'s `Extractor` is the one deliberate exception to "Strategy is selected from `settings`."** Every other Strategy here (`EmbeddingDriver`, `AnswerDriver`, `ChunkingStrategy`, `ChunkOverflowStrategy`) is chosen from an `.env` preference. `get_extractor(file_path)` instead dispatches on the file's extension — which extractor applies is a fact about the file, not a preference, so there's nothing to configure. Don't try to "fix" this into a `DOCUMENT_EXTRACTOR` setting.
- **`models.py` holds the core, framework-free data shapes** (`ChunkMetadata`, `Chunk`, `RetrievedChunk`) shared across ingestion and retrieval — typed `dataclass`es, not Pydantic (no Pydantic dependency in this project) and not a full Entity layer. Introduced 2026-09-29 to replace an untyped `{"content": ..., "metadata": {...}}` dict that was becoming risky to extend as chunk metadata grows more complex (header paths, section breadcrumbs for the legal corpus). Postgres JSONB storage is unaffected — `to_dict()`/`from_dict()` are the only serialization boundary.

**Re-discussed 2026-09-29** (Károly proposed a full Clean Architecture migration as the project scales to a 10,000+ document legal corpus): re-affirmed the position above rather than adopting Clean Architecture's layering (Entities/Use Cases/Interface Adapters/Frameworks). The concrete complaint — `add_document()`'s dedup/versioning branching reads like business logic embedded in an orchestrator — was real and got a real fix (`_resolve_ingest_action()`, above), and the "corpus will need richer metadata" concern got a real fix too (`models.py`, above). Both were achievable as small, targeted extractions; neither needed a new architectural layer to justify them. See `docs/decisions.md`'s 2026-09-29 entry for the full reasoning, including why the existing Strategy/Driver pattern already delivers most of Clean Architecture's practical benefit (swappable backends, one clear place to change persistence) without its ceremony.

## Documentation Standards

### Python Source Code
- Every **module** must have a module-level docstring explaining its purpose and key exports.
- Every **class** must have a docstring that explains its responsibility and, for `Settings`-style classes, lists all configurable attributes with their types and defaults.
- Every **public function/method** must have a docstring with: a one-line summary, an `Args:` block (if any non-obvious parameters), and a `Returns:` or `Raises:` block where applicable.
- Use **Google-style docstrings** consistently (same style as `config.py`).
- Private helpers (`_prefixed`) need at minimum a one-line summary docstring.

### Configuration (`config.py`)
- The `Settings` class docstring is the **single source of truth** for all supported environment variables.
- Every new env variable must be documented in the `Settings` docstring **before** the corresponding field is added.
- **`AGENT_ENV` must never be set inside `.env` or `.env.test`.** It selects *which* file(s) to load, so it must come from the real process/shell environment only (`make test` sets it automatically) — putting it inside a file that's conditionally loaded based on its own value is a circular bootstrapping bug. `.env` always loads first; `.env.test` is layered on top (`override=True`) only when `AGENT_ENV=test`, and only needs to contain the keys that actually differ (mainly `DATABASE_URL`).

### Migrations (`migrations/`)
- Each migration is a Python file (e.g. `0001_create_document_chunks_table.py`) defining exactly one class that subclasses `migrations.base.Migration`, with `up()`/`down()` methods running raw SQL directly (no ORM). See `migrations/base.py`.
- Every migration file must start with a module-level docstring explaining what it does and why (e.g. "Creates the document_chunks table for pgvector RAG storage").
- `down()` must be safe to call even if `up()` was never applied (e.g. `DROP TABLE IF EXISTS`) — `scripts/db_cli.py fresh` calls `down()` on every migration file unconditionally.
- Migrations are tracked by filename stem (without extension) in the `schema_migrations` table. Never rename an already-applied migration file without also reconciling its `schema_migrations` row.
- Use `make make-migration name=<snake_case_name>` to scaffold a new one — don't hand-roll the filename/numbering.

### Scripts (`scripts/`)
- Each script must include a module-level docstring explaining how to run it and what it does.

### README.md
- The **Roadmap** section must be kept in sync with `PLAN.md` after each completed step.
- The **Architecture** section must reflect the actual directory structure at all times.

### `docs/decisions.md`
- Add a new, dated entry (newest first) for: a rejected alternative or "why not the obvious fix" decision, a bug found through real-world testing (not just unit tests), or a default/threshold changed based on a measurement — the same kinds of things `README.md`'s "Honesty about what these numbers mean"/caveat sections already model. Name the commit hash(es) the entry came from.
- Do this **in the same session the decision is made**, not as a later cleanup pass — it's easy to forget once the code change itself feels done. If a session's own summary to the user describes a "why," that's the signal to also add it here before wrapping up.
- `README.md` stays current-state-only; point to `docs/decisions.md` for the "why" rather than re-explaining it inline (see the existing pointers in "Chunking & Token Limits" and "Retrieval" for the pattern).

### `.env.example`
- Every environment variable that exists in `config.py` must appear in `.env.example` with an inline comment explaining its purpose and accepted values.
- Variables must be kept in sync: if a variable is added to `config.py`, it must be added to `.env.example` in the same PR/commit.

## Git Commit Conventions

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <short description>

[optional body]

[optional footer(s)]
```

Common types used in this project: `feat` (new behavior), `fix` (bug fix), `refactor` (no behavior change), `test` (tests only), `docs` (README/AGENTS.md/PLAN.md/docstrings only), `build` (dependency/tooling changes, e.g. `uv`, `pyproject.toml`), `chore` (repo housekeeping with no source change).

- Subject line: imperative mood, lowercase after the colon, no trailing period.
- Prefer several small, single-purpose commits over one large one — each commit should tell one part of the story, and be revertable on its own where practical.
- Body explains *why*, not *what* — the diff already shows what changed.

## Privacy & Security

- **Never hardcode personal or client-specific data in source files.** This includes real filenames, document IDs, tax identifiers, or any path fragment that could reveal personal information.
- If a script needs a file path for testing (e.g. a sample document), it must read it from `settings.TEST_DOC_PATH` (`.env`) or from a CLI argument — never as a Python literal in the source.
- The `.env` file is already git-ignored. Keep it that way. Never commit `.env` itself.
