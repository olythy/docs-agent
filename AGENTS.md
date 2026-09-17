# Project Instructions & Conventions

## Language Conventions
- **Codebase Language**: All source code, docstrings, inline comments, variable/class names, log messages, exception messages, and code-level documentation must be written strictly in **English**.
- **Chat Language**: Interaction and communication with the user in chat is conducted in **Hungarian**.

## Architecture & Code Guidelines
- Follow Driver / Strategy pattern for modularity (e.g. Embedding drivers: Local vs OpenAI).
- Centralized configuration via `Settings` class with sensible defaults and `.env` overrides.
- Use strict typing, type annotations, and Ruff for linting/formatting.

### Design philosophy: SRP over DDD, flat layout over `src/`

Decided 2026-09-17 after explicit discussion — don't reintroduce these without discussing again.

- **The Single Responsibility Principle applies at the class/module level, not the file level.** Grouping a Strategy-pattern ABC, its concrete implementations, and its factory function in one file is normal and encouraged here, as long as they share one cohesive concept and one reason to change (e.g. `drivers/embedding.py`, `drivers/llm.py`, `ingestion/chunker.py`'s `ChunkOverflowStrategy`, and `migrations/base.py` alongside the migration files it supports). Splitting these into one-class-per-file would not improve SRP — it would just scatter one responsibility across more files.
- **No DDD tactical patterns** (aggregates, value objects, domain events, bounded contexts). This project's domain is a linear pipeline (extract → chunk → embed → store; embed → retrieve → answer) with no complex business invariants to protect — DDD ceremony would be disproportionate to the problem size.
- **No `src/` layout.** The project is intentionally not an installable/distributable package (`uv init --bare`, no build-system) — it runs in place via `uv run`. `src/` layout solves an import-shadowing problem specific to installed packages, which doesn't apply here.
- **Orchestrator functions must stay thin.** `add_document()` (`ingestion/ingest.py`) and `query_knowledge_base()` (`query/retrieval.py`) are the two agent-facing "tool" entry points (see `PLAN.md`) and must only coordinate: call the chunker, the embedding driver, the vector store, the LLM driver. They must never build or execute SQL directly — that belongs in the data-access layer:
  - `db.py` — only the Postgres connection factory (`get_connection()`). Nothing else.
  - `store.py` — the `VectorStore` class: owns all `document_chunks` persistence (save/search). Any change to how chunks are stored or queried belongs here, not in `ingest.py`/`retrieval.py`.
  - `migrations/base.py` — the `Migration` ABC stays inside `migrations/`, alongside the migrations that implement it (same one-cohesive-concept reasoning as the Strategy-pattern files above).
- **`docker/init-test-db.sql` only creates the `docs_agent_test` database — nothing else.** No tables, no `CREATE EXTENSION`. Schema/extension setup stays owned exclusively by `migrations/`, so there's one canonical source of truth for the schema regardless of which database (local Docker or a managed Postgres) it's applied to.

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
- `down()` must be safe to call even if `up()` was never applied (e.g. `DROP TABLE IF EXISTS`) — `scripts/migrate.py fresh` calls `down()` on every migration file unconditionally.
- Migrations are tracked by filename stem (without extension) in the `schema_migrations` table. Never rename an already-applied migration file without also reconciling its `schema_migrations` row.
- Use `make make-migration name=<snake_case_name>` to scaffold a new one — don't hand-roll the filename/numbering.

### Scripts (`scripts/`)
- Each script must include a module-level docstring explaining how to run it and what it does.

### README.md
- The **Roadmap** section must be kept in sync with `PLAN.md` after each completed step.
- The **Architecture** section must reflect the actual directory structure at all times.

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
- If a script needs a file path for testing (e.g. a sample document), it must read it from `settings.TEST_PDF_PATH` (`.env`) or from a CLI argument — never as a Python literal in the source.
- The `.env` file is already git-ignored. Keep it that way. Never commit `.env` itself.
