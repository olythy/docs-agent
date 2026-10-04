.DEFAULT_GOAL := help

.PHONY: help docker-up docker-down docker-down-clean doctor \
        db-migrate db-migrate-test db-flush db-refresh setup documents-sync extract-meta meta-coverage \
        migrate-status migrate-install migrate-fresh migrate-rollback migrate-reset migrate-refresh \
        make-migration add-document add-directory delete-document query chat \
        inspect-chunks extract-text eval eval-rerank eval-llm eval-all \
        log log-tail log-stats log-clear \
        mcp-dev mcp-install skills-install test lint lint-fix format

# Support direct positional arguments without path="...":
# e.g. `make add-document file1.pdf file2.md` or `make query "What is X?"`
SUPPORTED_CMD_TARGETS := add-document add-directory delete-document inspect-chunks extract-text query
ifeq ($(filter $(firstword $(MAKECMDGOALS)),$(SUPPORTED_CMD_TARGETS)),$(firstword $(MAKECMDGOALS)))
  CMD_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
endif

# Self-documenting: every target's `## ` comment is both its Makefile
# documentation and its `make help` output — one source, so it can't drift
# out of sync the way a hand-maintained list can.
help: ## Show this list of commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# --- Dev Environment & Containers (scripts/dev_cli.py) ---

docker-up: ## Start local Postgres (dev+test), wait until healthy
	uv run python scripts/dev_cli.py docker-up

docker-down: ## Stop the local Postgres container, keep its data
	uv run python scripts/dev_cli.py docker-down

docker-down-clean: ## Stop the container AND delete its data (full reset)
	uv run python scripts/dev_cli.py docker-clean

doctor: ## Check configuration files, Docker status, and database health
	uv run python scripts/dev_cli.py doctor

setup: ## One-shot onboarding: start Postgres + migrate both DBs
	uv run python scripts/dev_cli.py setup

lint: ## Check code style (ruff) and types (pyright)
	uv run python scripts/dev_cli.py lint

lint-fix: ## Auto-fix lint errors and format code
	uv run python scripts/dev_cli.py lint-fix

format: ## Format code with ruff
	uv run python scripts/dev_cli.py format

typecheck: ## Check types with pyright only
	uv run python scripts/dev_cli.py typecheck

# --- Database & Migrations (scripts/db_cli.py) ---

db-migrate: ## Migrate DATABASE_URL (.env) — the dev database
	uv run python scripts/db_cli.py up

db-migrate-test: ## Migrate DATABASE_URL from .env.test — the test database
	AGENT_ENV=test uv run python scripts/db_cli.py up

db-flush: ## Truncate document_chunks and documents (rows only, keeps the schema and the key catalog)
	uv run python scripts/db_cli.py flush

db-refresh: db-flush db-migrate ## Empty the table, then re-apply pending migrations

# --- Structured metadata (scripts/meta_cli.py) ---
documents-sync: ## Sync the documents table from the ingested chunks (idempotent)
	uv run python scripts/meta_cli.py sync-documents

extract-meta: ## Extract catalog metadata (LLM calls!) — usage: make extract-meta limit=50 [seed=7]
	uv run python scripts/meta_cli.py extract-meta $(if $(limit),--limit $(limit),) $(if $(seed),--seed $(seed),)

meta-coverage: ## Per-key metadata coverage (verified / absent / unverified / not tried)
	uv run python scripts/meta_cli.py coverage

migrate-status: ## Show applied vs. pending migrations
	uv run python scripts/db_cli.py status

migrate-install: ## Create the schema_migrations tracking table only
	uv run python scripts/db_cli.py install

migrate-fresh: ## Revert everything, drop tracking, re-apply from scratch
	uv run python scripts/db_cli.py fresh

migrate-rollback: ## Revert the most recently applied migration batch
	uv run python scripts/db_cli.py rollback

migrate-reset: ## Revert every applied migration
	uv run python scripts/db_cli.py reset

migrate-refresh: ## migrate-reset then db-migrate
	uv run python scripts/db_cli.py refresh

make-migration: ## Scaffold a new migration file — usage: make make-migration name=add_foo_column
	uv run python scripts/db_cli.py make $(name)

# --- Agent & Runtime (scripts/agent_cli.py) ---

add-document: ## Ingest document(s) — usage: make add-document file1.pdf [file2.md]
	uv run python scripts/agent_cli.py ingest $(if $(path),$(path),$(CMD_ARGS))

add-directory: ## Batch-ingest a directory — usage: make add-directory /path/to/dir [ext=.md]
	uv run python scripts/agent_cli.py ingest $(if $(path),$(path),$(CMD_ARGS)) $(if $(ext),--ext $(ext),)

delete-document: ## Delete document chunks by path or hash — usage: make delete-document file.pdf
	uv run python scripts/agent_cli.py ingest --delete $(if $(path),$(path),$(CMD_ARGS))

query: ## Ask a question (full pipeline, real LLM call) — usage: make query "What is X?" (or q="...")
	uv run python scripts/agent_cli.py query $(if $(q),"$(q)",$(CMD_ARGS))

chat: ## Start the interactive conversational agent REPL terminal
	uv run python scripts/agent_cli.py chat

mcp-dev: ## Run mcp_server.py under the MCP Inspector, for local testing
	uv run python scripts/agent_cli.py mcp-dev

mcp-install: ## Register mcp_server.py with Claude Desktop and patch config
	uv run python scripts/agent_cli.py mcp-install

skills-install: ## Symlink skills/ into .claude/skills/ so Claude Code discovers this project's skills
	uv run python scripts/agent_cli.py skills-install

# --- Evaluation & Diagnostics (scripts/eval_cli.py) ---

inspect-chunks: ## Compare chunking strategies for a file — usage: make inspect-chunks [path=file.pdf]
	uv run python scripts/eval_cli.py inspect $(if $(path),$(path),$(CMD_ARGS))

extract-text: ## Preview text extraction grouped by section — usage: make extract-text [path=file.pdf]
	uv run python scripts/eval_cli.py extract $(if $(path),$(path),$(CMD_ARGS))

eval: ## Run retrieval quality evaluation (vector vs hybrid)
	uv run python scripts/eval_cli.py eval

eval-rerank: ## Run retrieval evaluation with cross_encoder reranking
	uv run python scripts/eval_cli.py eval --with-rerank

eval-llm: ## Run retrieval evaluation with real LLM answer generation
	uv run python scripts/eval_cli.py eval --with-llm

eval-all: ## Run full evaluation benchmark: cross_encoder rerank + LLM generation
	uv run python scripts/eval_cli.py eval --with-rerank --with-llm

# --- Logging & Telemetry (scripts/log_cli.py) ---

log: ## Live-follow structured telemetry events — usage: make log [action=...]
	uv run python scripts/log_cli.py watch $(if $(action),--action $(action),)

log-tail: ## Print recent formatted events and exit — usage: make log-tail [n=20]
	uv run python scripts/log_cli.py tail $(if $(n),-n $(n),)

log-stats: ## Show telemetry summary (queries, dropped stopwords, rerank rates)
	uv run python scripts/log_cli.py stats

log-clear: ## Clear the telemetry log file
	uv run python scripts/log_cli.py clear

# --- Tests ---

test: ## Run the full test suite (unit + tests/db/)
	AGENT_ENV=test uv run pytest -v

# Catch-all to allow passing arguments directly after Make targets without "No rule to make target" errors
%:
	@:
