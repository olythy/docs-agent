.DEFAULT_GOAL := help

.PHONY: help docker-up docker-down docker-down-clean \
        db-migrate db-migrate-test db-flush db-refresh setup \
        migrate-status migrate-install migrate-fresh migrate-rollback migrate-reset migrate-refresh \
        make-migration test lint

# Self-documenting: every target's `## ` comment is both its Makefile
# documentation and its `make help` output — one source, so it can't drift
# out of sync the way a hand-maintained list (e.g. README's old Makefile
# table) can.
help: ## Show this list of commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

docker-up: ## Start local Postgres (dev+test), wait until healthy
	docker compose up -d --wait

docker-down: ## Stop the local Postgres container, keep its data
	docker compose down

# Also deletes the data volume — full reset, re-runs docker/init-test-db.sql
# on next docker-up.
docker-down-clean: ## Stop the container AND delete its data (full reset)
	docker compose down -v

db-migrate: ## Migrate DATABASE_URL (.env) — the dev database
	uv run python scripts/migrate.py up

# Runs migrations with AGENT_ENV=test, so config.py loads .env.test on top
# of .env and settings.DATABASE_URL resolves to the test database for this
# one process — same mechanism `test` below uses.
db-migrate-test: ## Migrate DATABASE_URL from .env.test — the test database
	AGENT_ENV=test uv run python scripts/migrate.py up

# One-shot onboarding: start the local Postgres, migrate both databases.
setup: docker-up db-migrate db-migrate-test ## One-shot onboarding: start Postgres + migrate both DBs
	@echo "Ready — dev and test databases are both migrated."

db-flush: ## Truncate document_chunks (rows only, keeps the schema)
	uv run python scripts/db_flush.py

# Empty document_chunks, then re-apply any pending migrations — a clean,
# schema-up-to-date slate in one command. Runs in this order (flush before
# migrate) because `make` executes prerequisites left to right.
db-refresh: db-flush db-migrate ## Empty the table, then re-apply pending migrations

# Laravel-artisan-style migration commands (colon names like `migrate:fresh`
# don't work in Make — `:` is the target/prerequisite separator — so these
# use hyphens instead).
migrate-status: ## Show applied vs. pending migrations
	uv run python scripts/migrate.py status

migrate-install: ## Create the schema_migrations tracking table only
	uv run python scripts/migrate.py install

migrate-fresh: ## Revert everything, drop tracking, re-apply from scratch
	uv run python scripts/migrate.py fresh

migrate-rollback: ## Revert the most recently applied migration batch
	uv run python scripts/migrate.py rollback

migrate-reset: ## Revert every applied migration
	uv run python scripts/migrate.py reset

migrate-refresh: ## migrate-reset then db-migrate
	uv run python scripts/migrate.py refresh

make-migration: ## Scaffold a new migration file — usage: make make-migration name=add_foo_column
	uv run python scripts/make_migration.py $(name)

# No db-migrate-test prerequisite here on purpose: schema setup is handled
# by a session-scoped autouse pytest fixture (tests/db/conftest.py) instead
# of a Make-level dependency, so the test database is ready regardless of
# how pytest gets invoked — this target, a bare `AGENT_ENV=test uv run
# pytest`, or an IDE's "run test" button, which bypasses Make entirely.
test: ## Run the full test suite (unit + tests/db/)
	AGENT_ENV=test uv run pytest -v

lint: ## Run ruff
	uv run ruff check .
