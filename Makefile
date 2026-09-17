.PHONY: docker-up docker-down docker-down-clean \
        db-migrate db-migrate-test db-flush db-refresh setup \
        migrate-status migrate-install migrate-fresh migrate-rollback migrate-reset migrate-refresh \
        make-migration test lint

docker-up:
	docker compose up -d --wait

docker-down:
	docker compose down

# Also deletes the data volume — full reset, re-runs docker/init-test-db.sql
# on next docker-up.
docker-down-clean:
	docker compose down -v

db-migrate:
	uv run python scripts/migrate.py up

# Runs migrations with AGENT_ENV=test, so config.py loads .env.test on top
# of .env and settings.DATABASE_URL resolves to the test database for this
# one process — same mechanism `test` below uses.
db-migrate-test:
	AGENT_ENV=test uv run python scripts/migrate.py up

# One-shot onboarding: start the local Postgres, migrate both databases.
setup: docker-up db-migrate db-migrate-test
	@echo "Ready — dev and test databases are both migrated."

db-flush:
	uv run python scripts/db_flush.py

# Empty document_chunks, then re-apply any pending migrations — a clean,
# schema-up-to-date slate in one command. Runs in this order (flush before
# migrate) because `make` executes prerequisites left to right.
db-refresh: db-flush db-migrate

# Laravel-artisan-style migration commands (colon names like `migrate:fresh`
# don't work in Make — `:` is the target/prerequisite separator — so these
# use hyphens instead).
migrate-status:
	uv run python scripts/migrate.py status

migrate-install:
	uv run python scripts/migrate.py install

migrate-fresh:
	uv run python scripts/migrate.py fresh

migrate-rollback:
	uv run python scripts/migrate.py rollback

migrate-reset:
	uv run python scripts/migrate.py reset

migrate-refresh:
	uv run python scripts/migrate.py refresh

# Usage: make make-migration name=add_foo_column
make-migration:
	uv run python scripts/make_migration.py $(name)

# No db-migrate-test prerequisite here on purpose: schema setup is handled
# by a session-scoped autouse pytest fixture (tests/db/conftest.py) instead
# of a Make-level dependency, so the test database is ready regardless of
# how pytest gets invoked — this target, a bare `AGENT_ENV=test uv run
# pytest`, or an IDE's "run test" button, which bypasses Make entirely.
test:
	AGENT_ENV=test uv run pytest -v

lint:
	uv run ruff check .
