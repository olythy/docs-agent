.PHONY: db-migrate db-flush db-refresh \
        migrate-status migrate-install migrate-fresh migrate-rollback migrate-reset migrate-refresh \
        make-migration test lint

db-migrate:
	uv run python scripts/migrate.py up

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

test:
	uv run pytest -v

lint:
	uv run ruff check .
