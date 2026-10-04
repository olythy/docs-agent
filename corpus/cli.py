"""Unified CLI for the real-estate-law evaluation corpus: download, generate-questions, eval.

Deliberately Typer-based, not argparse -- every other CLI in this project
(``scripts/*_cli.py``, ``corpus/download_court_decisions.py``) uses argparse;
this is a scoped pilot for learning Typer before considering a project-wide
migration (see ``docs/decisions.md`` for the full reasoning).

Kept thin on purpose, following Typer's own recommended "one file per
command" layout: each command lives in its own module under
``corpus/commands/``, each with its own ``typer.Typer()`` instance, merged
here via ``add_typer()`` with no explicit name so every command stays flat
at the top level (``download``/``generate-questions``/``eval`` aren't a
nested command group, just three independent commands).

Usage::

    uv run python corpus/cli.py download [options]
    uv run python corpus/cli.py generate-questions <persona_id> [--count N]
    uv run python corpus/cli.py eval [--persona ID] [--strategy hybrid] [--only-covered]
    uv run python corpus/cli.py compute-hub-scores
    uv run python corpus/cli.py coverage
"""

import sys
from pathlib import Path

import typer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from corpus.commands import (
    compute_hub_scores,
    coverage,
    download,
    eval,
    generate_questions,
)

app = typer.Typer(help=__doc__)
app.add_typer(download.app)
app.add_typer(generate_questions.app)
app.add_typer(eval.app)
app.add_typer(compute_hub_scores.app)
app.add_typer(coverage.app)

if __name__ == "__main__":
    app()
