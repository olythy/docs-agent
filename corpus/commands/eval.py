"""The `eval` command: golden-set evaluation (persona-bucketed accuracy + citation correctness).

Not built yet -- corpus/data/questions.json only has a handful of seed
questions so far, and the real per-persona/citation-correctness scoring
design still needs to be worked out (see docs/decisions.md).
"""

import typer

app = typer.Typer()


@app.command()
def eval() -> None:
    """Run the golden-set evaluation (persona-bucketed accuracy + citation correctness)."""
    print("Not implemented yet -- see corpus/commands/eval.py's module docstring.")
    raise typer.Exit(code=1)
