"""The `compute-hub-scores` command: compute each chunk's embedding-space "genericness".

Thin CLI wrapper around store.VectorStore.compute_hub_scores() -- see that
method's docstring for the mechanism (CSLS-style hubness correction) and
why it replaced an earlier, rejected hard-exclusion "boilerplate" filter
(see docs/decisions.md). Run after (re-)ingesting the corpus, since this
needs every chunk's real embedding already stored to compare against.
"""

import typer

from store import VectorStore

app = typer.Typer()


@app.command(name="compute-hub-scores")
def compute_hub_scores() -> None:
    """Compute hub_score for every chunk (see store.VectorStore.compute_hub_scores)."""
    print("Scanning document_chunks for each chunk's nearest-neighbor hub score ...")

    def _report_progress(processed: int, total: int) -> None:
        print(f"  ... {processed}/{total} chunks checked")

    updated_count = VectorStore().compute_hub_scores(on_progress=_report_progress)
    print(f"Updated hub_score for {updated_count} chunk(s).")
