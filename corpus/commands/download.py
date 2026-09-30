"""The `download` command: thin wrapper around download_court_decisions.py.

Builds the same ``DownloadConfig`` and calls the same ``run()`` that
module's own ``argparse`` CLI always did -- its internals are unchanged.
"""

import logging
import sys
from pathlib import Path
from typing import Annotated

import typer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from corpus.download_court_decisions import (
    DEFAULT_DECISION_TYPES,
    DEFAULT_KEYWORDS,
    DownloadConfig,
)
from corpus.download_court_decisions import (
    run as run_download,
)

CORPUS_DIR = Path(__file__).resolve().parent.parent

app = typer.Typer()


@app.command()
def download(
    kollegium: Annotated[
        str, typer.Option(help="Kollegium filter value, e.g. 'polgári'.")
    ] = "polgári",
    decision_types: Annotated[
        list[str],
        typer.Option(
            "--decision-types",
            help="HatarozatFajta values to search one at a time (the site allows only one per search).",
        ),
    ] = DEFAULT_DECISION_TYPES,
    keywords: Annotated[
        list[str],
        typer.Option(
            "--keywords",
            help="Keywords searched one at a time; results are merged and deduplicated across all of them.",
        ),
    ] = DEFAULT_KEYWORDS,
    year_from: Annotated[int, typer.Option()] = 2010,
    year_to: Annotated[int, typer.Option()] = 2024,
    max_documents: Annotated[
        int, typer.Option(help="Stop after downloading this many new documents.")
    ] = 10000,
    page_size: Annotated[int, typer.Option()] = 20,
    delay_min: Annotated[
        float, typer.Option(help="Minimum seconds between requests.")
    ] = 0.5,
    delay_max: Annotated[
        float, typer.Option(help="Maximum seconds between requests.")
    ] = 1.0,
    max_retries: Annotated[int, typer.Option()] = 3,
    out_dir: Annotated[
        Path,
        typer.Option(
            help="Directory to write raw/ and meta.csv into (default: corpus/)."
        ),
    ] = CORPUS_DIR,
    file_format: Annotated[
        str,
        typer.Option(
            "--format",
            help="Document format to download: 'pdf' or 'rtf' (default: rtf).",
        ),
    ] = "rtf",
) -> None:
    """Download court decisions from eakta.birosag.hu into corpus/raw/.

    Thin wrapper around corpus/download_court_decisions.py's existing
    DownloadConfig/run() -- that module's own internals are unchanged.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    if file_format not in ("pdf", "rtf"):
        raise typer.BadParameter("--format must be 'pdf' or 'rtf'")

    config = DownloadConfig(
        kollegium=kollegium,
        decision_types=decision_types,
        keywords=keywords,
        year_from=year_from,
        year_to=year_to,
        max_documents=max_documents,
        page_size=page_size,
        delay_min=delay_min,
        delay_max=delay_max,
        max_retries=max_retries,
        out_dir=out_dir,
        file_format=file_format,
    )
    run_download(config)
