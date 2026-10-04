"""The `meta-accuracy` command: how right is the extracted metadata, against known truth?

Compares what ``scripts/meta_cli.py extract-meta`` stored with the facts in
``corpus/meta.csv`` (the court and the decision type of every downloaded
decision). That file is a by-product of this corpus's downloader, so it is
*measurement truth only*: it must never be read by the generic metadata core.

Only the keys this corpus has truth for are compared -- ``issuing_body`` against
the court, and ``document_kind`` against the decision type, coarsened to the
three values ``meta.csv`` knows. Evidence verification (the quote is in the
document) is already guaranteed by extraction; this measures whether the *right*
value was chosen.
"""

import csv
import re
import unicodedata
from pathlib import Path

import typer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
META_CSV = PROJECT_ROOT / "corpus" / "meta.csv"

app = typer.Typer()

#: Our canonical ``document_kind`` tokens -> the coarser decision_type in meta.csv.
_KIND_TO_META = {
    "judgment": "Ítélet",
    "partial_judgment": "Ítélet",
    "interim_judgment": "Ítélet",
    "order": "Végzés",
    "other": "Egyéb",
}


def _alnum(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text.lower())
    return re.sub(
        r"[^a-z0-9]", "", "".join(c for c in folded if not unicodedata.combining(c))
    )


def agreement(
    triples: list[tuple[str, str, str]],
) -> tuple[int, int, list[tuple[str, str, str]]]:
    """Count exact agreement between (file, extracted, expected) triples.

    Args:
        triples: ``(file_name, extracted, expected)`` already normalised to the
            same vocabulary.

    Returns:
        ``(agreeing, total, disagreements)`` where each disagreement is
        ``(file_name, extracted, expected)``.
    """
    bad = [(f, got, want) for f, got, want in triples if got != want]
    return len(triples) - len(bad), len(triples), bad


@app.command(name="meta-accuracy")
def meta_accuracy(
    show: int = typer.Option(8, help="How many disagreements to list per key."),
) -> None:
    """Compare extracted issuing_body / document_kind with meta.csv's court / decision type."""
    from document_store import DocumentStore

    truth = {
        r["file_name"]: r for r in csv.DictReader(META_CSV.open(encoding="utf-8-sig"))
    }
    store = DocumentStore()

    court_pairs: list[tuple[str, str, str]] = []
    for file_name, value in store.list_values("issuing_body"):
        row = truth.get(file_name)
        if row and value.value_text:
            court_pairs.append(
                (file_name, _alnum(value.value_text), _alnum(row["court"]))
            )

    kind_pairs: list[tuple[str, str, str]] = []
    kind_distribution: dict[str, int] = {}
    for file_name, value in store.list_values("document_kind"):
        row = truth.get(file_name)
        if row and value.value_text in _KIND_TO_META:
            kind_distribution[value.value_text] = (
                kind_distribution.get(value.value_text, 0) + 1
            )
            kind_pairs.append(
                (file_name, _KIND_TO_META[value.value_text], row["decision_type"])
            )

    print(f"\n{'key':<18}{'compared':>9}{'agree':>7}{'rate':>8}")
    for name, pairs in (("issuing_body", court_pairs), ("document_kind", kind_pairs)):
        ok, total, bad = agreement(pairs)
        rate = f"{ok / total:.1%}" if total else "-"
        print(f"{name:<18}{total:>9}{ok:>7}{rate:>8}")
        for file_name, got, want in bad[:show]:
            print(
                f"    differs: {file_name.split('__')[-1][:34]:<36} extracted={got!r}  meta.csv={want!r}"
            )
    if kind_distribution:
        print(
            f"\ndocument_kind distribution (extracted): {dict(sorted(kind_distribution.items()))}"
        )

    print(
        """
How to read this:
  compared   documents that have a stored value AND a row in meta.csv.
  agree      the extracted value equals meta.csv's, after normalisation:
             issuing_body: accents/case/punctuation ignored;
             document_kind: our tokens coarsened (judgment, partial_judgment,
             interim_judgment -> Ítélet; order -> Végzés; other -> Egyéb).
  Documents with no stored value (absent / unverified / not yet extracted) are not
  in 'compared': see `make meta-coverage` for how many those are.
  meta.csv is measurement truth only; the generic metadata core never reads it."""
    )
