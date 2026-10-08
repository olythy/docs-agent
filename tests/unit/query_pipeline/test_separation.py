"""The new query pipeline must not depend on the original it is replacing.

The original implementation (``query/retrieval.py``'s ``retrieve_chunks`` and strategies, and
``query/router.py``) is the reference the new one is proven against, and is deleted when it
has been. That is only a deletion, not a rewrite, if nothing new reaches into it. The
dependency goes the other way: the original may import the new (the wording of a refusal, the
plan treatment), never the reverse.
"""

import ast
from pathlib import Path

import pytest

QUERY = Path(__file__).resolve().parents[3] / "query"

#: The new pipeline's modules.
NEW = [
    "facts",
    "outcome",
    "context",
    "step",
    "candidate_steps",
    "ranking_steps",
    "gate_steps",
    "selection_steps",
    "profiles",
    "runner",
    "observers",
    "legacy_trace",
    "service",
    "composition",
    "decision",
    "inflection",
]

#: What the new modules may not import.
ORIGINAL = {"query.retrieval", "query.router"}


def imports_of(path: Path) -> set[str]:
    """Every module a file imports, as a dotted name (``from a.b import c`` -> ``a.b``)."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            found |= {f"{node.module}.{alias.name}" for alias in node.names}
    return found


@pytest.mark.parametrize("module", NEW)
def test_a_new_module_does_not_import_the_original(module):
    reached = imports_of(QUERY / f"{module}.py") & ORIGINAL

    assert not reached, f"query/{module}.py imports the original: {sorted(reached)}"


def test_the_list_covers_every_module_of_the_new_pipeline():
    """A new module added to query/ has to be placed on one side of the line."""
    original = {"retrieval", "router"}
    reused = {
        "hybrid",
        "listwise_rerank",
        "time_filter",
        "decline_detection",
        "__init__",
    }
    on_disk = {p.stem for p in QUERY.glob("*.py")}

    assert on_disk - original - reused == set(NEW)
