"""MCP server exposing docs-agent's tools/prompt over stdio.

Meant to be added directly to an MCP host (Claude Desktop, Claude Code,
...) so documents can be ingested and the knowledge base searched from
inside a normal chat — see README.md's "MCP Server" section for how to
register it.

Deliberately exposes ``search_knowledge_base`` (wraps
``query.retrieval.retrieve_chunks()``), not ``query.retrieval.
query_knowledge_base()``: the latter makes its own ``LLM_DRIVER`` API call
to generate an answer, which defeats the point of using this from an MCP
host — the host's own model can write the grounded answer directly from
the returned excerpts, at no extra API cost to this project. ``add_document``
is the other tool, reusing ``ingestion.ingest.add_document()`` unchanged.

Also exposes a ``my-docs`` *prompt* (not a tool) — confirmed empirically
that a tool call alone isn't reliable: the host model sometimes answers
from its own memory of an unrelated, same-named real topic instead of
calling ``search_knowledge_base``, even with a strongly-worded tool
description. ``/my-docs <question>`` in Claude Desktop is a user-invoked
slash command, not the model's judgment call, so the tool call becomes
mandatory rather than a suggestion.

Usage::

    uv run mcp dev mcp_server.py       # MCP Inspector, for local testing
    uv run mcp install mcp_server.py --name "docs-agent" -f .env
                                        # registers with Claude Desktop
"""

import logging

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from ingestion.ingest import add_document as _add_document
from query.retrieval import retrieve_chunks

# ingestion.ingest/query.retrieval log their progress via `logging`, which
# defaults to stderr — safe on the MCP stdio transport, where stdout must
# stay reserved for the JSON-RPC protocol (a stray print() here corrupted a
# real client's message parsing mid-call, confirmed empirically). Configured
# at import time so it's in place before the first tool call, regardless of
# which transport mcp.run() ends up using.
logging.basicConfig(level=logging.INFO, format="%(message)s")

mcp = MCPServer("docs-agent")

_SEARCH_KNOWLEDGE_BASE_DESCRIPTION = (
    "ALWAYS call this tool for any question about the personal knowledge "
    "base — even if you believe you already know the answer, e.g. from "
    "memory of a past conversation or general knowledge. Do not rely on "
    "your own memory instead of calling this; confirmed empirically that "
    "an assistant will otherwise sometimes answer from memory about a "
    "same-named but entirely different, real project instead of what's "
    "actually in the knowledge base. Returns the raw retrieved excerpts "
    "(with their source file and page/section), not a generated answer. "
    "Answer the user's question yourself, using ONLY these excerpts — if "
    "they don't actually answer the question, say so explicitly rather "
    "than guessing or using outside knowledge."
)


def _to_search_result(chunk: dict) -> dict:
    """Strip a chunk dict down to what's useful to an MCP host model.

    Drops ``id``/``score`` — internal retrieval-plumbing fields (a
    primary key for RRF fusion, a similarity/rank score on a scale the
    host model has no calibration for) that would only add noise here.

    Args:
        chunk: A chunk dict as returned by
            :func:`query.retrieval.retrieve_chunks`.

    Returns:
        A dict with ``content``, ``source_file``, ``page_number``.
    """
    metadata = chunk.get("metadata", {})
    return {
        "content": chunk["content"],
        "source_file": metadata.get("source_file", "unknown"),
        "page_number": metadata.get("page_number", "?"),
    }


@mcp.tool(description=_SEARCH_KNOWLEDGE_BASE_DESCRIPTION)
def search_knowledge_base(question: str) -> list[dict]:
    """Retrieve knowledge-base excerpts relevant to ``question``.

    Args:
        question: The user's natural-language question.

    Returns:
        A list of excerpts (``content``/``source_file``/``page_number``),
        or an empty list if nothing cleared the relevance gate — an empty
        result means the knowledge base has nothing reliable on this, not
        that the search itself failed.
    """
    return [_to_search_result(chunk) for chunk in retrieve_chunks(question)]


@mcp.tool()
def add_document(file_path: str) -> str:
    """Ingest a document (PDF or Markdown) into the knowledge base.

    Args:
        file_path: Path to the document file to ingest.

    Returns:
        A short confirmation message.

    Raises:
        ToolError: For anything the model could plausibly react to and
            retry — a bad/missing path, an unsupported format, or the
            document already being present. Found the hard way: a plain
            ``ValueError``/``FileNotFoundError`` here is an *unexpected
            crash* as far as the MCP SDK is concerned, and it deliberately
            hides a crash's exception text from the client (only the
            server's own log gets it) — so the model (and the person
            reading its reply) saw a blank error with no way to tell what
            went wrong. ``ToolError``'s message reaches the model instead,
            which is what actually lets it read "already in the knowledge
            base" and decide what to do next.
    """
    try:
        _add_document(file_path)
    except (ValueError, FileNotFoundError) as e:
        raise ToolError(str(e)) from e
    return f"Document '{file_path}' was ingested successfully."


@mcp.prompt(name="my-docs")
def my_docs(question: str) -> str:
    """Answer a question using ONLY the docs-agent knowledge base.

    A ``prompts`` primitive, not a ``tools`` one, deliberately: a tool call
    is the *model's own judgment call* on whether to use it, and that
    judgment turned out unreliable in practice — confirmed empirically,
    the model chose to answer from its own memory instead of calling
    ``search_knowledge_base`` for a same-named but unrelated real topic.
    A prompt is *user*-invoked (a slash command in Claude Desktop's UI,
    e.g. ``/my-docs``), so calling the tool becomes mandatory, not a
    suggestion the model can talk itself out of. Worded even more strictly
    than the tool's own description ("no other tool", "every claim
    traceable to a specific excerpt") after a real test still showed
    Claude Desktop's own memory feature contributing to the answer
    alongside a correct, honest tool call — that's a client-side feature
    a prompt can't necessarily override, so this is a best-effort
    tightening, not a guarantee.
    """
    return (
        "For this question, you may use ONLY the docs-agent MCP server's "
        "search_knowledge_base tool — no other tool, no memory of past "
        "conversations, no general or outside knowledge, even if you "
        "believe you already know the answer. Call search_knowledge_base "
        "with the question below, then answer using ONLY the returned "
        "excerpts. Every claim in your answer must be traceable to a "
        "specific returned excerpt — cite its source file and page. If the "
        "excerpts don't answer the question, or only partially answer it, "
        "say exactly what you don't know rather than filling the gap from "
        "memory or general knowledge.\n\n"
        f"Question: {question}"
    )


if __name__ == "__main__":
    mcp.run()
