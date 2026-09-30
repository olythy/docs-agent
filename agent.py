"""Function-calling agent: lets an LLM choose which tool(s) to call.

The RAG pipeline exposes exactly two operations — ``add_document`` (ingest
a new file) and ``query_knowledge_base`` (answer a question from ingested
documents). Every earlier step in this project called them directly, from
Python. This module is what makes it an *agent* rather than "just a RAG
pipeline": the two operations are described to an LLM as OpenAI-style
``tools=[...]`` function schemas, and the model itself decides — from a
single, free-form user message — whether to ingest, to query, or neither.

Flow (the standard OpenAI tool-calling loop):
    1. Send the user's message + the two tool schemas to the LLM.
    2. If the model didn't request a tool call, return its reply directly.
    3. Otherwise, run the requested Python function(s) for real, feed each
       result back to the model as a ``role: "tool"`` message.
    4. Ask the model for a final, natural-language reply now that it has
       the tool result(s).

Caveat worth knowing: this uses the same ``AnswerDriver`` (and therefore
the same ``LLM_MODEL``) as ``query_knowledge_base``'s answer generation —
via its ``run_tool_calling_turn()``/``model`` — but tool-calling support is
model-dependent, and ``settings.LLM_MODEL``'s default (``openrouter/free``,
which auto-routes to *some* available free model) is not guaranteed to
support it. If the model doesn't support tools, expect either an API error
or a reply that ignores the tools entirely, depending on the provider. A
model explicitly known to support tool-calling is recommended for this
module specifically.

Usage::

    from agent import run_agent
    reply = run_agent("Mi van a dokumentumban a Player Centralról?")
    print(reply)

Or interactively:

    uv run python agent.py
"""

import json
import logging

from drivers.llm import ToolCallRequest, get_answer_driver
from ingestion.ingest import add_directory, add_document
from query.retrieval import query_knowledge_base

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add_document",
            "description": (
                "Ingest a single new document (PDF, Markdown, DOCX, or RTF) into the knowledge "
                "base, so its content becomes searchable by "
                "query_knowledge_base. Use this when the user asks to add, "
                "upload, or ingest a specific file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": (
                            "Path to the document file to ingest "
                            "(.pdf, .md, .markdown, .docx, or .rtf)."
                        ),
                    }
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_directory",
            "description": (
                "Batch-ingest all supported documents (PDF, Markdown, DOCX, or RTF) from "
                "a directory into the knowledge base. Use this when the user "
                "asks to add or ingest an entire folder or directory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dir_path": {
                        "type": "string",
                        "description": "Path to the directory containing documents.",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to scan subdirectories recursively (default: true).",
                    },
                },
                "required": ["dir_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_knowledge_base",
            "description": (
                "Answer a question using previously ingested documents. Use "
                "this whenever the user asks something that could be "
                "answered from the knowledge base, rather than answering "
                "from general knowledge."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The user's natural-language question.",
                    },
                    "source_file": {
                        "type": "string",
                        "description": (
                            "Optional filename to restrict the search to (e.g. 'notes.md' "
                            "or 'sample.pdf'). Only provide if the user explicitly asks "
                            "about a specific document."
                        ),
                    },
                },
                "required": ["question"],
            },
        },
    },
]

_SYSTEM_PROMPT = (
    "You are an assistant for a personal document knowledge base. You have "
    "three tools available: add_document (ingest a single file), "
    "add_directory (batch-ingest a directory of documents), and "
    "query_knowledge_base (answer a question using already-ingested documents). "
    "Decide which tool, if any, the user's message calls for, and call it. "
    "If the message needs none, reply directly."
)


def _tool_call_to_dict(tool_call: ToolCallRequest) -> dict:
    """Convert a driver's ToolCallRequest into an OpenAI-shaped tool-call dict.

    Only includes ``provider_data`` when a driver actually set it (e.g.
    GeminiAnswerDriver's thought_signature) — OpenAI/OpenRouter never set
    it, so their payload stays exactly as before rather than gaining an
    extra key their real API might reject.

    Args:
        tool_call: The tool call to convert.

    Returns:
        A dict matching OpenAI's ``tool_calls[]`` entry shape.
    """
    entry = {
        "id": tool_call.id,
        "type": "function",
        "function": {"name": tool_call.name, "arguments": tool_call.arguments},
    }
    if tool_call.provider_data is not None:
        entry["provider_data"] = tool_call.provider_data
    return entry


def _call_tool(name: str, arguments: dict) -> str:
    """Execute one tool call for real and return a string result for the model.

    Args:
        name: The tool name the model requested (must be a key the model
            was offered in :data:`TOOLS`).
        arguments: The model's parsed (already-JSON-decoded) arguments.

    Returns:
        A string describing the result, suitable to feed back to the model
        as a ``role: "tool"`` message.

    Raises:
        ValueError: If ``name`` isn't one of the tools this agent offers.
    """
    if name == "add_document":
        add_document(arguments["file_path"])
        return f"Document '{arguments['file_path']}' was ingested successfully."
    if name == "add_directory":
        summary = add_directory(
            arguments["dir_path"],
            recursive=arguments.get("recursive", True),
            force=arguments.get("force", False),
        )
        return (
            f"Directory '{arguments['dir_path']}' processed: "
            f"{len(summary['ingested'])} file(s) ingested, "
            f"{len(summary['skipped'])} skipped, "
            f"{len(summary['failed'])} failed (out of {summary['total_found']} found)."
        )
    if name == "query_knowledge_base":
        source_file = arguments.get("source_file")
        if source_file:
            return query_knowledge_base(
                arguments["question"], metadata_filter={"source_file": source_file}
            )
        return query_knowledge_base(arguments["question"])
    raise ValueError(f"Unknown tool requested by the model: '{name}'")


def run_agent(user_message: str) -> str:
    """Run one full tool-calling turn for a single user message.

    Args:
        user_message: The user's free-form message.

    Returns:
        The model's final natural-language reply, after executing whichever
        tool(s) it chose to call (if any).
    """
    driver = get_answer_driver()

    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    turn = driver.run_tool_calling_turn(messages, tools=TOOLS)

    if not turn.tool_calls:
        return turn.content or ""

    print(f"[agent] Model requested {len(turn.tool_calls)} tool call(s).")
    messages.append(
        {
            "role": "assistant",
            "content": turn.content,
            "tool_calls": [
                _tool_call_to_dict(tool_call) for tool_call in turn.tool_calls
            ],
        }
    )

    for tool_call in turn.tool_calls:
        arguments = json.loads(tool_call.arguments)
        print(f"[agent] Calling {tool_call.name}({arguments}) ...")
        try:
            result = _call_tool(tool_call.name, arguments)
        except Exception as e:  # noqa: BLE001 — deliberately broad: a failing
            # tool call (bad path, DB error, API error, ...) must be reported
            # back to the model as a tool result, not crash the whole loop.
            result = f"Error: {e}"
        messages.append(
            {"role": "tool", "tool_call_id": tool_call.id, "content": result}
        )

    final_turn = driver.run_tool_calling_turn(messages)
    return final_turn.content or ""


def run_interactive() -> None:
    """Run an interactive console session: read a line, run one isolated turn, repeat.

    Each turn is independent — no conversation memory carries over between
    lines (see :func:`run_agent`'s docstring: one full tool-calling turn per
    call). Exits cleanly on 'exit'/'quit' or EOF (Ctrl+D); Ctrl+C is handled
    by the caller (``scripts/agent_cli.py``'s ``__main__`` block).
    """
    print("docs-agent — interactive agent CLI. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except EOFError:
            break
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue
        print(f"\nAgent: {run_agent(user_input)}")


if __name__ == "__main__":
    # ingestion.ingest/query.retrieval log their progress via `logging`, not
    # print() (mcp_server.py needs stdout clean for the MCP protocol) — this
    # CLI still wants to see those messages, so configure a bare, print()-like
    # handler here rather than leaving them silent (logging's default when
    # nothing calls basicConfig).
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_interactive()
