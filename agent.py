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
via its new public ``get_client()``/``model`` — but tool-calling support is
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

from drivers.llm import get_answer_driver
from ingestion.ingest import add_document
from query.retrieval import query_knowledge_base

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add_document",
            "description": (
                "Ingest a new document (PDF or Markdown) into the knowledge "
                "base, so its content becomes searchable by "
                "query_knowledge_base. Use this when the user asks to add, "
                "upload, or ingest a file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": (
                            "Path to the document file to ingest "
                            "(.pdf, .md, or .markdown)."
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
                    }
                },
                "required": ["question"],
            },
        },
    },
]

_SYSTEM_PROMPT = (
    "You are an assistant for a personal document knowledge base. You have "
    "two tools available: add_document (ingest a new file into the "
    "knowledge base) and query_knowledge_base (answer a question using "
    "already-ingested documents). Decide which tool, if any, the user's "
    "message calls for, and call it. If the message needs neither, reply "
    "directly."
)


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
    if name == "query_knowledge_base":
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
    client = driver.get_client()

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    response = client.chat.completions.create(
        model=driver.model,
        messages=messages,
        tools=TOOLS,
    )
    message = response.choices[0].message

    if not message.tool_calls:
        return message.content or ""

    print(f"[agent] Model requested {len(message.tool_calls)} tool call(s).")
    messages.append(
        {
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in message.tool_calls
            ],
        }
    )

    for tool_call in message.tool_calls:
        arguments = json.loads(tool_call.function.arguments)
        print(f"[agent] Calling {tool_call.function.name}({arguments}) ...")
        try:
            result = _call_tool(tool_call.function.name, arguments)
        except Exception as e:  # noqa: BLE001 — deliberately broad: a failing
            # tool call (bad path, DB error, API error, ...) must be reported
            # back to the model as a tool result, not crash the whole loop.
            result = f"Error: {e}"
        messages.append(
            {"role": "tool", "tool_call_id": tool_call.id, "content": result}
        )

    final_response = client.chat.completions.create(model=driver.model, messages=messages)
    return final_response.choices[0].message.content or ""


if __name__ == "__main__":
    print("docs-agent — interactive agent CLI. Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue
        print(f"\nAgent: {run_agent(user_input)}")
