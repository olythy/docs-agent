"""LLM answer-generation driver abstractions.

Defines the common interface (``AnswerDriver``) for all chat-completion backends,
following the same Strategy / Driver pattern as ``drivers/embedding.py``.

The active driver is selected via ``settings.LLM_DRIVER``:
    - ``"openrouter"`` → :class:`OpenRouterAnswerDriver` (default; supports free models)
    - ``"openai"``     → :class:`OpenAIAnswerDriver` (requires paid subscription)

Both drivers use the ``openai`` Python SDK under the hood — OpenRouter exposes an
OpenAI-compatible API, so the only meaningful difference is the ``base_url`` and
a required ``HTTP-Referer`` header that OpenRouter uses for rate-limiting.

Usage::

    from drivers.llm import get_answer_driver
    driver = get_answer_driver()
    answer = driver.answer(question="Mi az SZJA tartozásom?", context_chunks=[...])
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, cast

from config import settings
from models import RetrievedChunk

if TYPE_CHECKING:
    # Only for type annotations — the real import is deferred to inside each
    # driver's _get_client() to avoid the openai SDK's import cost when this
    # module is merely imported, not actually used.
    from openai import OpenAI
    from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam


@dataclass(frozen=True)
class ToolCallRequest:
    """One tool call the model requested, in this project's own shape.

    Mirrors the OpenAI SDK's ``ChatCompletionMessageFunctionToolCall`` but
    is owned by this project — callers (e.g. ``agent.py``) depend on this
    type, never the SDK's own, so the SDK's exact message/tool-call
    TypedDict shapes stay confined to this driver module (see
    :meth:`AnswerDriver.run_tool_calling_turn`'s docstring for why).

    Attributes:
        id: The tool call's id, to echo back in the follow-up ``role: "tool"`` message.
        name: The requested tool's name (a key in the ``tools=[...]`` schema).
        arguments: The model's arguments, as a raw (not yet JSON-decoded) string.
    """

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class AgentTurnResult:
    """The result of one chat-completion call, in this project's own shape.

    Attributes:
        content: The model's natural-language reply, or ``None`` if it
            chose to call tool(s) instead of replying directly.
        tool_calls: Tool call requests, if any (empty when the model replied directly).
    """

    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)


class AnswerDriver(ABC):
    """Abstract base class for all LLM answer-generation backends.

    Both concrete drivers use the ``openai`` Python SDK (OpenRouter exposes
    an OpenAI-compatible API), so the only real difference between them is
    how the client is constructed — this base class owns everything else
    as a template method: build the prompt, get the client, call the chat
    completion, return the text. Subclasses only implement
    :meth:`_get_client`.
    """

    def __init__(self, model: str) -> None:
        """Store the model identifier; the client is created lazily.

        Args:
            model: The chat-completion model identifier to use.
        """
        self._model = model
        self._client: OpenAI | None = None  # Lazy init — avoids import cost

    @abstractmethod
    def _get_client(self) -> "OpenAI":
        """Lazily create and cache the provider-specific OpenAI-compatible client."""

    @property
    def model(self) -> str:
        """The chat-completion model identifier this driver is configured for."""
        return self._model

    def run_tool_calling_turn(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> AgentTurnResult:
        """Send one chat-completion request, optionally offering tool schemas.

        ``answer()`` (RAG-specific: fixed system prompt, no tools) is the
        only thing most callers need — but ``agent.py``'s function-calling
        loop needs to pass its own growing ``messages`` history and
        ``tools=[...]`` schemas, and read back tool-call requests, which
        ``answer()``'s fixed shape doesn't expose.

        This method (not a raw client getter) is the boundary instead: the
        OpenAI SDK's exact typed ``ChatCompletionMessageParam``/
        ``ChatCompletionToolParam`` shapes are only enforced *here*, via
        the ``cast`` calls below — callers work with plain dicts and this
        method's own :class:`AgentTurnResult`/:class:`ToolCallRequest`,
        never the SDK's types directly. The ``cast`` is a deliberate,
        narrow assertion (this driver is responsible for building
        correctly-shaped dicts), not a blanket type-check suppression.

        Args:
            messages: Chat history so far, OpenAI ``role``/``content`` dict shape.
            tools: OpenAI-style ``tools=[...]`` function schemas, or
                ``None``/empty to omit tool-calling entirely for this call.

        Returns:
            An :class:`AgentTurnResult` with the reply and/or requested tool calls.
        """
        client = self._get_client()
        create_kwargs: dict = {
            "model": self._model,
            "messages": cast("list[ChatCompletionMessageParam]", messages),
        }
        if tools:
            create_kwargs["tools"] = cast("list[ChatCompletionToolParam]", tools)

        response = client.chat.completions.create(**create_kwargs)
        message = response.choices[0].message

        tool_calls = []
        for tool_call in message.tool_calls or []:
            # Every schema this project offers is `"type": "function"` (see
            # agent.py's TOOLS) — a custom tool call is a different SDK
            # variant with no .function attribute, and shouldn't occur here.
            assert hasattr(tool_call, "function"), (
                f"Unexpected non-function tool call from the model: {tool_call!r}"
            )
            tool_calls.append(
                ToolCallRequest(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                )
            )

        return AgentTurnResult(content=message.content, tool_calls=tool_calls)

    def answer(
        self,
        question: str,
        context_chunks: list[RetrievedChunk],
        max_tokens: int = 1024,
    ) -> str:
        """Generate a grounded answer from retrieved context chunks.

        Instructs the model to base its answer only on the provided
        context, and to state clearly when the answer cannot be found —
        never fabricate information.

        Args:
            question: The user's natural-language question.
            context_chunks: Chunks as returned by the retrieval layer.
            max_tokens: Maximum tokens to generate (default: 1024). Prevents
                upstream aggregators (e.g. OpenRouter) from pre-authorizing
                the model's entire theoretical context limit against account credits.

        Returns:
            A string containing the answer, ideally citing the source document
            and page number for each piece of information used.
        """
        system_prompt, user_message = _build_prompt(question, context_chunks)
        client = self._get_client()

        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,  # Low temperature → more factual, less creative
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""


def _build_prompt(
    question: str, context_chunks: list[RetrievedChunk]
) -> tuple[str, str]:
    """Assemble the system prompt and user message for a RAG query.

    Keeping prompt construction in one place makes it easy to iterate on
    wording without touching driver code.

    Args:
        question: The user's question.
        context_chunks: Retrieved chunks.

    Returns:
        A tuple of ``(system_prompt, user_message)``.
    """
    # Build a numbered context block so the model can cite sources
    context_parts = []
    for i, chunk in enumerate(context_chunks, start=1):
        source = chunk.metadata.source_file or "unknown"
        page = (
            chunk.metadata.page_number
            if chunk.metadata.page_number is not None
            else "?"
        )
        context_parts.append(f"[{i}] Source: {source}, page {page}\n{chunk.content}")
    context_text = "\n\n".join(context_parts)

    system_prompt = (
        "You are a helpful assistant that answers questions based strictly on "
        "the provided document excerpts. Do not use your own general or "
        "training knowledge, and do not fill in gaps with what you believe "
        "is probably true, even if you feel confident about it — every "
        "claim in your answer must be traceable to a specific excerpt "
        "below. Always cite the source file and page number when "
        "referencing information. If the excerpts only partially answer "
        "the question, say exactly what they do and don't cover, rather "
        "than completing the picture from outside knowledge. If the "
        "answer cannot be found in the excerpts at all, respond with "
        "exactly: 'I could not find this information in the provided "
        "documents.'"
    )

    user_message = f"Document excerpts:\n\n{context_text}\n\nQuestion: {question}"

    return system_prompt, user_message


class OpenAIAnswerDriver(AnswerDriver):
    """Answer driver using the OpenAI Chat Completions API directly.

    Requires a paid OpenAI subscription and ``LLM_API_KEY`` in settings.
    """

    def __init__(self, model: str | None = None) -> None:
        """Initialise the OpenAI driver.

        Args:
            model: OpenAI model identifier. Defaults to ``settings.LLM_MODEL``.
        """
        super().__init__(model or settings.LLM_MODEL)

    def _get_client(self) -> "OpenAI":
        """Lazily create and cache the OpenAI client.

        Returns:
            An ``openai.OpenAI`` client instance.
        """
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=settings.LLM_API_KEY)
        return self._client


class OpenRouterAnswerDriver(AnswerDriver):
    """Answer driver using the OpenRouter API (OpenAI-compatible).

    OpenRouter aggregates many LLM providers and exposes them through an
    OpenAI-compatible endpoint. Free models (with ``:free`` suffix or via
    ``openrouter/free`` router) are available without a paid subscription.

    The ``base_url`` is a fixed constant of this driver class, not a config
    value — it is an implementation detail of OpenRouter, not something the
    user should configure.

    Requires ``LLM_API_KEY`` (your OpenRouter API key) in settings.
    """

    _BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, model: str | None = None) -> None:
        """Initialise the OpenRouter driver.

        Args:
            model: OpenRouter model identifier. Defaults to ``settings.LLM_MODEL``
                which is ``openrouter/free`` by default — this auto-routes to
                an available free model that supports chat completions.
        """
        super().__init__(model or settings.LLM_MODEL)

    def _get_client(self) -> "OpenAI":
        """Lazily create and cache the OpenRouter client.

        OpenRouter is accessed through the standard ``openai.OpenAI`` SDK
        by overriding the ``base_url``. This avoids adding any new dependency.

        Returns:
            An ``openai.OpenAI`` client pointed at OpenRouter.
        """
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=settings.LLM_API_KEY,
                base_url=self._BASE_URL,
                # OpenRouter requires HTTP-Referer for free-tier rate limiting
                default_headers={"HTTP-Referer": "https://github.com/docs-agent"},
            )
        return self._client


@lru_cache(maxsize=1)
def get_answer_driver() -> AnswerDriver:
    """Factory function: return the active LLM driver from settings.

    Reads ``settings.LLM_DRIVER`` and instantiates the matching driver.
    Cached with ``@lru_cache(maxsize=1)`` so repeated calls reuse the same
    driver instance instead of re-instantiating.

    Returns:
        An :class:`AnswerDriver` instance ready to call.

    Raises:
        ValueError: If ``LLM_DRIVER`` is set to an unknown value.
    """
    driver_name = settings.LLM_DRIVER.lower()

    if driver_name == "openrouter":
        return OpenRouterAnswerDriver()
    if driver_name == "openai":
        return OpenAIAnswerDriver()

    raise ValueError(
        f"Unknown LLM_DRIVER: '{driver_name}'. "
        "Valid options are: 'openrouter', 'openai'."
    )
