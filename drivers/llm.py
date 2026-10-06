"""LLM answer-generation driver abstractions.

Defines the common interface (``AnswerDriver``) for all chat-completion backends,
following the same Strategy / Driver pattern as ``drivers/embedding.py``.

The active driver is selected via ``settings.LLM_DRIVER``:
    - ``"openrouter"`` → :class:`OpenRouterAnswerDriver` (default; supports free models)
    - ``"openai"``     → :class:`OpenAIAnswerDriver` (requires paid subscription)
    - ``"gemini"``     → :class:`GeminiAnswerDriver` (Google's native AI Studio
                         API directly, not via OpenRouter — free-tier friendly,
                         rate-limited via ``LLM_REQUEST_DELAY_SECONDS``)
    - ``"vertex"``     → :class:`VertexAnswerDriver` (the same Gemini models,
                         served via Vertex AI instead of AI Studio — added
                         after AI Studio's free-tier quota kept being the
                         throughput ceiling; billed against GCP credit,
                         same VERTEX_PROJECT_ID as the other Vertex drivers)

``OpenAIAnswerDriver``/``OpenRouterAnswerDriver`` share an OpenAI-SDK-specific
base class (``_OpenAICompatibleAnswerDriver``) — OpenRouter exposes an
OpenAI-compatible API, so the only meaningful difference between them is the
``base_url`` and a required ``HTTP-Referer`` header OpenRouter uses for
rate-limiting. ``GeminiAnswerDriver`` talks to a structurally different SDK
(``google-genai``, not OpenAI-compatible) and implements the ``AnswerDriver``
contract independently, translating this project's own OpenAI-shaped
``messages``/``tools`` dicts to and from Gemini's ``Content``/``Part``/``Tool``
shapes internally.

Usage::

    from drivers.llm import get_answer_driver
    driver = get_answer_driver()
    answer = driver.answer(question="Mi az SZJA tartozásom?", context_chunks=[...])
"""

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, cast

from config import settings
from models import RetrievedChunk
from retry_policy import TransientAPIError, retry_on_transient_error

if TYPE_CHECKING:
    # Only for type annotations — the real import is deferred to inside each
    # driver's _get_client() to avoid each SDK's import cost when this
    # module is merely imported, not actually used.
    from google import genai
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
        provider_data: Opaque, driver-specific data a caller must echo back
            unchanged on the follow-up turn (e.g. ``agent.py`` copies it into
            the assistant message's tool-call entry) without needing to
            understand its contents. Used by :class:`GeminiAnswerDriver` to
            carry a function call's ``thought_signature`` — confirmed live
            that Gemini's newer models reject a reconstructed function-call
            turn missing it ("Function call is missing a thought_signature
            ... required for tools to work correctly"). ``None`` for every
            other driver.
    """

    id: str
    name: str
    arguments: str
    provider_data: dict | None = None


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


def _http_options():
    """The ``google-genai`` HTTP options: a request is abandoned after the configured time.

    The SDK takes the limit in *milliseconds*. Without it a request that never gets an
    answer blocks forever; with it the SDK raises ``httpx.TimeoutException`` (a
    ``httpx.TransportError``), which :meth:`GeminiAnswerDriver._generate` already turns
    into a retryable ``TransientAPIError``. See ``settings.API_REQUEST_TIMEOUT_SECONDS``.
    """
    from google.genai import types

    return types.HttpOptions(timeout=int(settings.API_REQUEST_TIMEOUT_SECONDS * 1000))


class AnswerDriver(ABC):
    """Abstract base class for all LLM answer-generation backends.

    Two methods define the contract every driver must implement:
    :meth:`answer` (RAG-specific: fixed system prompt, no tools) and
    :meth:`run_tool_calling_turn` (``agent.py``'s function-calling loop:
    caller-supplied ``messages`` history and ``tools=[...]`` schemas).
    Both take/return this project's own plain-dict/:class:`AgentTurnResult`
    shapes — never a provider SDK's types — so ``agent.py`` and
    ``query/retrieval.py`` stay provider-agnostic.

    ``OpenAIAnswerDriver``/``OpenRouterAnswerDriver`` share a concrete
    OpenAI-SDK-specific implementation via ``_OpenAICompatibleAnswerDriver``
    (both really are OpenAI-compatible endpoints, so sharing a template
    method there is legitimate, not premature abstraction). ``GeminiAnswerDriver``
    talks to a structurally different SDK and implements both methods directly
    — trying to force it through the same OpenAI-shaped template method would
    misrepresent it as OpenAI-compatible, which it isn't.
    """

    def __init__(self, model: str) -> None:
        """Store the model identifier.

        Args:
            model: The chat-completion model identifier to use.
        """
        self._model = model

    @property
    def model(self) -> str:
        """The chat-completion model identifier this driver is configured for."""
        return self._model

    @abstractmethod
    def run_tool_calling_turn(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> AgentTurnResult:
        """Send one chat-completion request, optionally offering tool schemas.

        Args:
            messages: Chat history so far, OpenAI ``role``/``content`` dict shape.
            tools: OpenAI-style ``tools=[...]`` function schemas, or
                ``None``/empty to omit tool-calling entirely for this call.

        Returns:
            An :class:`AgentTurnResult` with the reply and/or requested tool calls.
        """

    @abstractmethod
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
            max_tokens: Maximum tokens to generate (default: 1024).

        Returns:
            A string containing the answer, ideally citing the source document
            and page number for each piece of information used.
        """


class _OpenAICompatibleAnswerDriver(AnswerDriver):
    """Shared template-method implementation for OpenAI-compatible backends.

    Both concrete drivers below (:class:`OpenAIAnswerDriver`,
    :class:`OpenRouterAnswerDriver`) use the ``openai`` Python SDK — OpenRouter
    exposes an OpenAI-compatible API — so the only real difference between
    them is how the client is constructed. Subclasses only implement
    :meth:`_get_client`.
    """

    def __init__(self, model: str) -> None:
        """Store the model identifier; the client is created lazily.

        Args:
            model: The chat-completion model identifier to use.
        """
        super().__init__(model)
        self._client: OpenAI | None = None  # Lazy init — avoids import cost

    @abstractmethod
    def _get_client(self) -> "OpenAI":
        """Lazily create and cache the provider-specific OpenAI-compatible client."""

    def run_tool_calling_turn(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> AgentTurnResult:
        """Send one chat-completion request, optionally offering tool schemas.

        This method (not a raw client getter) is the type boundary: the
        OpenAI SDK's exact typed ``ChatCompletionMessageParam``/
        ``ChatCompletionToolParam`` shapes are only enforced *here*, via
        the ``cast`` calls below — callers work with plain dicts and this
        method's own :class:`AgentTurnResult`/:class:`ToolCallRequest`,
        never the SDK's types directly. The ``cast`` is a deliberate,
        narrow assertion (this driver is responsible for building
        correctly-shaped dicts), not a blanket type-check suppression.
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

        Args:
            question: The user's natural-language question.
            context_chunks: Chunks as returned by the retrieval layer.
            max_tokens: Maximum tokens to generate (default: 1024). Prevents
                upstream aggregators (e.g. OpenRouter) from pre-authorizing
                the model's entire theoretical context limit against account credits.

        Returns:
            A string containing the answer.
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


_SYSTEM_PROMPT_HEAD = (
    "You are a helpful assistant that answers questions based strictly on "
    "the provided document excerpts. Do not use your own general or "
    "training knowledge, and do not fill in gaps with what you believe "
    "is probably true, even if you feel confident about it — every "
    "claim in your answer must be traceable to a specific excerpt "
    "below. Always cite the source file and page number when "
    "referencing information. If the excerpts only partially answer "
    "the question, say exactly what they do and don't cover, rather "
    "than completing the picture from outside knowledge. "
)

#: The original refusal rule. Confirmed live that with a broad "how did the
#: practice develop ..." question and two relevant (but partial) excerpts, the
#: model read "cannot be found at all" as "the excerpts do not contain the
#: whole answer" and gave this exact sentence every time (3/3), even when a
#: reminder was appended after the question.
_REFUSAL_CLAUSE_STRICT = (
    "If the answer cannot be found in the excerpts at all, respond with "
    "exactly: 'I could not find this information in the provided "
    "documents.'"
)

#: Refuse only when NO excerpt is relevant; otherwise answer with what the
#: excerpts show. The exact refusal sentence is kept (the golden-set's decline
#: detection and the adversarial questions depend on it).
_REFUSAL_CLAUSE_PARTIAL = (
    "Refuse only when NONE of the excerpts is relevant to the question: then "
    "respond with exactly: 'I could not find this information in the provided "
    "documents.' Whenever at least one excerpt is relevant, you must answer "
    "with what it shows, even if that is only part of what was asked: list the "
    "relevant cases or facts with their file names and page numbers, and then "
    "say in one sentence which part of the question the excerpts do not cover. "
    "A partial answer is always better than a refusal."
)

_SAMPLE_NOTE = (
    " The excerpts are a small sample selected from a much larger collection "
    "of decisions, so they will rarely cover everything a broad question asks: "
    "describe what this sample shows and say what it does not cover, instead "
    "of refusing."
)


def _build_prompt(
    question: str, context_chunks: list[RetrievedChunk]
) -> tuple[str, str]:
    """Assemble the system prompt and user message for a RAG query.

    Keeping prompt construction in one place makes it easy to iterate on
    wording without touching driver code.

    Args:
        question: The user's question.
        context_chunks: Retrieved chunks, already ranked best-first (see
            ``query.retrieval.retrieve_chunks``).

    Returns:
        A tuple of ``(system_prompt, user_message)``.
    """
    # Build a numbered context block so the model can cite sources.
    #
    # Placed worst-to-best, not in ``context_chunks``' own best-first
    # order -- LLMs attend most to the very start and very end of a long
    # context and lose track of the middle ("lost in the middle";
    # confirmed as a real pattern worth applying here, see
    # docs/decisions.md). The single highest-ranked chunk is reversed into
    # the *last* position, immediately before "Question:", so the model
    # reads it right as it's about to answer, instead of it being the
    # first thing read and then potentially drowned out by everything
    # that follows. Citation correctness doesn't depend on the numbering
    # matching retrieval rank -- see corpus/commands/eval.py's
    # _resolve_cited_source_files, which verifies citations by extracting
    # identifiers from the answer text itself, not by these [i] labels.
    context_parts = []
    for i, chunk in enumerate(reversed(context_chunks), start=1):
        source = chunk.metadata.source_file or "unknown"
        page = (
            chunk.metadata.page_number
            if chunk.metadata.page_number is not None
            else "?"
        )
        date = (
            f", date: {chunk.metadata.document_date}"
            if settings.EXPOSE_DOCUMENT_DATE and chunk.metadata.document_date
            else ""
        )
        context_parts.append(
            f"[{i}] Source: {source}{date}, page {page}\n{chunk.content}"
        )
    context_text = "\n\n".join(context_parts)

    refusal_clause = (
        _REFUSAL_CLAUSE_PARTIAL
        if settings.ANSWER_PARTIAL_COVERAGE
        else _REFUSAL_CLAUSE_STRICT
    )
    system_prompt = _SYSTEM_PROMPT_HEAD + refusal_clause
    if settings.ANSWER_PARTIAL_COVERAGE:
        system_prompt += _SAMPLE_NOTE

    user_message = f"Document excerpts:\n\n{context_text}\n\nQuestion: {question}"

    return system_prompt, user_message


class OpenAIAnswerDriver(_OpenAICompatibleAnswerDriver):
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

            self._client = OpenAI(
                api_key=settings.LLM_API_KEY,
                timeout=settings.API_REQUEST_TIMEOUT_SECONDS,
            )
        return self._client


class OpenRouterAnswerDriver(_OpenAICompatibleAnswerDriver):
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
                timeout=settings.API_REQUEST_TIMEOUT_SECONDS,
            )
        return self._client


def _messages_to_gemini_contents(messages: list[dict]):
    """Translate this project's OpenAI-shaped ``messages`` into Gemini's own shape.

    Gemini has no ``"system"`` message role (``system_instruction`` is a
    separate ``GenerateContentConfig`` field, not part of ``contents``), and
    uses ``"model"`` rather than ``"assistant"`` for the model's own turns.
    Two things confirmed only by hitting the real API, not from the SDK's
    own docs (which show a different, no-longer-accepted shape):

    - A function response's ``Content.role`` must be ``"user"``, not
      ``"tool"`` — the live API rejects ``"tool"`` outright ("Role 'tool'
      is not supported... valid role: SYSTEM, ... USER, ASSISTANT, ...
      MODEL, USER").
    - Newer ("thinking") models reject a reconstructed function-call Part
      that's missing the ``thought_signature`` the model originally
      attached to it ("Function call is missing a thought_signature ...
      required for tools to work correctly") — so ``ToolCallRequest.id``
      is Gemini's own real per-call id (confirmed live it exists, e.g.
      ``"call_461665"``, contrary to the SDK docs' single-call examples
      never showing one), and ``provider_data["thought_signature"]``
      (set by :meth:`GeminiAnswerDriver.run_tool_calling_turn`) must be
      re-attached to the function-call Part when rebuilding this turn's
      history for the next request.

    A call id this project made up (``provider_data["synthesized_id"]``, see
    :meth:`GeminiAnswerDriver.run_tool_calling_turn`) is used only to pair the
    tool result with its call; it is not sent to the API.

    A tool-role message only carries ``tool_call_id`` (OpenAI's shape has
    no field for the function name there) — the id-to-name mapping built
    while walking the assistant message that made the call is needed to
    give the function-response Part its required ``name``.

    Args:
        messages: OpenAI-shaped chat history (this project's own dict convention).

    Returns:
        A ``(system_instruction, contents)`` tuple — ``system_instruction``
        is ``None`` if no ``"system"`` message was present.
    """
    from google.genai import types

    system_instruction = None
    contents = []
    call_id_to_name: dict[str, str] = {}
    synthesized_ids: set[str] = set()
    for message in messages:
        role = message["role"]
        if role == "system":
            system_instruction = message["content"]
        elif role == "user":
            contents.append(
                types.Content(
                    role="user", parts=[types.Part.from_text(text=message["content"])]
                )
            )
        elif role == "assistant":
            parts = []
            if message.get("content"):
                parts.append(types.Part.from_text(text=message["content"]))
            for tool_call in message.get("tool_calls", []):
                call_id = tool_call["id"]
                name = tool_call["function"]["name"]
                call_id_to_name[call_id] = name
                part = types.Part.from_function_call(
                    name=name, args=json.loads(tool_call["function"]["arguments"])
                )
                assert part.function_call is not None  # from_function_call() sets it
                provider_data = tool_call.get("provider_data") or {}
                if provider_data.get("synthesized_id"):
                    synthesized_ids.add(call_id)
                else:
                    part.function_call.id = call_id
                if "thought_signature" in provider_data:
                    part.thought_signature = provider_data["thought_signature"]
                parts.append(part)
            contents.append(types.Content(role="model", parts=parts))
        elif role == "tool":
            call_id = message["tool_call_id"]
            response_part = types.Part.from_function_response(
                name=call_id_to_name[call_id],
                response={"result": message["content"]},
            )
            assert (
                response_part.function_response is not None
            )  # from_function_response() sets it
            if call_id not in synthesized_ids:
                response_part.function_response.id = call_id
            # role="user", not "tool": see this function's docstring.
            contents.append(types.Content(role="user", parts=[response_part]))
        else:
            raise ValueError(f"Unknown message role for Gemini translation: {role!r}")
    return system_instruction, contents


def _tools_to_gemini(tools: list[dict]):
    """Translate this project's OpenAI-style ``tools=[...]`` schemas into Gemini's shape.

    Both shapes describe function parameters as plain JSON Schema, so
    ``parameters`` maps directly onto Gemini's ``parameters_json_schema`` —
    no schema translation needed, just a different wrapper structure.

    Args:
        tools: OpenAI-style ``tools=[...]`` function schemas.

    Returns:
        A single-element list of ``types.Tool``, Gemini's expected shape.
    """
    from google.genai import types

    function_declarations = [
        types.FunctionDeclaration(
            name=tool["function"]["name"],
            description=tool["function"].get("description"),
            parameters_json_schema=tool["function"].get("parameters"),
        )
        for tool in tools
    ]
    return [types.Tool(function_declarations=function_declarations)]


class GeminiAnswerDriver(AnswerDriver):
    """Answer driver using Google's native Gemini API (AI Studio) directly, not via OpenRouter.

    Talks to a structurally different SDK than the OpenAI-compatible drivers
    above (``google-genai``, not ``openai``) — implements :meth:`answer` and
    :meth:`run_tool_calling_turn` directly rather than sharing
    ``_OpenAICompatibleAnswerDriver``'s template methods, translating this
    project's own OpenAI-shaped ``messages``/``tools`` dicts to and from
    Gemini's ``Content``/``Part``/``Tool`` shapes via
    :func:`_messages_to_gemini_contents`/:func:`_tools_to_gemini`.

    For a free-tier AI Studio API key, which carries its own
    requests-per-minute limit distinct from any paid quota — throttled via
    a real sleep (``LLM_REQUEST_DELAY_SECONDS``) before each request, same
    pattern as :class:`drivers.embedding.GeminiEmbeddingDriver`.
    """

    def __init__(self, model: str | None = None) -> None:
        """Initialise the driver without creating the client yet.

        Args:
            model: Gemini chat model id. Defaults to ``settings.LLM_MODEL``.
        """
        super().__init__(model or settings.LLM_MODEL)
        self._client: genai.Client | None = None  # Lazy init — avoids import cost

    def _get_client(self) -> "genai.Client":
        """Lazily create and cache the ``google-genai`` client.

        Returns:
            The ``genai.Client`` instance.
        """
        if self._client is None:
            from google import genai

            self._client = genai.Client(
                api_key=settings.LLM_API_KEY, http_options=_http_options()
            )
        return self._client

    @retry_on_transient_error(max_attempts=3)
    def _generate(self, contents, tools=None, system_instruction=None, **config_kwargs):
        """Call ``generate_content``, rate-limited and retried like the embedding driver.

        Sleeps ``LLM_REQUEST_DELAY_SECONDS`` before every attempt (including
        retries, since the whole function re-runs from the top on each
        retry). Raises :class:`retry_policy.TransientAPIError` on a 429/5xx
        response to trigger a retry with exponential backoff (up to 3
        attempts, via :func:`retry_policy.retry_on_transient_error`); any
        other error propagates immediately.

        Also retries ``httpx.TransportError`` (connection failures, DNS
        errors, timeouts) the same way — see
        :meth:`drivers.embedding.GeminiEmbeddingDriver._embed_one_batch`'s
        docstring for why this is a separate exception hierarchy from
        ``APIError`` that was previously not retried at all, confirmed live
        with a real "No route to host" mid-ingestion crashing an otherwise
        unrelated run.

        Raises:
            TransientAPIError: If every retry is exhausted.
            Exception: Whatever the ``google-genai`` client raises, for a
                non-retryable error.
        """
        import time

        import httpx
        from google.genai import types
        from google.genai.errors import APIError

        if settings.LLM_REQUEST_DELAY_SECONDS > 0:
            time.sleep(settings.LLM_REQUEST_DELAY_SECONDS)

        client = self._get_client()
        if tools:
            config_kwargs["tools"] = tools
        if system_instruction is not None:
            config_kwargs["system_instruction"] = system_instruction
        # Confirmed live, a real finish_reason=MAX_TOKENS on a "thinking"
        # model (gemini-2.5-flash): max_output_tokens is a *shared* budget
        # across hidden reasoning and the visible answer -- a real case saw
        # 981 thinking tokens leave only 39 for the actual answer, truncating
        # it mid-sentence with no citation at all. Defaults to 0 (thinking
        # disabled) since this grounded-answer task needs the full budget
        # for the visible answer, not deep reasoning over already-retrieved
        # context. See docs/decisions.md.
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_budget=settings.LLM_THINKING_BUDGET
        )

        try:
            return client.models.generate_content(
                model=self._model,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        except httpx.TransportError as exc:
            raise TransientAPIError(
                f"Gemini generate_content network error: {exc}"
            ) from exc
        except APIError as exc:
            status = getattr(exc, "code", None)
            if status == 401:
                # Only meaningful for VertexAnswerDriver (which inherits
                # this method unchanged) -- a no-op for GeminiAnswerDriver,
                # which doesn't use gcloud_auth at all. Confirmed live,
                # during a real multi-hour bulk ingest, that a Vertex AI
                # driver's cached gcloud token can stop working before our
                # ~1-hour assumption expects; invalidating it here means
                # the retry actually fetches a fresh one instead of
                # failing identically with the same stale cached token.
                from drivers.gcloud_auth import invalidate as invalidate_gcloud_token

                invalidate_gcloud_token()
                raise TransientAPIError("Gemini generate_content status 401") from exc
            if status == 429 or (status is not None and status >= 500):
                raise TransientAPIError(
                    f"Gemini generate_content status {status}"
                ) from exc
            raise

    def run_tool_calling_turn(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> AgentTurnResult:
        """Send one ``generate_content`` request, optionally offering tool schemas.

        See :func:`_messages_to_gemini_contents` for the message translation.
        Reads function calls from ``response.candidates[0].content.parts``
        rather than the SDK's ``response.function_calls`` convenience
        property, because the latter only exposes ``FunctionCall``
        (name/args/id) and drops each Part's ``thought_signature`` —
        confirmed live that a newer ("thinking") model needs that
        signature preserved on the next turn's reconstructed function-call
        Part, or it rejects the request outright.
        """
        system_instruction, contents = _messages_to_gemini_contents(messages)
        gemini_tools = _tools_to_gemini(tools) if tools else None

        response = self._generate(
            contents, tools=gemini_tools, system_instruction=system_instruction
        )

        tool_calls = []
        candidates = response.candidates or []
        parts = (
            candidates[0].content.parts if candidates and candidates[0].content else []
        )
        for index, part in enumerate(parts or []):
            function_call = part.function_call
            if function_call is None:
                continue
            assert function_call.name is not None, (
                "a function call from the model always has a name"
            )
            provider_data: dict = {}
            call_id = function_call.id
            if call_id is None:
                # ``FunctionCall.id`` is optional in the SDK ("If populated").
                # Confirmed live that the Gemini API (AI Studio) fills it, but
                # Vertex AI mode does not -- ``make chat`` crashed on the first
                # question that triggered a tool call. Make one up for this
                # project's own bookkeeping (pairing the tool result with the
                # call), and remember it is ours so it is never sent back to
                # the API as if the model had issued it.
                call_id = f"call_{index}_{uuid.uuid4().hex[:8]}"
                provider_data["synthesized_id"] = True
            if part.thought_signature:
                provider_data["thought_signature"] = part.thought_signature
            tool_calls.append(
                ToolCallRequest(
                    id=call_id,
                    name=function_call.name,
                    arguments=json.dumps(function_call.args or {}),
                    provider_data=provider_data or None,
                )
            )
        content = None if tool_calls else (response.text or "")
        return AgentTurnResult(content=content, tool_calls=tool_calls)

    def answer(
        self,
        question: str,
        context_chunks: list[RetrievedChunk],
        max_tokens: int = 1024,
    ) -> str:
        """Generate a grounded answer from retrieved context chunks.

        Args:
            question: The user's natural-language question.
            context_chunks: Chunks as returned by the retrieval layer.
            max_tokens: Maximum tokens to generate (default: 1024).

        Returns:
            A string containing the answer.
        """
        from google.genai import types

        system_prompt, user_message = _build_prompt(question, context_chunks)
        contents = [
            types.Content(role="user", parts=[types.Part.from_text(text=user_message)])
        ]
        response = self._generate(
            contents,
            system_instruction=system_prompt,
            max_output_tokens=max_tokens,
            temperature=0.2,
        )
        return response.text or ""


class VertexAnswerDriver(GeminiAnswerDriver):
    """Answer driver using Gemini via Vertex AI instead of AI Studio.

    Same no-infra, same-trigger motivation as
    :class:`drivers.embedding.VertexEmbeddingDriver`/
    :class:`drivers.reranker.VertexRankerDriver`: AI Studio's free-tier
    Gemini quota (``LLM_DRIVER=gemini``) kept being the throughput
    ceiling, while this GCP project's Vertex AI quota -- backed by GCP
    credit rather than a separate free-tier allowance -- has consistently
    tolerated rapid, undelayed requests for every other Vertex driver
    added today.

    Overrides only :meth:`_get_client` -- every other method
    (:meth:`answer`, :meth:`run_tool_calling_turn`, :meth:`_generate`'s
    retry/throttling/error handling) is identical between AI Studio and
    Vertex AI in the ``google-genai`` SDK; the backend is purely a client-
    construction detail. Authenticates via
    :func:`drivers.gcloud_auth.get_credentials` -- confirmed live that the
    SDK's ``vertexai=True`` mode accepts a credentials object built from
    the already-authenticated ``gcloud`` CLI session, without needing a
    separate ``gcloud auth application-default login``.
    """

    def _get_client(self) -> "genai.Client":
        """Lazily create and cache a Vertex-AI-mode ``google-genai`` client."""
        if self._client is None:
            from google import genai

            from drivers.gcloud_auth import get_credentials

            self._client = genai.Client(
                vertexai=True,
                project=settings.VERTEX_PROJECT_ID,
                location=settings.VERTEX_LOCATION,
                credentials=get_credentials(),
                http_options=_http_options(),
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
    if driver_name == "gemini":
        return GeminiAnswerDriver()
    if driver_name == "vertex":
        return VertexAnswerDriver()

    raise ValueError(
        f"Unknown LLM_DRIVER: '{driver_name}'. "
        "Valid options are: 'openrouter', 'openai', 'gemini', 'vertex'."
    )
