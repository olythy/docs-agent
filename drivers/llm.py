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

from config import settings


class AnswerDriver(ABC):
    """Abstract base class for all LLM answer-generation backends.

    Subclasses must implement :meth:`answer`.
    The driver receives the user's question and a list of relevant text chunks
    retrieved from the vector store, and returns a grounded answer string.
    """

    @abstractmethod
    def answer(self, question: str, context_chunks: list[dict]) -> str:
        """Generate a grounded answer from retrieved context chunks.

        The implementation must instruct the model to base its answer only on
        the provided context, and to state clearly when the answer cannot be
        found — never fabricate information.

        Args:
            question: The user's natural-language question.
            context_chunks: A list of chunk dicts as returned by the retrieval
                layer. Each dict contains at minimum:
                    - ``content`` (str): The raw chunk text.
                    - ``metadata`` (dict): At least ``source_file`` and
                      ``page_number`` for source citation.

        Returns:
            A string containing the answer, ideally citing the source document
            and page number for each piece of information used.
        """


def _build_prompt(question: str, context_chunks: list[dict]) -> tuple[str, str]:
    """Assemble the system prompt and user message for a RAG query.

    Keeping prompt construction in one place makes it easy to iterate on
    wording without touching driver code.

    Args:
        question: The user's question.
        context_chunks: Retrieved chunks with ``content`` and ``metadata``.

    Returns:
        A tuple of ``(system_prompt, user_message)``.
    """
    # Build a numbered context block so the model can cite sources
    context_parts = []
    for i, chunk in enumerate(context_chunks, start=1):
        meta = chunk.get("metadata", {})
        source = meta.get("source_file", "unknown")
        page = meta.get("page_number", "?")
        context_parts.append(
            f"[{i}] Source: {source}, page {page}\n{chunk['content']}"
        )
    context_text = "\n\n".join(context_parts)

    system_prompt = (
        "You are a helpful assistant that answers questions based strictly on "
        "the provided document excerpts. "
        "Always cite the source file and page number when referencing information. "
        "If the answer cannot be found in the excerpts, respond with exactly: "
        "'I could not find this information in the provided documents.'"
    )

    user_message = (
        f"Document excerpts:\n\n{context_text}\n\n"
        f"Question: {question}"
    )

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
        self._model = model or settings.LLM_MODEL
        self._client = None  # Lazy initialisation — avoids cost at import time

    def _get_client(self):
        """Lazily create and cache the OpenAI client.

        Returns:
            An ``openai.OpenAI`` client instance.
        """
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=settings.LLM_API_KEY)
        return self._client

    def answer(self, question: str, context_chunks: list[dict]) -> str:
        """Generate an answer via the OpenAI Chat Completions API.

        Args:
            question: The user's question.
            context_chunks: Retrieved chunks from the vector store.

        Returns:
            The model's answer string.
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
        )
        return response.choices[0].message.content or ""


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
        self._model = model or settings.LLM_MODEL
        self._client = None  # Lazy initialisation

    def _get_client(self):
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

    def answer(self, question: str, context_chunks: list[dict]) -> str:
        """Generate an answer via the OpenRouter Chat Completions API.

        Args:
            question: The user's question.
            context_chunks: Retrieved chunks from the vector store.

        Returns:
            The model's answer string.
        """
        system_prompt, user_message = _build_prompt(question, context_chunks)
        client = self._get_client()

        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""


def get_answer_driver() -> AnswerDriver:
    """Factory function: return the active LLM driver from settings.

    Reads ``settings.LLM_DRIVER`` and instantiates the matching driver.

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
