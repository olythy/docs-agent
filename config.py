import os
from dataclasses import dataclass

from dotenv import load_dotenv

# AGENT_ENV must be read from the *real* process environment, before any
# .env file is loaded — never from a file itself. If it lived inside .env,
# deciding which file to load would require having already loaded a file,
# a circular bootstrapping problem. Set it as a real shell/CI env var (e.g.
# `AGENT_ENV=test uv run ...` — `make test` does this automatically),
# never inside .env/.env.test.
AGENT_ENV = os.getenv("AGENT_ENV", "local")

# Captured before any file is loaded, so CI (which sets DATABASE_URL as a
# real workflow env var and never has a .env.test file — it's gitignored)
# can be told apart from a local checkout that's simply missing .env.test.
_database_url_preset = "DATABASE_URL" in os.environ

# Always load the shared base file first (common defaults, credentials).
load_dotenv(".env")

# Layers test-specific overrides on top of .env — .env.test only needs to
# contain the keys that actually differ (mainly DATABASE_URL); everything
# else is inherited unchanged. override=True is required here since
# python-dotenv otherwise never replaces a value .env (or the real shell
# environment) already set.
if AGENT_ENV == "test":
    loaded_test_file = load_dotenv(".env.test", override=True)
    if not loaded_test_file and not _database_url_preset:
        raise RuntimeError(
            "AGENT_ENV=test but neither .env.test nor a real DATABASE_URL "
            "env var was found. Locally: create .env.test (see "
            ".env.test.example) pointing at a disposable test database — "
            "never reuse your real DATABASE_URL, tests/db/ runs real "
            "INSERT/DELETE/TRUNCATE against it. In CI: set DATABASE_URL "
            "directly in the workflow env."
        )


@dataclass(frozen=True)
class Settings:
    """Centralized, immutable application configuration.

    Values are resolved at import time from environment variables,
    with sensible defaults that allow the application to run without
    any .env file (using the local, free embedding driver).

    One canonical name per credential — no vendor-specific aliases.
    Mapping to old names is only kept temporarily inside this class
    and must be removed once the old names are purged from .env files.

    Environment variables (all optional unless marked REQUIRED):

    Embedding:
        EMBEDDING_DRIVER      Driver to use: ``local`` (default), ``openai``,
                              ``openrouter`` (routes to any OpenRouter-hosted
                              embedding model, e.g. ``google/gemini-embedding-001``,
                              using the same account/key shape as
                              ``LLM_DRIVER=openrouter``), ``gemini`` (Google's
                              native AI Studio API directly, not via OpenRouter --
                              for a free-tier API key with its own rate limit,
                              see EMBEDDING_REQUEST_DELAY_SECONDS below), or
                              ``jina`` (Jina AI's hosted Embeddings API --
                              chosen as a no-infra way to offload the CPU-bound
                              cost of the local model during bulk ingestion,
                              confirmed live to dominate ~90%+ of per-document
                              ingest time; see docs/decisions.md), or
                              ``vertex`` (Google Cloud's Vertex AI text
                              embedding API -- same no-infra motivation as
                              ``jina``, added after confirming live that
                              Jina's free-tier tokens-per-minute cap kept
                              interrupting a bulk ingest while Vertex AI's
                              default quota handled the same load without
                              throttling; billed against GCP credit instead
                              of a separate Jina API key).
        EMBEDDING_MODEL       Model name/id for the active driver (default:
                              ``intfloat/multilingual-e5-small`` for local).
                              Set to an OpenRouter embedding model id when
                              ``EMBEDDING_DRIVER=openrouter``, a Gemini model
                              id (e.g. ``gemini-embedding-001``) when
                              ``EMBEDDING_DRIVER=gemini``, or a Jina model id
                              (e.g. ``jina-embeddings-v3``) when
                              ``EMBEDDING_DRIVER=jina``.
        EMBEDDING_DIMENSION   Output vector dimension (must match the model; 384 for
                              e5-small; OpenRouter's ``dimensions`` request
                              parameter, the native Gemini API's
                              ``output_dimensionality`` config, and Jina's own
                              ``dimensions`` parameter all truncate a larger
                              native model output to this value, e.g.
                              gemini-embedding-001's native 3072 -> 384).
        EMBEDDING_API_KEY     API key for the embedding driver (only when
                              ``EMBEDDING_DRIVER`` is ``openai``, ``openrouter``,
                              ``gemini``, or ``jina`` — for ``openrouter``, the
                              same OpenRouter key as ``LLM_API_KEY``; for
                              ``gemini``, a native Google AI Studio key
                              (aistudio.google.com/apikey); for ``jina``, a
                              Jina AI API key (jina.ai/?sui=apikey) -- each a
                              different kind of key, none interchangeable.
                              Configured independently of LLM_API_KEY since
                              embedding and answer generation are separate concerns).
        EMBEDDING_REQUEST_DELAY_SECONDS
                              Seconds to sleep before each embedding request
                              (default: 0.0, i.e. no throttling). Meaningful for
                              ``EMBEDDING_DRIVER=gemini`` on a free-tier API key, or
                              ``EMBEDDING_DRIVER=jina`` (confirmed live: a bulk
                              ingest can hit Jina's tokens-per-minute limit without
                              this) -- check the actual current limit on your
                              provider's own quota page rather than assuming a
                              number here, since free-tier limits change over time.
        VERTEX_PROJECT_ID     GCP project id (only when EMBEDDING_DRIVER=vertex).
        VERTEX_LOCATION       GCP region for the Vertex AI endpoint (default:
                              ``us-central1``; only when EMBEDDING_DRIVER=vertex).

    Chunking:
        CHUNK_SIZE            Target word count per chunk (default: 250).
        CHUNK_OVERLAP         Word overlap between adjacent chunks (default: 30).
        WORDS_PER_TOKEN       Approximate words-per-token ratio used to estimate
                              token count from word count when checking chunk
                              size against an embedding model's token limit
                              (default: 0.75, an English-average heuristic).
                              Morphologically rich languages (e.g. Hungarian)
                              often tokenize worse than this — lower it if the
                              truncation warning under-fires for your content.
                              Only used by the ``warn`` overflow strategy.
        CHUNK_OVERFLOW_STRATEGY
                              What to do when a chunk likely/actually exceeds
                              the embedding model's token limit: ``split``
                              (default) measures each chunk's real token count
                              with the driver's own tokenizer and re-splits any
                              chunk that overflows so nothing is ever silently
                              truncated. Requires the active driver to support
                              real token counting (currently only the ``local``
                              driver) — falls back to ``warn`` otherwise.
                              ``warn`` estimates via WORDS_PER_TOKEN and logs
                              a warning, but still truncates silently at embed
                              time.
        CHUNK_SPLIT_OVERLAP_RATIO
                              Fraction of words or tokens (0.0 to 0.5) to overlap
                              between pieces when CHUNK_OVERFLOW_STRATEGY=split divides
                              an oversized chunk (default: 0.15, i.e. 15%). Preserves
                              sentence and conceptual context across boundaries.
        PDF_EXTRACTION_MODE   ``flat`` (default) joins each page's text with a
                              single space — fast, but paragraph structure is
                              lost. ``blocks`` additionally detects paragraph
                              breaks from word coordinates and preserves them,
                              so ``CHUNKING_STRATEGY=langchain`` can split on
                              them. See ``ingestion/pdf_loader.extract_document_text``.
        CHUNKING_STRATEGY     ``word`` (default) is the sliding word-count
                              window (unchanged behavior, applied document-wide
                              instead of per-page — this alone fixes chunks
                              being truncated at page boundaries).
                              ``langchain`` uses
                              ``langchain_text_splitters.RecursiveCharacterTextSplitter``
                              to split on paragraph/sentence boundaries first,
                              falling back to smaller units only if a piece
                              still doesn't fit.

    Ingestion:
        INGEST_EXTENSIONS     Comma-separated list of file extensions permitted
                              during batch directory ingestion (default:
                              ``.pdf,.md,.markdown,.docx,.rtf``). Can be overridden at runtime
                              via ``allowed_extensions`` argument.

    LLM (answer generation):
        LLM_DRIVER            Driver to use: ``openrouter`` (default), ``openai``,
                              or ``gemini`` (Google's native AI Studio API
                              directly, not via OpenRouter -- for a free-tier
                              API key with its own rate limit, see
                              LLM_REQUEST_DELAY_SECONDS below).
        LLM_API_KEY           REQUIRED for answer generation. API key for the
                              active LLM driver -- for ``gemini``, a native
                              Google AI Studio key (aistudio.google.com/apikey),
                              a different kind of key than the OpenRouter one
                              (configured independently of EMBEDDING_API_KEY
                              since answer generation and embedding are
                              separate concerns, even when both happen to use
                              the same Gemini account).
        LLM_MODEL             Model identifier. Defaults to
                              ``google/gemini-3.1-flash-lite`` on OpenRouter.
                              Set to a native Gemini model id (e.g.
                              ``gemini-2.5-flash``) when ``LLM_DRIVER=gemini``.
        LLM_REQUEST_DELAY_SECONDS
                              Seconds to sleep before each chat-completion
                              request (default: 0.0, i.e. no throttling). Only
                              meaningful for ``LLM_DRIVER=gemini`` on a
                              free-tier API key, which has its own
                              requests-per-minute limit -- check the actual
                              current limit on your own AI Studio quota page
                              rather than assuming a number here, since
                              free-tier limits change over time.

    Retrieval:
        RETRIEVAL_TOP_K       Number of chunks to retrieve per query (default: 4).
        RETRIEVAL_MIN_SCORE   Cosine similarity threshold (0–1). Chunks below this
                              score are considered too distant and ignored
                              (default: 0.25).
        RETRIEVAL_STRATEGY    ``hybrid`` (default) fuses vector + keyword
                              search via Reciprocal Rank Fusion, then
                              optionally reranks (see RERANKER_DRIVER
                              below). ``vector`` skips keyword search and
                              fusion, returning pure cosine-similarity
                              results only — the pre-hybrid-search
                              behavior, kept as a selectable strategy
                              mainly for comparison (see
                              ``scripts/evaluate_retrieval.py``); there's
                              little reason to prefer it in production,
                              since hybrid search only adds recall over
                              vector-only at negligible extra cost. See
                              ``query/retrieval.py``.
        RERANKER_DRIVER       ``cross_encoder`` (default) reorders the hybrid-search
                              candidate list with a local cross-encoder model
                              before truncating to RETRIEVAL_TOP_K, and is the
                              second, embedding-model-independent relevance
                              gate (see RERANKER_MIN_SCORE) -- necessary in
                              practice, not just a quality lever: confirmed
                              empirically that switching EMBEDDING_DRIVER can
                              shift the raw vector-similarity scale enough
                              that RETRIEVAL_MIN_SCORE alone no longer
                              separates relevant from irrelevant queries,
                              while the reranker (scoring question+chunk
                              jointly, not via embedding distance) still did.
                              ``none`` skips reranking entirely -- kept for
                              comparison/eval only (see ``make eval`` vs.
                              ``make eval-rerank``), not recommended for
                              production. ``jina`` offloads reranking to
                              Jina AI's hosted Reranker API instead of the
                              local cross-encoder. ``vertex`` offloads it
                              to Google Cloud's standalone Discovery Engine
                              Ranking API instead -- added after Jina's
                              free-tier tokens-per-minute cap interrupted a
                              real eval run; billed against GCP credit
                              instead, same VERTEX_PROJECT_ID as
                              EMBEDDING_DRIVER=vertex, no separate API key.
                              Only applies when RETRIEVAL_STRATEGY=hybrid.
                              See ``drivers/reranker.py``.
        RERANKER_MODEL        Model name/id for the active reranker driver
                              (the local cross-encoder's HuggingFace id, a
                              Jina reranker model id when
                              RERANKER_DRIVER=jina, or a Vertex ranking
                              model id, e.g. ``semantic-ranker-default@latest``,
                              when RERANKER_DRIVER=vertex).
        RERANKER_API_KEY      API key for the reranker driver (only when
                              RERANKER_DRIVER=jina -- vertex uses the same
                              gcloud-based auth as EMBEDDING_DRIVER=vertex,
                              no separate key).
        RERANKER_MIN_SCORE    Minimum logit relevance score required from the
                              cross-encoder reranker (default: -2.0). When
                              RERANKER_DRIVER=cross_encoder, candidates scoring
                              below this threshold are dropped, and if no candidate
                              clears it, the query is rejected early as having
                              no relevant information. Calibrated to allow
                              longer/diluted passages (> -2.0) while rejecting
                              out-of-domain or unanswerable queries (< -3.5).
        RETRIEVAL_CANDIDATE_POOL_SIZE
                              Candidate pool size for the vector + full-text
                              searches feeding RRF fusion (and, if enabled,
                              reranking) — always at least RETRIEVAL_TOP_K,
                              but wider by default so fusion/reranking has
                              something to actually reorder (default: 20).
                              See ``query/retrieval.py``.

    Database (REQUIRED):
        DATABASE_URL          PostgreSQL connection URL with pgvector enabled.

    Development / Testing / Observability:
        LOG_FILE              Path to the structured JSONL audit/events log file
                              (default: ``logs/log.jsonl`` in local, ``logs/log-test.jsonl``
                              when AGENT_ENV=test).
        TEST_DOC_PATH         Path to a local document (PDF, Markdown, DOCX, or RTF) used
                              by ``scripts/extract_text.py``,
                              ``scripts/inspect_chunks.py``, and the tests/db/
                              add_document() end-to-end test. The format is
                              detected from the extension, same as
                              add_document() itself.
        AGENT_ENV             ``local`` (default) or ``test``. Read from the
                              real process environment only — never from
                              .env/.env.test themselves (see module
                              docstring above ``load_dotenv``). Controls
                              whether .env.test is also loaded, layered on
                              top of .env. ``make test`` sets this
                              automatically; you shouldn't need to set it
                              by hand.
    """

    # --- Meta ---
    #: Mirrors the module-level AGENT_ENV — already resolved before this
    #: class body runs, so this is just exposing it as settings.AGENT_ENV
    #: for consistency with every other value here.
    AGENT_ENV: str = AGENT_ENV

    # --- Observability ---
    LOG_FILE: str = os.getenv(
        "LOG_FILE",
        "logs/log-test.jsonl" if AGENT_ENV == "test" else "logs/log.jsonl",
    )

    # --- Embedding ---
    EMBEDDING_DRIVER: str = os.getenv("EMBEDDING_DRIVER", "local")
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "intfloat/multilingual-e5-small"
    )
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))
    EMBEDDING_API_KEY: str = os.getenv("EMBEDDING_API_KEY", "")
    EMBEDDING_REQUEST_DELAY_SECONDS: float = float(
        os.getenv("EMBEDDING_REQUEST_DELAY_SECONDS", "0.0")
    )
    VERTEX_PROJECT_ID: str = os.getenv("VERTEX_PROJECT_ID", "")
    VERTEX_LOCATION: str = os.getenv("VERTEX_LOCATION", "us-central1")

    # --- Chunking ---
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "250"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "30"))
    WORDS_PER_TOKEN: float = float(os.getenv("WORDS_PER_TOKEN", "0.75"))
    CHUNK_OVERFLOW_STRATEGY: str = os.getenv("CHUNK_OVERFLOW_STRATEGY", "split")
    CHUNK_SPLIT_OVERLAP_RATIO: float = float(
        os.getenv("CHUNK_SPLIT_OVERLAP_RATIO", "0.15")
    )
    PDF_EXTRACTION_MODE: str = os.getenv("PDF_EXTRACTION_MODE", "flat")
    CHUNKING_STRATEGY: str = os.getenv("CHUNKING_STRATEGY", "word")

    # --- Ingestion ---
    INGEST_EXTENSIONS: str = os.getenv(
        "INGEST_EXTENSIONS", ".pdf,.md,.markdown,.docx,.rtf"
    )

    @property
    def parsed_ingest_extensions(self) -> frozenset[str]:
        """Return the normalized, lowercased set of extensions configured for ingestion."""
        return frozenset(
            ext.strip().lower()
            if ext.strip().startswith(".")
            else f".{ext.strip().lower()}"
            for ext in self.INGEST_EXTENSIONS.split(",")
            if ext.strip()
        )

    # --- LLM (answer generation) ---
    LLM_DRIVER: str = os.getenv("LLM_DRIVER", "openrouter")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "google/gemini-3.1-flash-lite")
    LLM_REQUEST_DELAY_SECONDS: float = float(
        os.getenv("LLM_REQUEST_DELAY_SECONDS", "0.0")
    )

    # --- Retrieval ---
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "4"))
    RETRIEVAL_MIN_SCORE: float = float(os.getenv("RETRIEVAL_MIN_SCORE", "0.25"))
    RETRIEVAL_STRATEGY: str = os.getenv("RETRIEVAL_STRATEGY", "hybrid")
    RERANKER_DRIVER: str = os.getenv("RERANKER_DRIVER", "cross_encoder")
    RERANKER_MODEL: str = os.getenv(
        "RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    )
    RERANKER_MIN_SCORE: float = float(os.getenv("RERANKER_MIN_SCORE", "-2.0"))
    RERANKER_API_KEY: str = os.getenv("RERANKER_API_KEY", "")
    RETRIEVAL_CANDIDATE_POOL_SIZE: int = int(
        os.getenv("RETRIEVAL_CANDIDATE_POOL_SIZE", "20")
    )

    # --- Database ---
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")

    # --- Development / Testing ---
    TEST_DOC_PATH: str = os.getenv("TEST_DOC_PATH", "")


#: Singleton settings instance — import this everywhere in the application.
settings = Settings()
