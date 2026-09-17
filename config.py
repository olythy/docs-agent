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
        EMBEDDING_DRIVER      Driver to use: ``local`` (default) or ``openai``.
        EMBEDDING_MODEL       Model name/id for the active driver.
        EMBEDDING_DIMENSION   Output vector dimension (must match the model).
        EMBEDDING_API_KEY     API key for the embedding driver (only when
                              ``EMBEDDING_DRIVER=openai``).

    Chunking:
        CHUNK_SIZE            Target word count per chunk (default: 500).
        CHUNK_OVERLAP         Word overlap between adjacent chunks (default: 50).
        WORDS_PER_TOKEN       Approximate words-per-token ratio used to estimate
                              token count from word count when checking chunk
                              size against an embedding model's token limit
                              (default: 0.75, an English-average heuristic).
                              Morphologically rich languages (e.g. Hungarian)
                              often tokenize worse than this — lower it if the
                              truncation warning under-fires for your content.

    LLM (answer generation):
        LLM_DRIVER            Driver to use: ``openrouter`` (default) or ``openai``.
        LLM_API_KEY           REQUIRED for answer generation. API key for the
                              active LLM driver.
        LLM_MODEL             Model identifier. Defaults to ``openrouter/free``
                              which auto-routes to an available free model.

    Retrieval:
        RETRIEVAL_TOP_K       Number of chunks to retrieve per query (default: 4).
        RETRIEVAL_MIN_SCORE   Cosine similarity threshold (0–1). Chunks below this
                              score are considered too distant and ignored
                              (default: 0.25).

    Database (REQUIRED):
        DATABASE_URL          PostgreSQL connection URL with pgvector enabled.

    Development / Testing:
        TEST_PDF_PATH         Path to a local PDF used by
                              ``scripts/extract_text.py`` and the tests/db/
                              add_document() end-to-end test.
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

    # --- Embedding ---
    EMBEDDING_DRIVER: str = os.getenv("EMBEDDING_DRIVER", "local")
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
    )
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))
    EMBEDDING_API_KEY: str = os.getenv("EMBEDDING_API_KEY", "")

    # --- Chunking ---
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    WORDS_PER_TOKEN: float = float(os.getenv("WORDS_PER_TOKEN", "0.75"))

    # --- LLM (answer generation) ---
    LLM_DRIVER: str = os.getenv("LLM_DRIVER", "openrouter")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "openrouter/free")

    # --- Retrieval ---
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "4"))
    RETRIEVAL_MIN_SCORE: float = float(os.getenv("RETRIEVAL_MIN_SCORE", "0.25"))

    # --- Database ---
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")

    # --- Development / Testing ---
    TEST_PDF_PATH: str = os.getenv("TEST_PDF_PATH", "")


#: Singleton settings instance — import this everywhere in the application.
settings = Settings()
