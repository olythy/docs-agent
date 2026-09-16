import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


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
        TEST_PDF_PATH         Absolute path to a local PDF used by
                              ``scripts/extract_text.py``. Never commit real
                              file paths containing personal identifiers here.
    """

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
