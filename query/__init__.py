"""Query package.

Contains the retrieval and answer-generation pipeline:

    retrieval  -- Embeds a question, fetches top-k similar chunks from the
                  vector store, and generates a grounded answer via the LLM driver.
"""
