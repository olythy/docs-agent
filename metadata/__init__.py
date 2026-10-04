"""Structured-metadata package.

Everything that fills and reads the typed per-document metadata layer (see
docs/structured-metadata-design.md). The core here knows nothing about any one
corpus; corpus-specific knowledge arrives as catalog data and optional adapters.

    catalog      -- Validating, loading and versioning the key catalog.
    verification -- Checking that an extracted value really exists in the text.
    date_parsers -- Language-specific date readers the verifier can be given.
"""
