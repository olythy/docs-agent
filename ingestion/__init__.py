"""Ingestion package.

This package contains the pipeline components for ingesting documents
into the RAG knowledge base:

    pdf_loader  -- PDF text extraction (pdfplumber).
    chunker     -- Text splitting into overlapping chunks.
    ingest      -- Orchestrates loader → chunker → embedder → storage.
"""
