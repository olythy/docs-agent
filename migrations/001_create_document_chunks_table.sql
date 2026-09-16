-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Table to store document chunks and embeddings for RAG
CREATE TABLE IF NOT EXISTS document_chunks (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding vector(384),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index for fast vector cosine similarity search
CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
ON document_chunks
USING hnsw (embedding vector_cosine_ops);
