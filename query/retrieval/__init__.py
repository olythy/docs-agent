"""Retrieval: which chunks of the documents in scope answer the question.

context, step  -- the shared frozen context and the contract of a step
steps/         -- the steps, by kind: candidates, ranking, gates, selection
profiles       -- the registered profiles (ordered lists of steps) and the factory
runner         -- the pipeline that validates and runs a chain, and the step observer
service        -- RetrievalService: one entry, owns the store session
hybrid, listwise_rerank -- the pure logic some steps use
"""
