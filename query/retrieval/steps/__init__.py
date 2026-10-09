"""The retrieval steps, grouped by what they do.

candidates -- embed, dense search, year widening, CSLS reorder, keyword search
ranking    -- RRF fusion, rerank, listwise rerank
gates      -- the relevance gate (cosine) and the reranker's score gate
selection  -- the final cut with the year quota and the spread over named documents
"""
