"""Query package: question -> decision -> (retrieval ->) answer.

    service, knowledge_base, composition, observers -- the entry layer
    facts, inflection, time_filter  -- reading the question
    outcome, decline_detection      -- refusals (the value, the wording, the eval's reading)
    answering                       -- the grounded and the exact answerer
    decision/                       -- which way the question goes, over which documents
    retrieval/                      -- which chunks answer it (context, steps, profiles)

Nothing is imported here on purpose: importing ``query.facts`` or ``query.outcome`` must not
load the model drivers. The design is in ``docs/query-pipeline-design.md``.
"""
