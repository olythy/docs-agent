# Structured metadata layer: design

Status: **agreed design; milestone 1 is built** (the tables, `DocumentStore`, the `sync-documents` command, tests) and applied to the test database; milestones 2-5 are not. The evidence behind it is in `docs/decisions.md` (the 2026-10-04 entries on counting/listing, the `document_date` accuracy measurement, and the key-extraction prototype).

## Why

Top-k chunk retrieval is good at *lookup* ("who represents the second defendant in case X") and structurally unable to *count* or *list* ("how many judgments were issued last October", "which invoices are due next week"). Measured on this corpus: a plain SQL filter over a date and a couple of keys gave 94-100% precision and 98-100% recall on 30 count questions; the current pipeline gave the exact count on 0 of 8. The fix is not a better retriever but typed, queryable per-document facts, plus a planner that turns a question into a query over them.

## Principles

1. **The core knows nothing about any corpus.** It knows keys, value types and filters. Corpus-specific knowledge lives only in (a) catalog *data* (key descriptions, allowed values), (b) optional *adapters* (a regex extractor, a sidecar-file importer), and (c) measurement tools under `corpus/`. A court decision, an invoice and a contract go through the same code.
2. **All identifiers are English.** Keys, enum tokens, tables and columns are English snake_case. Only *values* keep the document's language; categorical keys use canonical English tokens from an allowed list, with the original wording kept as evidence.
3. **Existence is guaranteed, selection is not.** Every extracted value carries a verbatim quote that the code verifies against the chunk text. That proves the value is in the document, not that the right one was picked, so ambiguous keys store *all* candidates with a role and let the query choose the aggregation.
4. **Unknown is a first-class answer.** A count says "N, plus K unknown", never a bare N.
5. **The LLM never writes SQL.** It emits a strict JSON plan; the code validates it against the catalog and compiles parameterised SQL.
6. **Additive.** Existing `document_chunks` metadata and the lookup pipeline keep working unchanged.

## Data model

`documents` -- identity and the root of every cascade.

| column | meaning |
|---|---|
| `content_hash` (PK) | SHA-256 of the whole document; already on every chunk's metadata |
| `source_file` | basename |
| `summary` | the per-document summary (today copied onto every chunk) |
| `ingested_at` | when it was (re)ingested |

`meta_keys` -- the key catalog.

| column | meaning |
|---|---|
| `doc_type`, `key` | scope and English snake_case name |
| `value_type` | `text`, `number`, `date`, `bool` |
| `description`, `example` | what the extractor reads to decide whether a key fits |
| `allowed_values` | for categorical keys: canonical English tokens |
| `multi_valued` | whether several rows per document are expected |
| `status` | `proposed` (not usable in queries), `approved`, `retired` |
| `version` | bumped when the description changes |

`document_meta` -- the values.

| column | meaning |
|---|---|
| `content_hash` (FK, `ON DELETE CASCADE`) | the document |
| `key`, `key_version` | which key, and which description version produced it |
| `value_text`, `value_number`, `value_date`, `value_bool`, `unit` | typed value; the database enforces that exactly one of the four `value_*` columns is set |
| `ordinal` | position among a multi-valued key's rows |
| `qualifiers` (JSONB) | role / party / instance etc.; the catalog declares which qualifiers a key allows |
| `evidence`, `evidence_chunk_index`, `page` | the verbatim quote and where it is (a chunk *index*, not a chunk id, which goes stale on re-ingest) |
| `source` | `deterministic`, `llm` or `sidecar` |
| `extracted_at` | |

`document_meta_status` -- what we know about a (document, key) pair, including the absence of a value.

| `state` | meaning |
|---|---|
| `present` | extracted and verified |
| `confirmed_absent` | looked at, the document does not state it |
| `unverified` | extracted, but the quote did not check out; never used by queries |
| `not_attempted` | no attempt yet (also the implicit state of a missing row) |

`document_chunks` gets a **generated `content_hash` column with an index**, so "search only inside this set of documents" is an indexed join instead of a JSONB scan. No re-embedding.

## Components

Each has one reason to change. SQL lives only in the store (as `VectorStore` does for chunks); orchestrators stay thin.

| component | responsibility | abstraction |
|---|---|---|
| `DocumentStore` | all SQL for the four tables above; executes compiled plans | -- |
| `MetaSource` (ABC) | produce candidate values for one document | `LLMMetaSource` (default), `RegexMetaSource` (the existing date/identifier extractors, optional), `SidecarMetaSource` (later) |
| `EvidenceSelector` | for each catalog key, rerank the document's chunks using the key *description* as the query and keep the best 1-2 | `RerankerDriver` |
| `EvidenceVerifier` | quote present in the chunk text; number/date derivable from the quote | pluggable, language-specific date and number parsers |
| `MetaExtractionRunner` | the batch command's orchestrator; idempotent, resumable, prints a coverage report (red when a key is missing on many documents) | -- |
| `QueryPlan` + `PlanCompiler` | validate a plan against the catalog, compile to `(sql, params)`; a pure function | -- |
| `DateRangeResolver` + `Clock` | turn a symbolic interval ("last year, October", "next week") into a concrete range | `SystemClock`, `FixedClock` for tests |
| `QueryPlanner` (ABC) | question + catalog + today's date -> plan | `LLMQueryPlanner` |

## Query plan

```json
{
  "operation": "count",
  "filters": [
    {"key": "issuing_body", "op": "eq", "value": "Debreceni Ítélőtábla"},
    {"key": "decision_date", "op": "between", "value": {"relative": {"year": -1, "month": 10}}}
  ],
  "group_by": null,
  "limit": 50,
  "residual": null
}
```

- `operation`: `lookup`, `list`, `count`, `sum`, `overview`. The planner is the router: `lookup` runs the existing retrieval restricted to the document set the filters return; a case number in the question always forces `lookup`.
- Dates are resolved by `DateRangeResolver`, never by LLM arithmetic. A range the symbolic forms cannot express may be given as ISO and is validated (start <= end, sensible length).
- `residual` is the part of the question no key covers. It is evaluated by map-reduce over the filtered candidates under a cost cap, and the answer then says "N of M candidates examined"; a count over a residual is not exact. A residual that recurs is promoted to a backfilled key; keys are **not** created lazily as a side effect of a query (that would make coverage depend on query history).
- The answer always states the executed filter ("counted: issuing_body = ..., decision_date 2025-10-01..2025-10-31") and the unknowns.

## Extraction

A separate batch command (modelled on `compute-hub-scores`), run after ingest, idempotent and resumable: it only processes (document, key) pairs that are missing or whose `key_version` is stale. One LLM call per document with ~3k tokens of selected evidence. The extractor must reuse an existing key whose description fits and may only *propose* a new one (`proposed`, unusable in queries until approved or merged). Categorical values are chosen from `allowed_values`. Where a deterministic extractor already exists and is configured (e.g. the court corpus's date and case-number regexes), it is a `MetaSource` and the LLM is not asked for the same fact.

## Evaluation

- Count/list: questions generated from known fields with exactly computable answers, scored with set precision/recall (no LLM grader).
- Extraction accuracy: ground truth for this corpus is `corpus/meta.csv` and the signature-line date, used **only under `corpus/`** (it is a by-product of this corpus's downloader and must never reach the core). For a corpus without such truth, hand-label only the disagreements between two runs or two models.
- Role-tagged amounts: measure stability on the *role*, not on the value.

## Milestones

1. Migration, `DocumentStore`, generated `content_hash` column; backfill `documents` from the existing chunks. **Built.** Keeping `documents` in step with ingestion (register on ingest, remove on replace) is not wired yet: `sync-documents` does it on demand and is idempotent.
2. `extract-meta` for objective keys (`issuing_body`, `decision_date`, `document_kind` as a categorical key) with the LLM source and the existing regex date as an adapter; coverage report.
3. Plan DSL, `PlanCompiler`, `DateRangeResolver`, `Clock`: testable without any LLM.
4. `LLMQueryPlanner` and a `corpus/` eval command that scores count/list questions exactly.
5. Wire the planner into `query_knowledge_base` as the router.

Out of scope for now: role-tagged amounts, `case_category`, postal codes and settlements (anonymisation made them unreliable here), and a sidecar importer.
