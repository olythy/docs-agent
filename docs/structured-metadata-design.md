# Structured metadata layer: design

Status: **agreed design; milestones 1 and 2 are built** (tables, `DocumentStore`, catalog loader, evidence verifier, `EvidenceSelector`, the LLM and adapter sources, `MetaExtractionRunner`, `extract-meta` / `coverage`, and the `meta-accuracy` measurement); a first 50-document live run is recorded in `docs/decisions.md`. **Milestone 3 is built** (see below). **Milestone 4 is coded and unit-tested but not yet measured** (the planner has not been run against a model on the full extraction). **Milestone 5 is built behind a switch** (`QUERY_ROUTER`, off by default). Measured on the golden set: single-document questions unchanged; content questions improved reproducibly on a biased 8-question sample (3/24 -> 13/24); the full repeated run is still to do (see `docs/decisions.md`, 2026-10-05).

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
| `value_type` | `text`, `number`, `date`, `bool`, `identifier` (stored as text, compared in a normalised form: see `metadata/identifiers.py`) |
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

`document_chunks` first got a generated `content_hash` column with an index (migration 0004) so "search only inside this set of documents" was an indexed join instead of a JSONB scan. That link was replaced by the numeric `document_id` (migrations 0005 and 0006, see "Document types and numeric document ids" below): the generated column no longer exists.

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

1. Migration, `DocumentStore`, generated `content_hash` column; backfill `documents` from the existing chunks. **Built.** The ingest now registers the document before it saves its chunks (see below); `sync-documents` only removes documents that were left without chunks.
2. `extract-meta` for objective keys (`issuing_body`, `decision_date`, `document_kind` as a categorical key) with the LLM source and the existing regex date as an adapter; coverage report. **Built**: the catalog loader/versioning (`metadata/catalog.py`, `meta_cli.py load-catalog`, the court catalog as data in `corpus/data/meta_catalog.json`), the `EvidenceVerifier` with a pluggable `DateParser`, `EvidenceSelector`, `LLMMetaSource` and `ChunkMetadataSource`, `MetaExtractionRunner` and the `extract-meta` / `coverage` commands. The first live run (50 documents, not a random sample) verified every value; the one disagreement with ground truth was a redacted court name that the document itself does not contain.
3. Plan DSL, `PlanCompiler`, `DateRangeResolver`, `Clock`: testable without any LLM. **Built**: `metadata/plan.py` (strict `parse_plan`), `metadata/date_ranges.py` and `clock.py`, `metadata/compiler.py`, `metadata/executor.py`, and `DocumentStore.execute_query` (a read-only transaction with a statement timeout). Verified on a real Postgres, including the opening question (how many decisions did a court issue last October) and the "+K unknown" accounting.
4. `LLMQueryPlanner` and a `corpus/` eval command that scores count/list questions exactly. **Coded, unmeasured**: `metadata/planner.py` (English prompt with today's date, the approved keys and the date grammar; the plan is validated by the same compiler that runs it; one retry with the exact error, then `PlanningFailed`, no fallback plan) and `corpus/commands/meta_plan_eval.py` (`meta-plan-eval`: a fact (scope + period) is picked from the stored values and its exact answer computed in Python; an LLM only phrases it as a question in any language; relative periods run on a fixed clock inside the data's date range; exact-match and set precision/recall). Run it on full extraction coverage.
5. Wire the planner into `query_knowledge_base` as the router. **Built, off by default** (`QUERY_ROUTER`): `query/router.py`. The planner always decides: an identifier in the question is a parameter, not an intent (a lookup that names one is not restricted by the plan's filters; a request for documents *similar* to a named one is answered "not supported yet", `unsupported` operation); count/list/sum/overview are answered exactly **only when the plan has no residual** (a plan with a residual cannot be exact: it is read like a lookup, with the filters narrowing the documents), phrased by an LLM that must reproduce every figure (else the plain facts are returned, with a warning); a lookup with filters runs the normal retrieval through `VectorStore(selection=...)`, which restricts the vector, full-text and identifier searches to the selected documents with `document_id IN (<sub-select>)`: the planner's filter is evaluated by the database, so no list of documents travels and there is no cap on the match. Nothing is silent: the executed filter, the "+K unknown", a part of the question no key covers, and a plan that could not be made are all stated.

## Document types and numeric document ids (agreed 2026-10-05; the id and the restricted search are built, the types are not)

Two gaps remain after milestone 5: the system does not know *what kind* of document a document is (the catalog's `doc_type` is declared by the operator, and the router reads it from a setting), and chunks reach their document through a 64-character hash. Decisions, with the reasoning:

**Document types are data, in a table.** `document_types(type, name, description, status, created_at)`: `type` is the English snake_case identifier (text, not a number: it is readable in queries and logs, and `meta_keys.doc_type` already refers to it), `name` is for display, `description` is what the classifier and the planner read, `status` is `proposed`, `approved` or `retired` (a `StrEnum` in code and a `CHECK` in the database, like `meta_keys`). `documents.document_type` is a nullable reference to it (a plain column, not a metadata key: "how many invoices" is the most basic query and must not need an `EXISTS`; and the type decides *which* catalog applies, so it has to exist before any key). `meta_keys.doc_type` becomes a reference to the same table.

**No separate classification-state column.** The state follows from the type: `document_type` empty = not classified yet *or* the classifier failed (a failure is reported and retried, never stored, like a failed extraction); pointing at a `proposed` type = the classifier suggested a new type and a person has to decide; `approved` = a valid classification; `retired` = the type was withdrawn or merged. This replaces a vague "no match": the answer to "what did not fit" is always a named, described, proposed type. Rules that keep it honest: the classifier sees the *existing* proposed types too (so it does not mint fifty near-duplicates); a classification carries a quote (such as the heading line) that the code verifies, as for keys; the LLM never creates a usable type on its own (a proposed type is unusable in queries until approved with the CLI, or declared in the catalog file, which is the reviewed-in-git route); counts over a type say "+K unknown" for documents with an empty or `proposed` type.

**The catalog file holds many types:** `{"types": [{"type", "name", "description", "keys": [...]}]}`. A type declared in the file is loaded as `approved`.

**The planner chooses the type** from the descriptions and plans only over that type's keys; when the question fits none or several it says so. The compiler also applies `document_type = ...` as a filter (today it only checks the keys), and `QUERY_ROUTER_DOC_TYPE` goes away. Classification is a separate batch command run after `sync-documents`; the ingest is untouched.

**A numeric document id replaces the hash as the link** (a correction of an earlier "no surrogate id"): `documents.id BIGSERIAL` is the primary key, `content_hash` stays `UNIQUE` (it identifies duplicates); `document_chunks.document_id` (a real column, not generated) references it with `ON DELETE CASCADE`, and so do `document_meta` and `document_meta_status`. Why it matters: at 25 million chunks the hash costs about 1.6 GB plus its index, against about 200 MB for a number; and, more importantly, the restricted search must not ship a *list* of documents (the current `content_hashes=[...]` sends up to 5,000 hashes per query and cannot exceed that) but a *sub-select* (`document_id IN (SELECT id FROM documents WHERE document_type = ... AND ...)`), so the database does the filtering. The migration is gentler than 0004: adding a nullable column is instant (0004's generated column rewrote the table and its HNSW index), then it is back-filled from `documents`; the generated `content_hash` column on the chunks is dropped in a *later* migration once the code uses `document_id` (dropping a column is also instant). The `content_hash` inside the chunk metadata JSON stays, because the ingest writes it and the ingest is not touched yet.

**Status:** steps (1) to (6) are built: the planner chooses the document type from the types' descriptions and the plan names it (`"document_type"`, required, null only for a `lookup`); the compiler restricts a plan to documents of that type and counts documents with no type as *unknown*; `QUERY_ROUTER_DOC_TYPE` is gone; extraction, coverage and the stored-value lists are per type, and the CLI's `--doc-type` no longer defaults to a corpus-specific name. Earlier: steps (1) to (4) and the contract migration (6) (migration 0006: `document_id` is `NOT NULL` and cascades everywhere, `documents` has `id` as its primary key and `content_hash` as a unique natural key, the old `content_hash` columns of the chunks, values and statuses are gone, and a schema test pins the end state) (the catalog file declares many types, each with a name, a description and its keys; the old single-type format is refused with the way to convert it), and the restricted search uses the sub-select (measured on the real data: identical or better recall than the hash list, 4-85 ms; see `docs/decisions.md`, 2026-10-05). The classifier (`metadata/classifier.py`, `metadata/classification_runner.py`, `meta_cli.py classify-documents`) was tried on 10 random documents (10 of 10 `court_decision`, quote verified, none failed) and on three texts that are not in the database, where it proposed `company_policy` and `invoice` (quotes verified); the existing corpus was then typed with the manual `assign-type` shortcut because it has one kind of document. (Done since: (5).)  Still to do, in the larger plan: (5) the planner's type choice, (6) the contract migration.

**Order:** (1) migration 0005 (types table, `documents.id` / `document_type`, the new reference columns, back-filled); (2) `DocumentStore` / `VectorStore` on the new columns with the sub-select, with tests; (3) the multi-type catalog format and loader; (4) the classifier. The dev database is backed up first (`make db-dump`, restore verified), and a migration runs there only with approval.

Out of scope for now: role-tagged amounts, `case_category`, postal codes and settlements (anonymisation made them unreliable here), and a sidecar importer.
