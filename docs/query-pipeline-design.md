# Query pipeline: design of the rewrite

Status: **slice 1 implemented and proven equal on a six-question sample under five settings** (see section 6 and `docs/decisions.md`, 2026-10-06); nothing in production uses the new pipeline yet, except behind the temporary `QUERY_ENGINE=v2` switch.
It builds on `docs/query-workflow.md` (the as-is map and its tangles) and on an
independent design review (Opus, read-only) of the first draft. Items marked
**[to confirm]** are proposals that still need an explicit decision.

## 1. Goal and scope

Rewrite from zero the path *question → decision → retrieval → answer*, as small classes
with one responsibility each, so that "what happens where, when and why" is visible and
every refusal is attributable to a stage.

**In scope** (today: `QueryRouter`, `as_routed`, the identifier check, `retrieve_chunks`,
`HybridRetrievalStrategy.select_chunks`, `_build_prompt`'s policy flags, the four places
that decide "no answer").

**Kept as black boxes, reused unchanged:** `VectorStore`, `DocumentStore`, the embedding /
LLM / reranker drivers, the metadata planner / compiler / executor, `reciprocal_rank_fusion`,
`extract_years`, `extract_identifier_tokens`, `ResultPhraser`, ingestion, migrations.

**Not in scope until a measurement asks for it:** comparison, similarity retrieval,
map-reduce / document-level reading, per-question-kind profiles other than the default.
(The claim "top_k fails for synthesis" is a hypothesis here: top_k 8 did not help, the
refusal wording and the filter restriction did.)

The new code is built **beside** the old one, proven equal, then the old code is deleted.

## 2. Principles

1. One class, one reason to change. Flat modules under `query/`, one concept per file.
2. Steps never read `Settings`; parameters and collaborators arrive by constructor.
   `Settings` is read in exactly one place, the composition root (`build_query_service`).
3. Typed, explicit results: a refusal is a value (`Declined`), never an empty list.
4. Behaviour first, new features later: the new engine must reproduce today's retrieval
   exactly (characterization tests + `retrieval-snapshot --compare` IDENTICAL).
5. No silent fallbacks; every stage that cuts or refuses says so in the explain record.

## 3. Overview

```
QueryRequest ─► QueryService
                 │
                 ├─ QueryFactsReader.read(question)         → QueryFacts   (identifiers, years: ONCE)
                 ├─ Decider.decide(facts, request)          → Decision
                 │     ├─ AnswerExactly(plan)   → ExactAnswerer (executor + phraser)
                 │     ├─ Refuse(Declined)      → RefusalRenderer
                 │     └─ ReadDocuments(profile name, scope, anchors)
                 │            │
                 │            ├─ ProfileResolver  (name + settings overrides → ResolvedProfile)
                 │            ├─ PipelineFactory.build(profile, deps, scoped store)   per query
                 │            └─ RetrievalPipeline.run(state, observer)
                 │                    steps ─► Answerable(chunks) | Declined(reason, stage)
                 │                 → GroundedAnswerer (prompt policy + answer driver)
                 └─► QueryResult(text, outcome, explain)
```

Two entry methods share the same decision: `answer()` (full) and `retrieve()` (passages
only, for the MCP search tool and the eval).

## 4. Modules and classes

| Module | Contents |
|---|---|
| `query/facts.py` | `QueryFacts`, `QueryFactsReader` |
| `query/outcome.py` | `Answerable`, `Declined`, `DeclineReason`, `RefusalRenderer`, `ModelRefusalRecognizer`; the user-facing refusal texts move here |
| `query/decision.py` | `Decision` (tagged union), `Scope`, `Decider`, `PlanningDecider`, `UnplannedDecider` (temporary), `ScopeResolver`, `ProfileSelector` |
| `query/context.py` | `RetrievalContext` (the one shared, frozen context), `Slot` |
| `query/step.py` | `RetrievalStep` (the abstract contract of a step), `Continue` / `Halt` (`StepResult`), `AuxRecord`, `StepName` |
| `query/candidate_steps.py` | embed, dense search, year widening (dense / keyword), CSLS reorder, keyword search, identifier pin |
| `query/ranking_steps.py` | RRF fusion, rerank, listwise rerank |
| `query/gate_steps.py` | relevance gate (cosine), rerank-score gate |
| `query/selection_steps.py` | top-k with guarantees, cosine cut |
| `query/profiles.py` | `StepSpec`, `ProfileSpec`, `PROFILES`, `ProfileOverrides`, `ProfileResolver`, `StepRegistry`, `PipelineFactory` |
| `query/runner.py` | `RetrievalPipeline`, `StepObserver`, `TraceRecorder`, `AuditLogObserver`, `ProgressLogObserver`, `StageRecord` |
| `query/legacy_trace.py` | `LegacyTraceProjection`: reproduces today's `RetrievalTrace` keys exactly (deliberate, isolated debt) |
| `query/answering.py` | `AnswerPolicy`, `GroundedPromptBuilder`, `GroundedAnswerer`, `ExactAnswerer` |
| `query/service.py` | `QueryRequest`, `QueryService`, `QueryResult`, `RetrievalResult`, `Explain`, `build_query_service` |

### 4.1 Facts and decision
- `QueryFacts` (frozen): `question`, `identifiers`, `years`. The one place identifiers and
  years are detected; years are always computed, a profile decides whether they are used.
  This removes the duplicate identifier detection and the duck-typed `period_filter`.
- `Decision` is a tagged union: `ReadDocuments(facts, plan, profile, scope, anchors)`,
  `AnswerExactly(facts, plan)`, `Refuse(facts, declined)`.
- `PlanningDecider` = today's `QueryRouter.route` without executing or phrasing:
  planning failure → `Refuse(COULD_NOT_INTERPRET)`; `unsupported` → `Refuse(NOT_SUPPORTED)`;
  `as_routed`; exact operation → `AnswerExactly`; a lookup that names an identifier →
  unrestricted `ReadDocuments` with anchors; other lookups → `ScopeResolver`.
- `ScopeResolver` turns a lookup plan into a `Scope` (selection, metadata filter, note) or a
  `Declined(NO_MATCHING_DOCUMENTS)`.
- `ProfileSelector` chooses a profile **name**. One rule today: the configured default. The
  hook exists so the choice belongs to the decision step; no kind-of-question mapping is
  added before a second profile has been measured.
- `UnplannedDecider` serves `QUERY_ROUTER=false` and is deleted with the flag.

### 4.2 Outcome: one place for "no answer"
- `Declined(reason, stage, detail, note)`; `DeclineReason`: `COULD_NOT_INTERPRET`,
  `NOT_SUPPORTED`, `NO_MATCHING_DOCUMENTS`, `NOT_RELEVANT` (cosine gate), `RERANK_REJECTED`
  (cross-encoder gate).
- `RefusalRenderer` produces today's exact texts (both retrieval gates render
  `NO_RESULTS_MESSAGE`; they differ only in the explain record).
- `ModelRefusalRecognizer` recognises the model's own refusal sentence (same constant the
  prompt uses) and sets `explain.model_declined`; it never changes the text.
- The two retrieval gates are **not merged into one step**: they use different signals at
  different points (cosine before fusion, deliberately; cross-encoder after rerank). What
  is centralised is the type and the rendering, not the place.

### 4.3 Context and steps
- **One** frozen `RetrievalContext`, changed with `dataclasses.replace`: fixed `facts`, `scope`,
  and typed optional slots (`query_vector`, `dense_pool`, `keyword_pool`, `ranked`,
  `pins`, `selected`). Slot names describe *what* is held, not the phase that produced it
  (this avoids today's misnamed `listwise` key). New needs add a slot; each slot has one
  writer step. **Decided: slots.**
- `RetrievalStep`: `name`, `requires`, `provides` (sets of `Slot`), `run(context) -> StepResult`.
  `StepResult` is `Continue(state, records, notes)` or `Declined`. Order is validated when a
  profile is resolved (a mis-ordered profile cannot be built); about 30 lines.
- The runner is generic over the state type, so a later document-level state is possible
  without designing it now.

Steps that reproduce today's behaviour (names stable, parameters via constructor):

| Step | Replaces | Note |
|---|---|---|
| `EmbedQueryStep` | embed in `retrieve_chunks` | skipped when a vector is supplied |
| `DenseSearchStep(pool_size)` | `store.search(min_score=0.0)` | `pool_size = max(top_k, POOL_SIZE)` |
| `RelevanceGateStep(depth, min_score)` | `_passes_relevance_gate` | **before** year widening, on the unmerged pool |
| `YearDenseWideningStep` | years block | merge-unique, then stable sort by score |
| `CslsReorderStep` | `_csls_rerank` | stable; scores untouched; no hub score = raw score |
| `KeywordSearchStep` / `YearKeywordWideningStep` | `search_fulltext` | limit = `len(dense_pool)`; the year merge only appends, no sort |
| `RrfFusionStep` | `reciprocal_rank_fusion` | dense first, keyword second |
| `IdentifierPinStep(per_token)` | identifier rescue | pins = all matches; only new ones are prepended; ignores years and `metadata_filter` |
| `RerankStep` | `reranker.rerank` | sees the full list, pins included |
| `RerankScoreGateStep(min_score)` | the `isinstance(CrossEncoderRerankerDriver)` branch | keeps score ≥ min or pinned; empty → `Declined(RERANK_REJECTED)` |
| `ListwiseRerankStep` | `_maybe_listwise_rerank` | only when enabled |
| `TopKWithGuaranteesStep(top_k, diversify, year_quota)` | `_apply_top_k_with_guarantees` | rounds robin by `source_file`; guarantees capped at top_k; year reserve ⌈top_k/2⌉ |
| `CosineCutStep` | `VectorRetrievalStrategy` | |

`RerankStep` and `RerankScoreGateStep` are separate on purpose: `reranked` is recorded
between them and the second one can refuse. The score gate is included in a profile iff the
resolved reranker driver is the cross-encoder (this keeps today's class check without
touching the driver). The small pure helpers are **ported into the step that owns them**,
not imported from the old module; equality is proven by tests, not by shared code.

### 4.4 Profiles
- A profile is **data**: `ProfileSpec(name, steps: [StepSpec(kind, params, toggle)], measured)`;
  `measured` names the `decisions.md` entry and snapshot it was verified with.
- Profiles live only in the `PROFILES` registry (one Python module). `.env` selects a
  **name**; the existing flags are typed overrides applied in one `ProfileResolver`, so each
  flag fans out in one visible place (`DIVERSIFY` → identifier pin + top-k; `PERIOD_FILTER`
  → both widening steps + the year quota). No step lists in `.env`/YAML, no runtime
  composition: every combination would be an unmeasured pipeline.
- Shipped profiles: `hybrid` (embed, dense, gate, [year dense], CSLS, keyword, [year
  keyword], RRF, identifier pin, rerank, [score gate], [listwise], top-k) and `vector`
  (embed, dense, gate, cosine cut). Resolved profiles have a fingerprint, recorded in the
  explain record.
- `PipelineFactory` builds a pipeline **per query**: the store is scoped per query
  (`restricted_to(selection)`), so it cannot be fixed in a step at start-up.
- `ProfileOverrides` replaces today's constructor overrides and the `top_k` override used by
  the eval tools; `top_k` still moves the gate depth and the pool size (kept for now).
- This resolves the open disagreement as a middle way: profiles are data, as asked, but only
  registered, measured ones are valid. **[to confirm]**

### 4.5 Runner, observer, trace
- `RetrievalPipeline.run(state, observer)` opens one store session, asserts the embedding
  dimension, runs steps in order, stops at the first `Declined`.
- Cross-cutting concerns are an **observer called by the runner** around every step, not
  decorators on each step: `TraceRecorder` (uniform `StageRecord`s), `AuditLogObserver`
  (same `LogAction` payloads as today), `ProgressLogObserver`.
- **Retries stay in the drivers** (`retry_policy`). A step-level retry decorator would
  multiply attempts and stretch stalls.
- `LegacyTraceProjection` maps the records onto today's trace keys, quirks included
  (`fused` is the output of the identifier pin; `listwise` is the list entering selection;
  `final` is `[]` after a late decline and absent after the relevance gate), so `funnel`,
  `retrieval-snapshot` and the characterization tests read an unchanged contract. Renaming
  `listwise` → `pre_selection` is a later, announced re-baseline.

### 4.6 Answering and the service
- `AnswerPolicy(partial_coverage, expose_document_date)`; `GroundedPromptBuilder` must be
  byte-identical to today's `_build_prompt` (tested for both flag values). One driver change
  is needed later: `AnswerDriver.generate(system, user)` with `answer()` as a wrapper; it
  comes after the switch, not in the first slice.
- `GroundedAnswerer` (driver + prompt builder + refusal recognizer), `ExactAnswerer`
  (executor + phraser). There is no "unsupported answerer": that case is a `Declined`.
- `QueryService.answer(request) -> QueryResult(text, outcome, explain)` and
  `retrieve(request) -> RetrievalResult`. `query_knowledge_base` and `retrieve_chunks`
  remain as thin wrappers so existing callers keep their signatures.
- `Explain` (typed): facts, decision, profile fingerprint, stage records, notes, final chunk
  ids, `model_declined`, timings. The eval and `funnel` will read it instead of
  re-running retrieval (a later, separate commit: it changes how the eval measures).

### 4.7 MCP and the single entry (**decided: out of scope for now**)
The MCP `search` tool keeps calling `retrieve_chunks` unchanged; what it should become is
left open. For reference, the option considered: the MCP `search` tool uses `QueryService.retrieve()`, the same decision as `answer()`:
documents → excerpts as today; an exact decision → one `exact_result` item; could-not-
interpret / not-supported / no-matching → an explicit tool error; the two retrieval gates →
`[]` (today's contract). Cost: one planner LLM call per MCP search, and a changed response
contract. Therefore its own commit, behind the router flag.

## 5. Parity traps (each pinned by tests or `decisions.md`)

1. The gate looks at the **unmerged** dense pool, depth = `top_k`, `>=`, before year widening.
2. The two declines leave different traces (see 4.5).
3. Keyword and identifier limits are `len(dense_pool)` **after** the year merge, not the pool
   size (a later, deliberate-change candidate, kept as is).
4. The dense year merge sorts (ties: primary first); the keyword year merge only appends.
5. CSLS: stable, scores untouched, missing hub score = raw score.
6. RRF: dense list first, `setdefault` keeps the dense chunk object, ranks from 1, k = 60.
7. Identifiers are read from the raw question; the identifier search ignores `metadata_filter`
   and years but honours the store restriction.
8. The reranker sees the whole fused list; the score gate keeps `score >= min or pinned`;
   with the dev `.env` (`RERANKER_DRIVER=vertex`) the gate never runs live, so the parity
   run must be repeated with `RERANKER_DRIVER=cross_encoder`.
9. Listwise runs on the post-gate list; it is LLM-nondeterministic, so compare it with a
   stubbed disambiguator or only up to `reranked`.
10. Selection groups by `source_file`; guarantees capped at `top_k`; output order:
    guaranteed, year extras, the rest; the vector profile never uses years.
11. `metadata_filter` reaches dense, keyword and both year searches, never the identifier
    search.
12. Router texts: `"\n\n"` before an appended note on answers, `" "` inside "No documents
    match"; the wording that `decline_detection` relies on must not change.
13. One store session for the whole run; `restricted_to` is created before it opens; the
    embedding driver is still constructed when a query vector is supplied (its dimension is
    used).
14. The characterization harness patches module globals of `query.retrieval`; the new engine
    takes everything by injection, so the harness gets a second runner per engine with the
    expected results unchanged.

## 6. Build plan

**Slice 1 – retrieval only, no user-visible change**
1. `test:` parametrise the characterization harness over engines (`legacy` only; green).
2. `feat(query):` facts, outcome types, state, step contract, runner, trace recorder, legacy
   projection; unit tests with toy steps (order validation, decline stops the run, projection
   quirks).
3. Vector profile steps + `vector` profile; add `v2` to the harness.
4. CSLS, keyword, RRF, identifier pin, rerank, score gate, top-k.
5. Year widening, year quota, listwise; all 14 scenarios green on both engines.
6. `QueryService.retrieve` + a temporary `QUERY_ENGINE=legacy|v2` read only in the
   composition root; `retrieval-snapshot` honours it. **Decided: yes, but only for testing**
   (it exists to compare the two engines and is removed at the end). Live gate: baseline twice (matches itself), then `--compare` v2 = IDENTICAL; repeat
   under `RETRIEVAL_PERIOD_FILTER=true`, `RERANKER_DRIVER=cross_encoder`,
   `RETRIEVAL_STRATEGY=vector`.

**Slice 2:** the decider (port the router test cases), `QueryService.answer` behind the flag,
golden eval on both engines (retrieval identical, answers within LLM noise), then the MCP
change as its own commit, then the switch.

**End:** delete the legacy code and `QUERY_ENGINE`; the eval and `funnel` read `explain`;
the prompt-builder driver change.

**Deleting `QUERY_ROUTER` and `ANSWER_PARTIAL_COVERAGE` is not part of the refactor.** Both
default to *off*, so deleting them changes the default behaviour. Each is its own measured
commit after parity, conditioned on the full `--repeat 3` golden run.

Left out of the first slice: prompt builder and driver change, decider, MCP, kind-of-question
profiles, trace key renames, map-reduce, comparison, similarity.

## 7. Risks

- Listwise nondeterminism blocks an "IDENTICAL" claim for that step (see trap 9).
- The score gate is unexercised live under `vertex` (trap 8).
- Eval numbers shift slightly once the eval stops retrieving twice: a measurement change,
  its own commit.
- Over-engineering: keep requires/provides to an enum plus about 30 lines; no YAML, no
  decorator layer, no plugin registry, typed `Explain` (not a dict).
- Two pipelines must not live side by side for long: the flag, the parity proof and the
  deletion of the old code are the definition of done.

## 8. After parity: facts resolved into the scope

Today neither the identifier nor the date reaches the retrieval from the metadata: the
planner's filters (over `document_meta`) become the `Scope`, while the retrieval has its
own regex facts (`extract_years` filters the chunk's `document_date`; identifier tokens
drive `search_by_identifier`). Two mechanisms, one of them narrowing and one widening or
pinning. The direction after parity, **measured, never as part of the parity slices**:

- A **fact resolver** per kind of fact (identifier, date, later others) turns a fact of the
  question into a set of documents or a document condition; the `Scope` is composed from
  them. `ScopeResolver` becomes a composition of `ScopeSource`s: the planner's filters
  (which already resolve dates, with `DateRangeResolver`) and a deterministic
  `IdentifierResolver` (no LLM).
- **Combination:** between kinds of fact AND (they narrow); within one kind OR (several
  identifiers, several years).
- **`IdentifierResolver`:** normalises (punctuation, case, known suffixes) and matches the
  normalised identifiers of the documents (exactly, or by prefix); it returns, per
  identifier, a **set** of document ids (zero, one or several: the same value can be an
  invoice number and another document's case number, and nothing is picked between them;
  ranking decides by content). The scope is the union.
- With a resolved scope the retrieval runs only over those documents, so the pin is not
  needed for them. `IdentifierPinStep` remains for **unresolved** identifiers (a reference
  to a case rather than the case itself, or missing metadata) and the explain record says
  so. The round-robin by document stays, a multi-document scope needs it.
- **Safeguards (the step-1 regression):** every narrowing reports the documents it could not
  decide; a scope of zero documents is an explicit `Declined`, never a silent empty search;
  a partly resolved question (one identifier found, one not) keeps the found ones in the
  scope and searches the text for the rest.
- **Open, to measure:** when the plan already narrows by date, does the regex year
  widening (`YearDenseWideningStep`, the year quota) still add anything? And for
  identifiers: single-document questions (fact_finder, adversarial, practical personas) must
  not get worse, and a "which cases cite X" question needs its own case.

## 9. Decisions still needed

Decided: MCP stays out for now (4.7); state uses named slots (4.3); `QUERY_ENGINE` exists
only for testing (6).

Still to confirm: profiles as the middle way in 4.4 (data, registered and measured only).
