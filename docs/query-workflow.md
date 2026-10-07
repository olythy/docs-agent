# The question-to-answer workflow

> **Status (2026-10-07):** this describes the **original** implementation (`QUERY_ENGINE=legacy`, still the default), which is the reference the rewrite is proven against. A step-based retrieval now exists beside it (`QUERY_ENGINE=v2`): it removes the retrieval-side tangles listed in section 2 (the hidden second refusal, the class check for the rerank threshold, identifiers and years detected in several places, the misnamed `listwise` trace key, the funnel reading a trace the eval never produced). The decision side (the router) is not rewritten yet. See `docs/query-pipeline-design.md` for the design and its build status.

This document walks the path of one question, from `query_knowledge_base` to the
answer, at file and method level. It records **what the code does today** (section 1),
**where decisions are made and where it is tangled** (section 2), **the two flows** we
agreed on (section 3), and **the direction** with the one open question (section 4).
It is a design document: no code changes are implied by it.

Line numbers are deliberately left out; names are stable, lines are not.

## 0. Who calls what

| Caller | Calls | Goes through the router? |
|---|---|---|
| `scripts/agent_cli.py` (`query` command) | `query_knowledge_base(question)` | yes, if `QUERY_ROUTER=true` |
| `corpus/commands/eval.py` | `query_knowledge_base(...)` and, separately, `retrieve_chunks(...)` (to grade with the retrieved passages) | routing decided once, then both reuse it; but retrieval still runs **twice** |
| `agent.py` / `mcp_server.py` (`search` tool) | `retrieve_chunks(question)` **directly** | **no**: MCP search never plans, never counts exactly, never says "not supported" |
| `corpus/commands/funnel.py`, `retrieval_snapshot.py`, `compare_retrieval.py` | `retrieve_chunks(...)` | no (diagnostics of the retrieval part only) |

So the router lives only in front of `query_knowledge_base`. That is a fact worth
remembering when we talk about "the pipeline": there are two entries, and only one has
the decision step.

## 1. As-is map

```
query_knowledge_base(question, ...)                       query/retrieval.py
│
├─ [if settings.QUERY_ROUTER and routing is None]
│    get_query_router().route(question)                   query/router.py
│    │
│    ├─ load_catalogs(DocumentStore)                      metadata/planner.py   (approved types + their approved keys)
│    ├─ collect_known_values(...)                         metadata/planner.py   (stored text values, shown to the planner)
│    ├─ LLMQueryPlanner.plan(question, catalogs, known)   metadata/planner.py   ← LLM call #1 (JSON plan)
│    │     ├─ _prompt(...)                                (the _TEMPLATE: operations, types, filter grammar, rules)
│    │     ├─ parse_plan(extract_json(reply, strict), type_names)   metadata/plan.py
│    │     ├─ PlanCompiler.compile(plan, keys)            metadata/compiler.py  (validates; skipped for `unsupported`)
│    │     └─ one retry with the error text; then PlanningFailed
│    │
│    ├─ PlanningFailed        → Routing(answer=COULD_NOT_INTERPRET_MESSAGE)            ── ends
│    ├─ operation UNSUPPORTED → Routing(answer="… not supported yet … (reason)")       ── ends
│    ├─ as_routed(plan)       exact operation + residual ⇒ treated as LOOKUP
│    ├─ operation LOOKUP
│    │     ├─ extract_identifier_tokens(question) non-empty → Routing()  (unrestricted; retrieval pins the document)
│    │     └─ _restricted_lookup(plan)
│    │           ├─ no type, no filters → Routing()                       (unrestricted)
│    │           ├─ PlanExecutor.execute(plan) → count + selection (a sub-select)
│    │           ├─ count 0 → Routing(answer="No documents match the filter (…)")   ── ends
│    │           └─ else    → Routing(selection=…, note=<undecidable docs>)
│    └─ operation COUNT / LIST / SUM / OVERVIEW
│          PlanExecutor.execute(plan)                     metadata/executor.py  → compile → SQL → PlanResult
│          ResultPhraser.phrase(question, plan, result)   query/router.py       ← LLM call #2 (wording only;
│                                                          every figure must reappear, else plain facts)
│          → Routing(answer=…)                                                                 ── ends
│
├─ routing.answer is not None → return it                 (the exact / refusal paths end here)
├─ apply_routing(routing, store)   selection → VectorStore.restricted_to(selection)
│                                  (`document_id IN (sub-select)` on every chunk query)
│
└─ _answer_from_documents(...)                            query/retrieval.py   ── flow A
     ├─ retrieve_chunks(question, ...)
     │    ├─ embed_query(question)                        drivers/embedding.py
     │    ├─ store.search(vector, top_k=candidate_k, min_score=0.0)       "vector" stage (pool of 20)
     │    ├─ _passes_relevance_gate(pool, top_k, min_score)               gate #1: cosine, over pool[:top_k]
     │    │      fails → return []                        (→ NO_RESULTS_MESSAGE)
     │    ├─ get_retrieval_strategy()   RETRIEVAL_STRATEGY: "hybrid" | "vector"
     │    ├─ years = extract_years(question)  only if strategy.period_filter (duck-typed getattr)
     │    │      years → second, year-restricted vector pool, merged ("vector_years")
     │    └─ strategy.select_chunks(question, pool, store, top_k, min_score, ...)
     │
     │         VectorRetrievalStrategy: filter by min_score, cut to top_k.
     │
     │         HybridRetrievalStrategy.select_chunks:
     │           1. _csls_rerank(pool)                    re-order by 2*score − hub_score        "vector_csls"
     │           2. store.search_fulltext(question)       Postgres FTS, `hungarian`               "fulltext"
     │              [+ years: second FTS pool, merged]                                            "fulltext_years"
     │           3. reciprocal_rank_fusion(vector, fts)   query/hybrid.py                         "fused"
     │           4. identifier rescue: extract_identifier_tokens(question) (second detection!)
     │              → store.search_by_identifier(...)  merged in front of the fused list          "identifier"
     │           5. get_reranker_driver → reranker.rerank(question, fused)                       "reranked"
     │           6. IF reranker is a CrossEncoderRerankerDriver (isinstance):
     │                 keep score ≥ RERANKER_MIN_SCORE or identifier match       gate #2
     │                 nothing kept → return []  (→ NO_RESULTS_MESSAGE, a second hidden "no result")
     │           7. _maybe_listwise_rerank(...)           LISTWISE_RERANK_ENABLED → LLM call      "listwise"
     │           8. _apply_top_k_with_guarantees(chunks, identifier ids, top_k,
     │                 diversify=RETRIEVAL_DIVERSIFY_GUARANTEES, years)
     │                 identifier chunks first (round-robin over documents when diversify),
     │                 then ≥ ceil(top_k/2) from the question's years, then best by score
     │
     │    └─ trace.record("final", chunks)
     ├─ no chunks → NO_RESULTS_MESSAGE
     └─ get_answer_driver().answer(question, context_chunks)       ← LLM call #3 (the answer)
           drivers/llm.py `_build_prompt`: chunks reversed (best last), "[i] Source: file[, date], page n",
           ANSWER_PARTIAL_COVERAGE chooses the refusal clause (+ the "sample" note),
           EXPOSE_DOCUMENT_DATE shows each excerpt's date.
```

The answer is `f"{answer}\n\n{routing.note}"` when the router left a note.

### Settings and where each is read

| Setting | Read in | Effect |
|---|---|---|
| `QUERY_ROUTER` | `query_knowledge_base` | whether the router runs at all (default **off**) |
| `RETRIEVAL_TOP_K` (4) | `retrieve_chunks` | final number of passages; also the **depth of gate #1** |
| `RETRIEVAL_MIN_SCORE` (0.25) | `retrieve_chunks` | cosine threshold of gate #1 (and of the vector strategy's filter) |
| `RETRIEVAL_CANDIDATE_POOL_SIZE` (20) | `retrieve_chunks` | size of every candidate pool |
| `RETRIEVAL_STRATEGY` | `get_retrieval_strategy` | `hybrid` or `vector` |
| `RETRIEVAL_PERIOD_FILTER` | `HybridRetrievalStrategy.__init__` → `.period_filter` | year widening + the year quota in step 8 |
| `RETRIEVAL_DIVERSIFY_GUARANTEES` | `HybridRetrievalStrategy` | round-robin by document in step 4 (`per_token`) **and** step 8 |
| `RERANKER_DRIVER` / `RERANKER_MODEL` | `get_reranker_driver` | which reranker (the dev `.env` has `vertex`) |
| `RERANKER_MIN_SCORE` | step 6 | only for the cross-encoder class |
| `LISTWISE_RERANK_ENABLED`, `_MAX_CANDIDATES` | step 7 | optional LLM reorder |
| `ANSWER_PARTIAL_COVERAGE` | `_build_prompt` | refusal clause; both flags are to be deleted at the end |
| `EXPOSE_DOCUMENT_DATE` | `_build_prompt` | dates in the excerpts |
| `API_REQUEST_TIMEOUT_SECONDS` | `drivers/llm.py`, `drivers/embedding.py` | every API call |

### LLM calls per question

* exact question (count/list/sum/overview): planner + phraser = 2 calls, no retrieval;
* lookup: planner + answer = 2 calls (+1 with listwise);
* router off: the answer call only.

## 2. Decision points and known tangles

Where each decision is made, and what is wrong with it today. The first six come from the
independent Opus review (verified against the code), the rest are what reading the path
again showed.

| # | Decision | Made in | Tangle |
|---|---|---|---|
| 1 | Which flow the question takes | `QueryRouter.route` (planner + `as_routed` + identifier check) | The identifier check re-detects what retrieval detects again (see 5). `as_routed` lives in the router file but is also used by `routing-eval` and `eval`. |
| 2 | "Is there anything relevant at all" | `_passes_relevance_gate` (cosine) | Depth is `top_k` (a retrieval parameter), so changing `top_k` changes how often we refuse. |
| 3 | A second "nothing relevant" | `select_chunks` step 6 (cross-encoder acceptance) | Hidden inside a strategy, returns `[]` that the caller cannot tell apart from gate #1. |
| 4 | Rerank threshold | `isinstance(reranker, CrossEncoderRerankerDriver)` | Behaviour depends on a *class*, not on a declared property of the driver. The Vertex reranker (the dev default) skips it. |
| 5 | Identifiers | router (`extract_identifier_tokens`) **and** `select_chunks` step 4 | Detected twice, in two files. |
| 6 | Years | `retrieve_chunks` via `getattr(strategy, "period_filter")`, then again in `select_chunks` and step 8 | Duck-typed attribute; one flag drives three places. |
| 7 | `RETRIEVAL_DIVERSIFY_GUARANTEES` | step 4 and step 8 | One flag, two effects. |
| 8 | Trace key `listwise` | step 6/7 | Also written when listwise is off (holds the post-threshold list). `funnel` reads it. |
| 9 | "No answer" | `route` (could not interpret / not supported / no match), `retrieve_chunks` (gate), step 6, `_build_prompt` refusal clause | Decided in four places, three of them return different messages; `decline_detection` has to know all the wordings. |
| 10 | Entry points | `mcp_server.py` calls `retrieve_chunks` without the router | The MCP `search` tool cannot count and cannot say "not supported". |
| 11 | Evaluation | `eval.py` retrieves twice (once for the grader, once inside `query_knowledge_base`); `funnel` never routes | What the eval grades is not exactly what ran. |
| 12 | The restriction | `Routing.selection` → `apply_routing` → `VectorStore.restricted_to` | Clean, but only a *lookup* gets it; the answer path cannot say "restrict, and also pin this document". |

None of these is a bug that changes answers today; they are the reason it is hard to see
"what happens where, when and why", and they are what any further change (profiles,
similarity, comparison) would have to untangle first.

## 3. The two flows (agreed)

**Flow A – the best passages contain the answer.** Pinpoint questions ("what did the
court decide in case X"), content questions ("why did the court reject the claim"), and,
for now, synthesis questions ("how did the practice develop"). Today's retrieval,
optionally restricted to the documents the plan's filters select, the identifier pinned.
Synthesis stays here until a measurement shows that top_k is not enough.

**Flow B – top_k is surely not the solution.** Everything where reading four passages
cannot give the answer:

| Intent | State |
|---|---|
| count / list / sum / overview (exact, from the metadata) | built: planner → executor → phraser |
| similar cases to a named one | **recognised**, answered "not supported yet" (`unsupported`) |
| comparison of named documents | not built (today it is read as a lookup) |

The routing is measured by `routing-eval` (13 cases, planner only, 42/42 at the last run);
the answers by the golden-set `eval` personas.

## 4. Direction (not started)

Agreed order: first make A and B "covered properly" (more routing cases, then
measurements), and only then restructure. What the restructuring would be:

1. **Pull the steps out of `HybridRetrievalStrategy.select_chunks`** into named
   functions with the same behaviour (safety net exists: the 14 characterization
   scenarios plus `retrieval-snapshot --compare` against the baseline, which must stay
   IDENTICAL).
2. **Phases with strategies**: candidates (vector / full-text / identifier), fusion,
   rerank (a `RerankerStrategy` per kind), selection. A step is not always "list in,
   list out" (the identifier match is a *pin*, the gate is a *decision*), so each phase
   gets its own small contract over a shared accumulating state.
3. **Profiles**: a few presets chosen by the *kind of question* (pinpoint, content,
   synthesis) plus a default. The choice belongs to the decision step, not to retrieval.
4. **An explain record** returned with the answer (which flow, which plan, which stages
   kept how many chunks, why a refusal), so `eval` and `funnel` read what actually ran
   instead of re-running retrieval.
5. **One place for "no answer"** instead of four.
6. Later: similarity retrieval, the comparison flow, the two answerers
   (structured / document) split, removing the two flags.

### The open question

How are profiles defined?

* **Composed from configuration** (the user's view): the steps of a profile are listed in
  config and registered like middleware; presets are just named lists; the runtime can
  also change them.
* **Named frozen Python profiles** (the Opus review's view): a profile is a small frozen
  dataclass in code, `.env` selects a *name*, and existing flags override single fields.
  Argument: every combination of steps is an unmeasured combination; named profiles are
  the ones we have evaluated.

A middle way worth discussing: profiles *are* data (an ordered list of step names with
parameters, as the user wants), but only the ones listed in one registry module are
valid, and each shipped profile carries the golden-set result it was measured with.
Nothing here is decided.

### Questions for the next discussion

1. Is the map in section 1 what you expected, and is anything in it surprising?
2. Section 2, items 3 and 9 (the hidden second refusal and "no answer" in four places):
   should these be fixed *before* anything else, since they are the least visible?
3. Item 10: should the MCP `search` tool go through the router too, or stay a pure
   retrieval tool on purpose?
4. Profiles: composed from config, named in code, or the middle way?
