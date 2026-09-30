---
name: generate-golden-questions
description: Draft golden-set evaluation questions (question + expected answer + citation) from real, already-ingested corpus documents, per persona defined in corpus/data/personas.json.
---

# Generate Golden Questions

This skill drafts entries for `corpus/data/questions.json`, grounded in real
content from the ingested `document_chunks` table — never invented facts. It
is used two ways (see `skills/README.md` for why the same text serves both):

- **Interactively, inside Claude Code**: read the real chunk content given
  below, and draft the question/answer/citation yourself, using your own
  judgment — this is the higher-quality path, since you can notice when a
  chunk doesn't actually support a clean question and ask for a different
  sample instead of forcing one.
- **Unattended, via `corpus/cli.py`'s `generate-questions` command**: the same
  instructions below are sent as a prompt to the project's own `LLM_DRIVER`
  (whatever is configured — Gemini, OpenRouter, or OpenAI) and the response
  is parsed as JSON. This path is noisier (no judgment call to skip a bad
  sample), which is exactly why every draft — from either path — goes
  through the verification step before being trusted.

## Part 1 — Drafting a question

You will be given:
- One persona object from `corpus/data/personas.json` (id, description,
  question_style, document_scope, citation_expectation).
- One or more real chunks: `(source_file, court, case_number, content)` tuples
  pulled from `document_chunks`.

**The given `case_number` is only a best-effort guess, reconstructed from
the filename — it is NOT authoritative and is confirmed, live, to sometimes
be wrong** (missing prefixes, wrong punctuation, or even the wrong trailing
number entirely — see `docs/decisions.md` for a real example). Before
writing a citation's `case_number`, find the actual case number as it's
written in the given `content` itself (typically near "Az ügy száma:" or
similar) and use *that* verbatim — never the given hint unread. If you
cannot find a real case number written in the content, say so in `notes`
rather than guessing or falling back to the hint.

Draft exactly one question matching the persona's `question_style`, using
**only** facts that actually appear in the given content. Do not invent case
numbers, dates, amounts, or legal reasoning not present in the text.

For the `adversarial` persona specifically: you will *not* be given real
content to ground it in — instead, invent a case number/court combination
that is clearly fabricated (e.g. an implausible year or collegium code), or
ask about a real-sounding fact that the given content does not actually
contain. The correct `expected_answer` for this persona is always an
explicit statement that the information isn't available — never a
plausible-sounding fabricated answer.

Output **exactly** this JSON shape (no surrounding prose):

```json
{
  "question": "...",
  "expected_answer": "...",
  "citations": [
    {"court": "...", "case_number": "...", "source_file": "..."}
  ],
  "difficulty": "easy | medium | hard",
  "notes": "one sentence on how this was grounded, or why it's a deliberate trap"
}
```

`citations` is an empty list `[]` only for the `adversarial` persona.

## Part 2 — Verifying a drafted question

Given a drafted question (as above) and the **real, full content** of each
cited `source_file`, answer only:

1. Does each cited `(court, case_number)` actually match a real ingested
   document? (This check is deterministic — done in code, not by you — see
   `corpus/commands/generate_questions.py`'s `verify_citation_exists()`. Skip straight to step 2.)
2. **Does each cited `case_number` appear verbatim (or near-verbatim — minor
   whitespace differences are fine, but not missing/extra segments) in that
   document's real content?** This is a literal text-presence check, unlike
   step 3 — confirmed live that a citation can cite the wrong case number
   while the *answer's content* still reads as substantively correct (the
   drafting model copied an unreliable filename-derived hint instead of
   reading the real case number from the content — see Part 1). If the
   exact case number string isn't findable in the content, the verdict must
   be `NOT_SUPPORTED` regardless of step 3's answer.
3. Does the cited content actually support `expected_answer`, in substance
   (paraphrasing is fine — this is not a string-match check)? Answer with
   exactly one of:
   - `SUPPORTED` — the content clearly backs the claimed answer AND step 2 passed.
   - `NOT_SUPPORTED` — the content contradicts or doesn't mention the claim,
     or step 2 failed (wrong/unfindable case number).
   - `UNCLEAR` — partially supported, ambiguous, or you're not confident either way.

   Follow your verdict with one sentence explaining why, explicitly noting
   whether it was step 2 or step 3 that failed if the verdict isn't
   `SUPPORTED`. This check must be run by a *separate* call from whichever
   process drafted the question — never let the same generation pass grade
   its own work.

Output for this part:

```json
{"verdict": "SUPPORTED | NOT_SUPPORTED | UNCLEAR", "reason": "..."}
```
