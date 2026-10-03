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

**The given `court` and `case_number` are both only best-effort guesses,
reconstructed from the filename — neither is authoritative, and both are
confirmed, live, to sometimes be wrong.** `case_number` can have missing
prefixes, wrong punctuation, or even the wrong trailing number entirely —
see `docs/decisions.md` for a real example. `court` is reconstructed by
replacing the filename's underscores with spaces, so it is plain ASCII with
no Hungarian diacritics (e.g. "Budapest Kornyeki Torvenyszek" instead of
"Budapest Környéki Törvényszék") — confirmed live that a drafted question's
prose correctly used the real, accented court name read from `content`,
while its structured `citations[].court` field still echoed the unaccented
filename-derived hint verbatim, instead of the same real name already used
in the prose. Before writing a citation's `court` or `case_number`, find
both as they're actually written in the given `content` itself (the court
name typically opens the document; the case number is typically near "Az
ügy száma:" or similar) and use *those* verbatim — never either given hint
unread. If you cannot find a real case number written in the content, say
so in `notes` rather than guessing or falling back to the hint.

**If you cannot find a verifiable court/case number for one of the given
samples, do not cite that document at all — pick a different sample
instead of drafting a citation with blank or guessed fields.** Confirmed
live: a drafted `synthesizer` question correctly left a citation's
`court`/`case_number` blank rather than guessing (the right call per the
instruction above), but still went ahead and cited that document as one
of two sources — the blank fields then failed verification's verbatim
check anyway, for a reason that was foreseeable at drafting time. Leaving
fields blank is only safe to do in `notes`, as a *reason you're not using*
a sample; it's not a safe way to still use one in `citations`. For a
multi-document persona, this means: if a given sample doesn't yield a
verifiable identifier, draft the question around the sample(s) that do,
even if that means citing only one document instead of the number you
were given, or saying so in `notes` and requesting a different sample
rather than forcing a multi-document question out of what you have.

**`source_file`, unlike `court`/`case_number`, IS authoritative — copy it
character-for-character from the given sample, never "corrected."**
Confirmed live: a drafted citation's `source_file` read
`Budapest_Kornyeki_Torvenyszék__...` (with an accented "é") when the real
filename is `Budapest_Kornyeki_Torvenyszek__...` (plain ASCII, like every
filename in this corpus) — the model "fixed" the court name's spelling
inside what must be an exact, literal filename, breaking the lookup this
citation depends on. Hungarian institution names are correctly spelled
*with* diacritics in prose (the question text, `expected_answer`, and the
citation's `court` field once read from real content) — but `source_file`
is a filename, not prose, and this corpus's filenames are deliberately
unaccented; don't "improve" it.

Draft exactly one question matching the persona's `question_style`, using
**only** facts that actually appear in the given content. Do not invent case
numbers, dates, amounts, or legal reasoning not present in the text.

**For multi-document personas (`document_scope: "multi"`, e.g.
`precedent_seeker`, `synthesizer`) that don't cite a case number in the
question itself: the question must still carry enough distinguishing detail
to point at the specific cited document(s), not just at a court and a broad
topic.** The corpus can contain multiple, near-identical anonymized
boilerplate decisions from the *same court* on the *same broad topic*
(placeholder party names, same year, same one-line case type) that differ
only in a narrower sub-topic or outcome — confirmed live: a question like
"Milyen ügyekben hozott ítéletet a Debreceni Törvényszék kisajátítási
ügyben?" matched two real, unrelated documents equally well, because the
one fact that actually distinguished them (one judgment upheld the claim in
part, the other rejected it outright) only ended up in `expected_answer`,
never in the question (see `docs/decisions.md`'s 2026-10-01 entry). Before
finalizing such a question, check: does it already name the specific
outcome, sub-topic, or legal principle you're about to write into
`expected_answer` — the detail that makes *this* document the right one,
not just *a* plausible one? If not, rewrite the question to include it.

**Never phrase a question as if the reader can already see the document(s) you were given.** You see the source chunks/sample while drafting; a real user asking the deployed system never does. Confirmed live: a drafted `precedent_seeker` question read "...Törvényszék 2022-ben *az alábbi esetekben*..." — "the following cases" — a phrase that only makes sense if the reader is looking at the same document list you were, which no real user is. The same leak shows up as "a megadott korpuszban/dokumentumokban" ("in the given corpus/documents") or "a fentebb említett ügyekben" ("in the above-mentioned cases"). Any such phrase makes the question under-specified for retrieval (it points at "whatever I was shown," not at a findable fact) and unrealistic as a user query. Before finalizing, check the question for "alábbi", "megadott", "fenti"/"fentebb", "ezen/ezek az esetek" or any other wording that implicitly points back at your own drafting context, and rewrite using only self-contained facts (court, year, topic, outcome, etc.) that *name* what you mean instead of pointing at it.

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

1. Does each cited `source_file` actually match a real ingested document?
   (This check is deterministic — done in code, not by you — see
   `corpus/commands/generate_questions.py`'s `verify_citation_exists()`,
   which matches on `source_file` alone, not `court`/`case_number` — those
   two are free text, not part of this check. Skip straight to step 2.)
2. **Does each cited `case_number` appear verbatim (or near-verbatim — minor
   whitespace differences are fine, but not missing/extra segments) in that
   document's real content, and does each cited `court` name match the real,
   accented court name as written in that content (not the unaccented
   filename-derived hint)?** This is a literal text-presence check, unlike
   step 3 — confirmed live, twice, that a citation can cite the wrong case
   number, or echo the unaccented filename-derived court name, while the
   *answer's content* still reads as substantively correct (the drafting
   model copied an unreliable hint instead of reading the real value from
   the content — see Part 1). If the exact case number string isn't
   findable in the content, or the cited court name doesn't match the real
   accented name in the content, the verdict must be `NOT_SUPPORTED`
   regardless of step 3's answer.
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
