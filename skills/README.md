# skills/

This project's own Claude Code skills — canonical source, not the copy Claude Code actually loads.

`make skills-install` (`scripts/agent_cli.py`'s `install_skills()`) symlinks this whole directory into `.claude/skills/`, so editing a skill here takes effect immediately, with no reinstall step, and adding a new skill subdirectory here needs no reinstall either (the symlink covers the whole tree, not one link per skill).

Each skill's `SKILL.md` is written to double as its own content: the same file is both the interactive Skill body Claude Code follows when invoked directly, and the raw prompt text a companion script (when one exists) sends to the project's own `LLM_DRIVER` for unattended, batched runs — one recipe, not duplicated across an interactive and an automated copy.

## `generate-golden-questions`

Drafts entries for `corpus/golden_set/questions.json` (question + expected answer + citation, per persona defined in `corpus/golden_set/personas.json`), grounded in real content from the already-ingested `document_chunks` table.

Every draft — whether written interactively by Claude Code following `SKILL.md`, or by `cli.py` sending the same instructions to the project's own `LLM_DRIVER` for an unattended run — goes through a two-tier verification before being trusted: a deterministic citation-existence check, then a *separate* LLM call judging whether the cited content actually supports the drafted answer. Anything that doesn't clearly pass gets `verification_status: "needs_review"` instead of being silently accepted — see `SKILL.md` for the full reasoning, and `docs/decisions.md` for why (large-scale, legal-domain content, where a "zero wrong citations" target makes ungrounded generation a real risk, not just a nice-to-have check).
