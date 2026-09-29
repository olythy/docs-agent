# skills/

This project's own Claude Code skills — canonical source, not the copy Claude Code actually loads.

`make skills-install` (`scripts/agent_cli.py`'s `install_skills()`) symlinks this whole directory into `.claude/skills/`, so editing a skill here takes effect immediately, with no reinstall step, and adding a new skill subdirectory here needs no reinstall either (the symlink covers the whole tree, not one link per skill).

Each skill's `SKILL.md` is written to double as its own content: the same file is both the interactive Skill body Claude Code follows when invoked directly, and the raw prompt text a companion script (when one exists) sends to the project's own `LLM_DRIVER` for unattended, batched runs — one recipe, not duplicated across an interactive and an automated copy.

Currently empty — the first skill here will be for the large-scale RAG accuracy evaluation work (question generation per user profile, golden-set review).
