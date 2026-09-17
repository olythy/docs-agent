-- Runs once, on first container startup (empty data dir). Only creates the
-- test database — schema/extension setup stays owned by migrations/, so
-- there's one canonical source of truth for the schema.
CREATE DATABASE docs_agent_test;
