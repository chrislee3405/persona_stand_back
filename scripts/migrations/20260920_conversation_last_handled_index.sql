-- Adds conversation.last_handled_index, the readiness gate's pending-message
-- cursor. `create_all` creates missing TABLES but never adds a column to an
-- existing one, so an existing database needs this run once.
--
-- EXISTING CONVERSATIONS START FULLY HANDLED, not at -1. The pending group is
-- "every user row above the cursor", so -1 on a conversation with history made
-- its first turn after deployment treat EVERY earlier user message as one
-- unanswered question: all of them were joined into the message being
-- answered and dropped from the prompt history. Worse, a rejected message on
-- that turn (privacy, length, rate limit) releases the pending group, which
-- retags every one of them not_saved_user -- the visitor's side of the
-- conversation gone from every later prompt. Every message that exists before
-- this release was handled by the old backend, which answered each one or
-- failed it, so the cursor is backfilled to the conversation's last row.
--
-- The column is added and backfilled together, and ONLY on the run that adds
-- it. A rerun changes nothing: re-backfilling a cursor the gate is already
-- maintaining would mark genuinely held messages as handled.
--
-- Run it with the old backend stopped, immediately before starting the new
-- one. A turn the old backend answers after the backfill sits above the
-- cursor, so the new backend would read it once more as pending.
BEGIN;
SET LOCAL lock_timeout = '10s';
DO $$
BEGIN
    IF to_regclass('conversation') IS NULL THEN
        RAISE EXCEPTION 'No conversation table exists; use the new application schema for a fresh database';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'conversation'
          AND column_name = 'last_handled_index'
    ) THEN
        ALTER TABLE conversation
            ADD COLUMN last_handled_index integer NOT NULL DEFAULT -1;

        UPDATE conversation AS c
           SET last_handled_index = latest.max_index
          FROM (
                SELECT conversation_id, max(order_index) AS max_index
                  FROM message
                 GROUP BY conversation_id
               ) AS latest
         WHERE latest.conversation_id = c.conversation_id;
    END IF;
END $$;
COMMIT;
