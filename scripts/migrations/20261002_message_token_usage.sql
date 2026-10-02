-- Additive migration: preserves every message and any usage already recorded.
BEGIN;
SET LOCAL lock_timeout = '5s';
DO $$
BEGIN
    IF to_regclass('message') IS NULL THEN
        RAISE EXCEPTION 'No message table exists; check the database and search_path';
    END IF;
END $$;
ALTER TABLE message ADD COLUMN IF NOT EXISTS token_usage jsonb;
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_attribute
        WHERE attrelid = 'message'::regclass AND attname = 'token_usage'
          AND atttypid = 'jsonb'::regtype AND NOT attnotnull AND NOT attisdropped
    ) THEN
        RAISE EXCEPTION 'message.token_usage must be nullable jsonb; existing column was not changed';
    END IF;
END $$;
COMMIT;
