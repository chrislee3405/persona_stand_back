-- For databases that already have model_configuration. Safe to rerun.
-- Reserved metadata only: this does not enable provider fallback.
BEGIN;
SET LOCAL lock_timeout = '5s';
ALTER TABLE model_configuration
    ADD COLUMN IF NOT EXISTS use_priority integer NOT NULL DEFAULT 1
    CONSTRAINT mc_use_priority_positive CHECK (use_priority > 0);
COMMIT;
