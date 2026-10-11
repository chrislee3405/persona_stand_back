-- Run after 20261007_model_configuration_providers.sql with old backend stopped.
-- Preserve all existing provider settings and enabled states.
BEGIN;
SET LOCAL lock_timeout = '5s';
LOCK TABLE model_configuration IN ACCESS EXCLUSIVE MODE;
DROP INDEX IF EXISTS mc_one_active;
CREATE UNIQUE INDEX IF NOT EXISTS mc_enabled_priority_unique
    ON model_configuration (use_priority) WHERE is_active;
COMMIT;
