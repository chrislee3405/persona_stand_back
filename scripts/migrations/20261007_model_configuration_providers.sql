-- Run after the model_configuration and use_priority migrations, before deployment.
-- Keeps the existing provider/models active and adds missing provider presets.
BEGIN;
SET LOCAL lock_timeout = '5s';
LOCK TABLE model_configuration IN ACCESS EXCLUSIVE MODE;
ALTER TABLE model_configuration DROP CONSTRAINT IF EXISTS model_configuration_singleton;
ALTER TABLE model_configuration DROP CONSTRAINT IF EXISTS model_configuration_id_check;
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema = current_schema() AND table_name = 'model_configuration'
                     AND column_name = 'is_active') THEN
        ALTER TABLE model_configuration ADD COLUMN is_active boolean NOT NULL DEFAULT false;
        UPDATE model_configuration SET is_active = true WHERE id = 1;
    END IF;
END $$;
CREATE UNIQUE INDEX IF NOT EXISTS mc_provider_unique ON model_configuration (provider);
CREATE UNIQUE INDEX IF NOT EXISTS mc_one_active ON model_configuration (is_active) WHERE is_active;
INSERT INTO model_configuration (id, provider, default_model, use_priority, is_active)
SELECT COALESCE(MAX(id), 0) + 1, 'openai', 'gpt-4.1-mini', 1, COUNT(*) = 0
FROM model_configuration
ON CONFLICT (provider) DO NOTHING;
INSERT INTO model_configuration (id, provider, default_model, summary_model, use_priority, is_active)
SELECT COALESCE(MAX(id), 0) + 1, 'gemini', 'gemini-3.5-flash-lite', 'gemini-2.5-flash', 2, false
FROM model_configuration
ON CONFLICT (provider) DO NOTHING;
COMMIT;
