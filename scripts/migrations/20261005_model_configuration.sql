-- Run before deploying the backend. Re-running never overwrites owner settings.
BEGIN;
SET LOCAL lock_timeout = '5s';
CREATE TABLE IF NOT EXISTS model_configuration (
    id integer PRIMARY KEY CHECK (id = 1),
    provider varchar(16) NOT NULL CHECK (provider IN ('gemini', 'openai', 'grok')),
    use_priority integer NOT NULL DEFAULT 1 CONSTRAINT mc_use_priority_positive CHECK (use_priority > 0),
    default_model varchar(200) NOT NULL CHECK (length(trim(default_model)) > 0),
    readiness_model varchar(200) CHECK (readiness_model IS NULL OR length(trim(readiness_model)) > 0),
    topic_selection_model varchar(200) CHECK (topic_selection_model IS NULL OR length(trim(topic_selection_model)) > 0),
    example_selection_model varchar(200) CHECK (example_selection_model IS NULL OR length(trim(example_selection_model)) > 0),
    grounding_model varchar(200) CHECK (grounding_model IS NULL OR length(trim(grounding_model)) > 0),
    response_model varchar(200) CHECK (response_model IS NULL OR length(trim(response_model)) > 0),
    response_check_model varchar(200) CHECK (response_check_model IS NULL OR length(trim(response_check_model)) > 0),
    response_split_model varchar(200) CHECK (response_split_model IS NULL OR length(trim(response_split_model)) > 0),
    summary_model varchar(200) CHECK (summary_model IS NULL OR length(trim(summary_model)) > 0),
    updated_at timestamptz NOT NULL DEFAULT now()
);
INSERT INTO model_configuration (id, provider, default_model)
VALUES (1, 'openai', 'gpt-4.1-mini')
ON CONFLICT (id) DO NOTHING;
COMMIT;
