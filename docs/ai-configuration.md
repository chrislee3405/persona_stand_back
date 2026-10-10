# AI provider and stage model configuration

`app/models/model_configuration.py` stores one row per provider. `is_active` means
enabled for selection; multiple providers can be enabled. At least one must be
enabled. Enabled providers must have unique positive `use_priority` values;
smaller numbers are checked first.

For each stage, choose the first enabled provider with a non-NULL stage model.
Only when every enabled provider has NULL for that stage, use the highest-priority
enabled provider's `default_model`. NULL therefore means continue searching, not
immediately use that row's default. Disabled providers are ignored entirely.
This is configuration selection, not automatic failover after API errors.

The seed enables both providers on a fresh database. Gemini has priority 1 and
uses `gemini-3.5-flash-lite` for readiness, topics, grounding, writing, rewrites,
review and splitting, plus `gemini-2.5-flash` for summaries. Its example model is
NULL so OpenAI, priority 2, supplies `gpt-4.1-mini` only for example selection,
which returns a fixed-choice example ID rather than writing a reply.
Local Compose seeds before backend startup. Re-running seeding adds missing
providers as inactive and never overwrites existing settings or activation.

Existing databases must run these migrations in order before this backend starts:

1. `scripts/migrations/20261005_model_configuration.sql`
2. `scripts/migrations/20261006_model_configuration_use_priority.sql`
3. `scripts/migrations/20261007_model_configuration_providers.sql`
4. `scripts/migrations/20261008_model_configuration_selection.sql`

The provider migration inserts missing presets; the selection migration removes
the single-active-row restriction and preserves all settings. No private seed data or credentials are
required for this migration. Fresh databases created by the current ORM do not
need these upgrade migrations.

To switch to Gemini in pgAdmin (use `openai` instead to switch back):

```sql
BEGIN;
LOCK TABLE model_configuration IN SHARE ROW EXCLUSIVE MODE;
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM model_configuration WHERE provider = 'gemini') THEN
        RAISE EXCEPTION 'Gemini configuration missing; run the provider migration';
    END IF;
END $$;
UPDATE model_configuration SET is_active = false, updated_at = now() WHERE is_active;
UPDATE model_configuration SET is_active = true, updated_at = now() WHERE provider = 'gemini';
COMMIT;
```

The SQL above selects Gemini exclusively. To enable multiple providers, assign
distinct priorities and set each desired row's is_active to true. Disable a row
with is_active=false to retain its settings while excluding it from selection.
Settings changes affect new turns and summary runs without a restart; runs
already in progress retain their configuration snapshot. Provider credentials must
already be available to the backend. Switching provider sends subsequent chat
context to that provider.

The required highest-priority `default_model` applies only when all enabled
providers lack an explicit model for the stage. Overrides are
`readiness_model`, `topic_selection_model`, `example_selection_model`,
`grounding_model`, `response_model`, `response_check_model`, `response_split_model`
and `summary_model`. Regeneration uses `response_model`. There is no model or
provider failover, and grounding has no correction retry.

Grounding claims contain `source_quote`, `statement`, `status` and `answer_part`.
The quote preserves original wording; statement proposes a direct restatement or
reasonable deduction. Python checks structure and fixed-choice values without
comparing text. The writer preserves the supported interpretation and the AI gate
checks quote-to-statement-to-reply meaning, without receiving original documents.
Existing stored traces remain readable; no database migration is required.
Reference seed edits do not overwrite existing database content.

## Credentials and activation

Gemini retains the existing Vertex/GCP authentication and `GCP_PROJECT_ID`.
OpenAI uses `OPENAI_API_KEY`; Grok uses `XAI_API_KEY`. Set these in the backend
environment/secret store, never the database or committed configuration file.
Inactive providers do not require credentials. The fresh seed selects OpenAI
for examples, so leaving `OPENAI_API_KEY` blank does not implement a Gemini-only
setup. Use the exclusive-provider SQL above if that is the intended routing.

| Environment | Gemini | OpenAI | Grok |
| --- | --- | --- | --- |
| Local Compose | `GCP_PROJECT_ID`, host `GOOGLE_ADC_PATH` file mounted at `/app/adc.json`, and `GOOGLE_APPLICATION_CREDENTIALS=/app/adc.json` | `OPENAI_API_KEY` in backend `.env` | `XAI_API_KEY` in backend `.env` |
| Production Compose | `GCP_PROJECT_ID` and the existing mounted WIF configuration | `OPENAI_API_KEY` is explicitly passed from the EC2 environment | No mapping yet: add a reviewed `XAI_API_KEY` backend environment mapping before enabling Grok |

Local Compose reads the whole backend `.env`; production passes only its listed
environment variables. Adding a production key to `.env` without a Compose
mapping does not expose it to the backend. Recreate the backend to apply changed
environment values; `docker compose restart` alone retains the old environment.
Neither configuration queries nor `/api/health/ready` make provider calls, so
they cannot establish that credentials or remote model access work.

Use a JSON file with the same settings, excluding `id` and `updated_at`:

```json
{
  "provider": "openai",
  "default_model": "YOUR_SUPPORTED_MODEL",
  "response_model": "YOUR_SUPPORTED_REPLY_MODEL"
}
```

From the backend root:

```sh
python -m scripts.configure_ai settings.json --check-only
python -m scripts.configure_ai settings.json
```

For the local Docker stack, put `settings.json` in the backend repository and
prefix those commands with `docker compose exec backend`. The repository is
bind-mounted there, so both the helper and JSON are available, and the database
host `db` resolves correctly. A host Python run needs its own exported environment
and a host-reachable database URL (local PostgreSQL is `localhost:5434`). The
normal production image does not contain `scripts/configure_ai.py`; use a
reviewed operator checkout or the transactional database instructions rather
than assuming the helper is installed in EC2's backend container.

Both commands make live text and structured-output probes against each distinct
configured model and incur provider usage. The second updates and activates the selected provider
row in one transaction only after all probes pass. Omitted stage overrides become
NULL, so old provider model names cannot accidentally remain through this command.
Other providers remain enabled; the helper preserves the target's priority when
available, otherwise assigns a priority after currently enabled providers. Edit
priorities in the database to change preference. Direct SQL updates are possible but
bypass capability/access probes; update provider and all model fields together and
set `updated_at = now()`. Database constraints cannot verify remote model access.
No owner UI or public configuration endpoint is added.

Probes check basic access and output support, not grounding quality or every schema
constraint. Evaluate representative conversations when selecting new models. The
adapters currently target text and structured Chat Completions-compatible models;
models requiring other endpoints or unsupported schema features are not supported.

## Connection lifetime and observability

Context gathering ends its initial read transaction before concurrent AI calls and
ends the selected-reference read transaction before grounding/generation. These
boundaries refuse unexpected pending ORM writes. The application's existing
`expire_on_commit=False` keeps already-loaded history attributes usable without
new reads; the configuration is copied into a frozen dataclass. The request session
can borrow another connection for later writes. Coordination locks keep their
separate pool and retain the existing turn serialization.

Background summaries retain their separate read/write sessions. This change does
not remove the coordination connection or promise that unrelated write paths never
borrow connections. AI usage records include provider and concrete model names.
Each request in `message.token_usage.runs[].stages.<stage>.requests[]` also stores
`response`: the model's returned text, including raw JSON for structured calls.
Capture occurs before parsing and validation, so invalid JSON and partial output
remain available. No returned text means null; OpenAI-compatible refusal text is
stored separately as `refusal` when present. Retries have separate request entries;
background summaries use their existing separate run. These fields remain out of
chat API responses, conversation prompts and ordinary metrics logs. No schema
migration is needed for these additive JSONB fields; old entries are unchanged.

Adapters keep bounded async calls and output limits. OpenAI/Grok schemas translate
uppercase Gemini types and wrap scalar/array roots in an object, then unwrap the
result for existing stage validation. No transport retries are added.

API references: [OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
and [Grok structured outputs](https://docs.x.ai/developers/model-capabilities/text/structured-outputs).

Model-output validation checks readable structure, required text fields and fixed-choice
values. Free-text answer-part labels need not repeat request descriptions. Python
does not compare grounding quotes, response-review quotations or split-message wording
with original text; the model prompts carry those content-preservation requirements.
