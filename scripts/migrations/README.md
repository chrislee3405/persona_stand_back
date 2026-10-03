# Per-turn token usage (v1.1.0)

Before deploying the new backend, open `20261002_message_token_usage.sql` in
pgAdmin's Query Tool connected to the application's RDS database and execute the
whole file. It adds nullable `message.token_usage` JSONB without changing existing
messages. Run `ROLLBACK;` first if a previous Query Tool operation left a failed
transaction. The migration uses a five-second lock timeout; if the table is busy,
retry the whole file during a quiet period. It is safe to rerun and preserves
already-recorded usage. An incompatible existing column causes a rollback.

Check the migration:

```sql
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_schema = current_schema() AND table_name = 'message'
  AND column_name = 'token_usage';
-- Expected: token_usage | jsonb | YES
```

Deploy the backend after this succeeds. Startup and `/api/health/ready` reject
an existing database without the correct column. Fresh databases get the column
through normal model schema creation. Rolling back the application does not
require dropping the column; older code ignores it. No reseeding is needed.

After a chat turn, inspect the triggering **user** row (including a row retagged
`not_saved_user` after failure). No usage is duplicated onto persona fragments,
system notices or regeneration review rows:

```sql
SELECT message_id, conversation_id, order_index, token_usage
FROM message WHERE token_usage IS NOT NULL
ORDER BY message_id DESC LIMIT 20;
```

The object has `schema_version: 1` and a `runs` array. Each run has a unique ID,
UTC start time, kind (`message`, `continue`, `summary`), generation status,
readiness decision, stage timings and per-request model/counts. Continuations
append to the last held user message; background summaries append to the last
included user message. Runs are recorded before reply publication, so a response
later superseded or failing publication still has its model usage recorded.
The run's status describes generation, not successful delivery to the browser.

Input, visible output, thinking, cached and provider-total tokens are stored
separately. Cached tokens are a subset of input: do not add them to input again.
Provider totals are authoritative; missing counts remain `null`, not zero. A
stage/turn aggregate is `null` when any request lacks that count; the per-request
records retain the available counts. Empty/skipped work consumes zero known
tokens. Application retries each have a request record; retries hidden inside
the SDK, provider usage not returned after network failure, process termination
and database write failures cannot be measured reliably by this mechanism.
This is operational usage telemetry, not an invoice reconciliation system.

Usage contains no prompt or reply text. It is excluded from prompts and API
responses and is not loaded with ordinary message queries. Writes are bounded,
best effort and separate from reply publication so logging errors do not discard
a valid reply. Concurrent runs append under a row lock and duplicate run IDs are
ignored. Historical messages stay `NULL` because their usage cannot be recovered.

# Asset table rename

`20260917_site_media.sql` renames `site_image` to `site_media` and `image_path`
to `media_path`, including the standard index, primary key and sequence names.
It preserves records and tag connections. It is safe to rerun after success;
ambiguous tables or columns cause a rollback instead of merging or dropping data.

For an existing database:

1. Back up the database and stop old backend instances (including workers).
2. Run with the intended database connection:
   `psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f scripts/migrations/20260917_site_media.sql`
   Use a PostgreSQL URI, not a SQLAlchemy `postgresql+asyncpg` URI.
3. Start the new backend, then deploy the frontend. Verify `/api/site-content`
   returns the expected `media` entries and test images, video/posters and PDFs.

Fresh databases use `SiteMedia` through normal schema creation and
`app/models/seed/site_media.json`. Existing site tables are not updated by the
seed loader. Never create a second empty table to replace the old one.

The API temporarily returns both `media` and the legacy `images` alias from
one query for already-open browsers. The new frontend prefers `media` and can
read `images` during deployment. Each entry still has `description` and `path`;
only the database/seed field is `media_path`. Remove the alias in a later
coordinated release. Source/poster/image tags are unchanged.

Application rollback requires stopping the new backend and reversing the table,
column, index, primary-key and sequence renames before starting old code.
The forward SQL has not been applied to a development or production database
by this change.

# Readiness gate: conversation.last_handled_index

`20260920_conversation_last_handled_index.sql` adds the pending-message cursor
the readiness gate reads and advances. `create_all` creates missing tables but
never adds a column to an existing one, so an existing database needs this run
once. Fresh databases get the column from the model.

For an existing database:

1. Stop the old backend.
2. Run with the intended database connection:
   `psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f scripts/migrations/20260920_conversation_last_handled_index.sql`
   Use a PostgreSQL URI, not a SQLAlchemy `postgresql+asyncpg` URI.
3. Start the new backend. The frontend change is backward compatible (an older
   bundle simply never sees a reply-less status), so the order is not critical.

Existing conversations are backfilled to their last message's `order_index`,
i.e. fully handled: the old backend answered or failed every message they
hold. Leaving them at -1 would make each one's first turn after deployment
treat every earlier user message as a single unanswered question, and a
rejected message on that turn would retag them all out of the prompt history.

Why the old backend should be stopped first: a turn it answers after the
backfill sits above the cursor, so the new backend reads that question once
more as pending.

Safe to rerun: the column is added and backfilled together, only on the run
that adds it. A second run changes nothing, so it can never move a cursor the
gate is already maintaining.

Rolling back the backend past this release needs no undo: the column is
ignored by older code.

## Readiness after migration

Startup refuses an existing conversation table without the required handled cursor.
The container health check uses `/api/health/ready`, which checks database access
and the current schema; `/docs` only proves that HTTP is serving. After migrating,
verify readiness and new guest/invite conversations before ending the maintenance
window. This check never seeds data or repairs old cursors automatically.
