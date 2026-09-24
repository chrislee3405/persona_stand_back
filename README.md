# persona_stand_back

Automated tests and CI instructions: [TESTING.md](TESTING.md).

Every branch push runs independent backend tests, then publishes a commit-specific GHCR image if they pass. Pull requests test without publishing. `persona_stand_ec2yml` selects frontend/backend digests and runs the combined browser suite. Minor updates stay in GHCR; approved major releases copy the tested images unchanged into ECR for EC2 deployment. No AWS publishing role is needed in this repository.

FastAPI + SQLAlchemy (async) + PostgreSQL. The chat pipeline and the
site-content API.

Full setup tutorial: https://github.com/chrislee3405/persona_stand_ec2yml

Current version 1.0.0

---

## Local development

The stack is self-contained — it runs its own Postgres. Nothing tunnels to
production any more.

### 1. Create `.env`

Copy the block below into `persona_stand_back/.env`.

```env
# --- Database ------------------------------------------------------------
# The host is the compose SERVICE NAME, not localhost: the backend reaches
# Postgres over the compose network. pgAdmin on your host uses localhost.
DATABASE_URL=postgresql://persona:persona_dev_password@db:5432/persona

# --- Session signing -----------------------------------------------------
# Signs the httpOnly session cookie, which carries consent and invite-code
# verification. Generate a real one:
#   python -c "import secrets; print(secrets.token_urlsafe(48))"
# Changing it logs every visitor out.
SESSION_SECRET_KEY=change-me-generate-a-real-one

# --- Google Cloud / Vertex AI --------------------------------------------
GCP_PROJECT_ID=<your-gcp-project-id>

# Where your own application-default credentials live. Create them with
#   gcloud auth application-default login
# Linux/macOS: ~/.config/gcloud/application_default_credentials.json
# Windows:     C:/Users/<you>/AppData/Roaming/gcloud/application_default_credentials.json
# Deployment uses Workload Identity Federation instead and needs no key file.
GOOGLE_ADC_PATH=~/.config/gcloud/application_default_credentials.json
# docker-compose.yml mounts that file read-only at /app/adc.json. This line is
# what tells the Google client to read it there -- without it the mount does
# nothing and every chat turn fails to authenticate.
GOOGLE_APPLICATION_CREDENTIALS=/app/adc.json

# --- Logging and cookie security -----------------------------------------
# Leave these unset locally unless you need them. See "LOG_LEVEL,
# SESSION_COOKIE_SECURE and CHAT_TRACE" below before setting any of them
# anywhere that is deployed.
# LOG_LEVEL=INFO
# SESSION_COOKIE_SECURE=true
# CHAT_TRACE=true    # local debugging only: logs full prompts and replies
```

`DATABASE_URL` is written in the plain `postgresql://` form and rewritten to
`postgresql+asyncpg://` at startup (`app/database.py`), so the same string
works unchanged in psql and pgAdmin. An `?sslmode=require` is understood and
translated for asyncpg.

**`LOG_LEVEL`, `SESSION_COOKIE_SECURE` and `CHAT_TRACE`** (`app/runtime_settings.py`):

- `LOG_LEVEL` (`DEBUG` / `INFO` / `WARNING` / `ERROR`). Anything serving real
  visitors runs at `INFO` or above; the deploy script refuses `DEBUG`.
- `CHAT_TRACE` (`true` / `false`, default `false` everywhere). The only way
  conversation content reaches the logs: full prompts, model responses, and
  what a model wrote about a message (the readiness reason, grounding notes,
  response-gate quotes). All of it goes through one logger,
  `app.chat_trace` (`app/chat_trace.py`), which is silent without this switch,
  **even at `DEBUG`**. Turn it on locally to watch the pipeline work. The
  deploy script refuses it, and the combined browser tests fail if a chat
  message's text, or any `app.chat_trace` record, appears in the logs.
- `SESSION_COOKIE_SECURE` (`true` / `false`). A `Secure` cookie is never sent
  over plain `http://`, so turning this on before the site is HTTPS end to end
  locks every visitor out of chat — their session is empty on every request,
  consent never sticks, and every turn returns 403.
- Unset, both follow the older `ENV` variable: `ENV=production` means `INFO`
  and `Secure`, anything else means `DEBUG` and not `Secure`. An unrecognised
  value for either stops the backend at startup instead of guessing.

### 2. Bring the stack up

```bash
docker compose up --build
```

Postgres comes up first (the backend waits on its healthcheck), then the
backend creates any missing tables on startup.

### 3. Seed the database

```bash
docker compose exec backend python -m app.models.seed.load
```

The real content seed files are **gitignored** — this repository is public and
they hold the owner's CV, transcript and contact details. A fresh clone seeds
only the consent policy; see
[`app/models/seed/README.md`](app/models/seed/README.md) for how to bring real
content onto a new machine, and how to add a local invite code (which must
never reach a deployed database). To check the seed files without writing
anything:

```bash
docker compose exec backend python -m app.models.seed.load --dry-run
```

The site is then at http://localhost and the API at http://localhost:8000.

---

## Connecting pgAdmin to the local database

The `db` service publishes **host port 5434** (container 5432), so pgAdmin
connects to your host, not to the container network.

Why not 5432: Windows lets two processes listen on the same port, so a
natively installed `postgres.exe` and Docker's port proxy can both bind 5432 —
and an incoming connection goes to whichever wins. The symptom is misleading:
`FATAL: password authentication failed for user "persona"`, because the
connection reached the *other* server, which has no such role. Check with
`netstat -ano | findstr :5432` — two LISTENING lines means you have this.
(5433 is avoided too: that's the SSH tunnel to production RDS.)

1. Make sure the stack is running: `docker compose ps` should show
   `persona_db` as `healthy`.
2. In pgAdmin: right-click **Servers** → **Register** → **Server…**
3. **General** tab → **Name**: `persona local` (any label — this is just
   pgAdmin's own name for the connection).
4. **Connection** tab:

   | Field | Value |
   |---|---|
   | Host name/address | `localhost` |
   | Port | `5434` |
   | Maintenance database | `persona` |
   | Username | `persona` |
   | Password | `persona_dev_password` |
   | Save password | tick it |

5. **Save**. The tables are under
   `persona` → **Schemas** → `public` → **Tables**.

Notes:

- **Host is `localhost`, not `db`.** `db` is the compose service name and only
  resolves *inside* the compose network. pgAdmin runs on your machine.
- **If pgAdmin itself runs in Docker**, `localhost` is that container, not your
  host — use `host.docker.internal` (Windows/macOS) as the host, or put pgAdmin
  on the `persona_network` and use `db` as the host.
- **The port is 5434, not 5432** — see the note above. If 5434 is also taken
  on your machine, change the left-hand side of the mapping in
  `docker-compose.yml` and point pgAdmin at whatever you pick. `DATABASE_URL`
  does not change: the backend goes through the compose network as `db:5432`
  and never touches the published port.
- **`docker compose down` keeps your data** (it lives in the `persona_pgdata`
  named volume). `docker compose down -v` deletes it; re-run the seed loader
  afterwards.

Useful queries once connected:

```sql
-- what the site is currently serving (highest id per section)
SELECT DISTINCT ON (section) section, id, created_at
FROM site_content ORDER BY section, id DESC;

-- today's rate-limit counters, busiest first
SELECT key, count FROM rate_limit_counter
WHERE day = CURRENT_DATE ORDER BY count DESC;

-- failed turns kept for review (no tracebacks stored -- grep the log for the incident id)
SELECT created_at, text FROM message WHERE sender = 'error' ORDER BY created_at DESC LIMIT 20;

-- force a BM25 rebuild after editing question_bank
DELETE FROM corpus_cache;
```

---

## Deleting a conversation on request

There is no self-service deletion. Under the chat window each visitor sees a
**conversation reference** — the conversation's id, which is random and says
nothing about who they are — and the consent notice tells them to contact the
site owner to have something removed. When a request quotes a reference:

```sql
-- 1. confirm it exists, and see what will go
SELECT conversation_id, created_at, code,
       (SELECT count(*) FROM message m WHERE m.conversation_id = c.conversation_id) AS messages
FROM conversation c
WHERE conversation_id = '<reference>';

-- 2. delete it -- messages first, they reference the conversation
BEGIN;
DELETE FROM message      WHERE conversation_id = '<reference>';
DELETE FROM conversation WHERE conversation_id = '<reference>';
COMMIT;
```

Paste the reference exactly — it is a full UUID, and matching on a prefix
could delete someone else's conversation.

What this does **not** remove, deliberately:

- **`consent_record`** — proof that consent was given (and, if they used
  "Disagree with consent", when it was withdrawn). It is keyed to a random
  session id, not to the conversation, and holds no message content.
- **`rate_limit_counter`** — daily counts keyed by session id or IP, with no
  content; swept automatically after `RATE_LIMIT_RETENTION_DAYS`.

Withdrawing consent in the chatroom stops further collection but does not
delete anything already stored; deletion is always this manual step.

---

## Validating content before it reaches the database

The JSONB content columns have no `CHECK` constraint and the frontend casts the
payload rather than parsing it, so a wrong type in one row throws mid-render in
every visitor's browser. Run any hand-written content through the validator
first:

```bash
python -m app.validators.content_validator my-content.json
python -m app.validators.content_validator --section journey journey-only.json
```

The seed loader runs the same checks on every payload it inserts.
