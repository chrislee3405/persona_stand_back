# Seed data

Media assets now use `site_media.json`, model `SiteMedia`, and key field
`media_path` (images, videos, posters and PDFs). Existing databases must first
run [the table migration](../../../scripts/migrations/README.md); seeding does
not rename existing tables or update populated site tables.

The starting contents of an empty database, as JSON, plus the loader that puts
them there.

```bash
# from persona_stand_back/, with DATABASE_URL set
python -m app.models.seed.load             # create any missing tables, then load
python -m app.models.seed.load --dry-run   # validate every file, write nothing
```

Inside the local compose stack:

```bash
docker compose exec backend python -m app.models.seed.load
```

## Why this exists

Two reasons, one of which used to be a bug.

**A local database to develop against.** `docker-compose.yml` now runs its own
Postgres, so local development never touches production. Before this, the
documented local loop tunnelled to the production RDS instance — so every
local run wrote to production, and there was nowhere safe to run a test.

**Seeding is data, not application code.** `app/main.py` used to insert a
consent policy at import time, with a comment on the row calling its own
wording a placeholder. That placeholder became the live legal notice on the
first boot of any new environment: it is what every visitor agreed to and
what every `consent_record` row was stamped with, silently, unless somebody
remembered to replace it. A policy that has not been written yet should be
**absent** — and an absent policy correctly closes the chat gate rather than
quietly inventing terms.

## Most of these files are not in the repository

**This repository is public.** The seed files carry the site owner's real
content — CV, academic transcript, residency status, contact details and the
persona's profile — so `.gitignore` keeps every `*.json` here out of git
except one:

| Committed | Why it is safe |
|---|---|
| `consent_policy.json` | the consent notice, which is public text shown to every visitor |

Everything else lives only on your machine, `invite_code.json` included — a
working invite code has no business in a public repository, even a local one.
A fresh clone therefore seeds just the consent policy — the loader skips any
file that is absent — and renders the site's empty states; the chat has no
persona profile to answer from until the reference files are loaded. To get
real content onto a new machine, export it from a database that already has
it:

```bash
# writes one JSON dump per table into audit/db/ (also gitignored)
docker compose exec backend python scripts/export_content.py
```

and convert those dumps into this directory's shapes (map-shaped `site_content`
/ `site_journey` / `site_project`, `id` and `created_at` dropped). Never copy
them anywhere that is not gitignored.

### A local invite code

To test the invite tier locally, create `invite_code.json` yourself (it stays
gitignored) and re-run the loader:

```json
[{ "code": "LOCAL-DEV-CODE", "description": "local development only" }]
```

or insert one directly: `INSERT INTO code (code, description) VALUES ('LOCAL-DEV-CODE', 'local development only');`

⚠️ **A local code must never reach a deployed database.** Anything written in
this README is public, so a code like the one above seeded into RDS would be a
working invite code — invite tier, exempt from the per-IP limit — for anyone
who reads it. Real codes are issued per company; generate one with
`python -c "import secrets; print(secrets.token_urlsafe(12))"`.

### The consent policy

Write new wording as a **new** row with a new version — never edit an existing
one, or past `consent_record` rows stop meaning what they recorded.

`condition_text` is JSONB, `{"header"?, "condition"}` — a one-line purpose
statement shown above the box, and the detailed terms inside it. `condition`
is required and is the only part the client echoes back on submission;
`header` is optional and the dialog names itself by the terms when it is
absent. The `condition` goes through `<Prose>`, so blank lines become
paragraphs and `- ` lines become bullets, exactly like every other body field
on the site. A bare string is still accepted for rows written before the
column became JSONB — `ConsentService.normalise_terms` reads it as
`{"header": "", "condition": <the old text>}`.

Terms that cannot be read produce **no** consent gate rather than a broken
one: `GET /api/chatroom_initialize` reports them unavailable, the popup shows its inert
card with both choices dead, and every chat turn is refused with 403. That is
the correct fail-closed behaviour, but it is also silent — nothing crashes —
so `validate_consent_terms` checks the shape at seed time to keep a typo from
becoming a sitewide chat outage discoverable only from a log line.

## Validation

Every payload goes through `app/validators/content_validator.py` before it is
inserted, and `--dry-run` runs the same checks against the files alone.

This matters because the JSONB columns have no `CHECK` constraint,
`GET /api/site-content` has no response model, and the frontend casts the
payload with `as` rather than parsing it — so the database was the only thing
standing between a typo and a `TypeError` thrown mid-render in every visitor's
browser, and it checked nothing. A `"body": 42` where a string was meant put
React Router's default error boundary, stack trace and all, on screen for
everyone; `"links": "see contact"` on the footer would take down the whole
application, chatroom included, because the footer renders inside
`RootLayout`.

The same validator works on a hand-edited file, which is the quickest way to
check a content update before pasting it into `psql`:

```bash
python -m app.validators.content_validator my-new-content.json
python -m app.validators.content_validator --section journey journey-only.json
```

## What each file becomes

| File | Table | Re-run behaviour |
|---|---|---|
| `site_content.json` | `site_content` | skipped if the table has any rows |
| `site_journey.json` | `site_journey` | skipped if the table has any rows |
| `site_project.json` | `site_project` | skipped if the table has any rows |
| `site_media.json` | `site_media` | skipped if the table has any rows |
| `consent_policy.json` | `consent_policy` | each version inserted only if that version is absent |
| `invite_code.json` | `code` | each code inserted only if that code is absent |
| `doc_reference.json` | `doc_reference` | each topic inserted only if that topic is absent |
| `scenario_reference.json` | `scenario_reference` | each topic inserted only if that topic is absent |
| `personality_reference.json` | `personality_reference` | skipped if the table has any rows |
| `question_bank.json` | `question_bank` | skipped if the table has any rows |

The `site_*` tables are append-only by design — "to change a section, INSERT a
new row with the same slug; reads take the highest id" — so re-running the loader
against a database that already has content would stack a second, identical
version on top of every section. Hence "skipped if the table has any rows"
rather than an upsert. To reload them, delete the rows first.

`question_bank` is all-or-nothing for a different reason: there is no
uniqueness on a question, so a second run would duplicate every row and skew
every BM25 document frequency.

## A note on `question_bank` size

Keep at least **three** rows. BM25's IDF floors any term that appears in most
of the corpus, and with one or two documents *every* term qualifies. Retrieval
still works on a corpus that small — with no positive IDF to scale from,
`_compute_idf` falls back to a flat weight and logs a warning, so ranking
reduces to plain word overlap — but real IDF weighting, where a rare word
counts for more than a common one, needs three or more documents.

## After changing `question_bank`

The BM25 corpus is precomputed and cached in the `corpus_cache` table. Delete
that row to force a rebuild:

```sql
DELETE FROM corpus_cache;
```

The next chat turn recomputes it and stores the new one. (This genuinely works
now — `BM25Service._get_corpus` checks the row's id on every turn. It used to
consult a process-wide cache first and never re-check, which made this `DELETE`
a no-op until the container restarted.)
