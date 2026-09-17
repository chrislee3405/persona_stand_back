# Backend automated tests

This repository tests its API, business rules and database writes. The frontend
repository tests browser-side behavior independently. `persona_stand_ec2yml`
selects two versioned images and owns the combined Playwright tests and release.

## What runs

- `tests/unit/`: content validation, response splitting and database-reset safety.
- `tests/media/`: media-table migration and real media API checks in isolated
  schemas of the disposable PostgreSQL database, without chat/NLP dependencies.
- `tests/integration/`: real FastAPI requests, signed session cookies, PostgreSQL,
  the real local Presidio privacy detector, and the complete chat pipeline.
- `tests/fakes/fake_gemini.py`: predictable model answers and simulated failures.
  Only the external Gemini boundary is replaced. Tests reject attempts to
  construct a real Gemini client.
- `tests/support/seed.py`: fictional consent, invite, reference and portfolio data.
  It never reads the owner's local content exports or seed JSON.

API tests cover consent and withdrawal, policy changes, conversation ownership,
invite verification and revocation, session rotation, length and privacy gates,
durable daily quotas, pending-slot cleanup, generation deadlines, failed-message
history exclusion, ordered successful turns and content revision selection.

## Run locally (PowerShell)

Install application and test dependencies in the backend virtual environment:

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
.venv/Scripts/python.exe -m pytest tests/unit
```

The full suite requires a dedicated disposable database. With Docker Desktop
running, start one (port 55432 avoids the normal development PostgreSQL port):

```powershell
docker run --detach --rm --name persona-backend-tests-db -e POSTGRES_USER=persona_test -e POSTGRES_PASSWORD=persona_test -e POSTGRES_DB=persona_test -p 127.0.0.1:55432:5432 postgres:16-alpine
docker exec persona-backend-tests-db pg_isready -U persona_test -d persona_test
$env:TEST_DATABASE_URL = 'postgresql://persona_test:persona_test@127.0.0.1:55432/persona_test'
.venv/Scripts/python.exe -m pytest --junitxml=test-results/backend.xml
docker stop persona-backend-tests-db
Remove-Item Env:TEST_DATABASE_URL
```

Wait for `pg_isready` to report accepting connections before running pytest.
On Linux/macOS, use `python -m pytest` in an activated virtual environment and
`export TEST_DATABASE_URL=...` to set the same variable.

Each integration test empties and reseeds all application tables in `persona_test`.
Never put important data there. The guard permits that exact database name only
on localhost/loopback or the `db`/`postgres` test-container hostnames and rejects
URL query parameters. Bootstrap ignores your normal `DATABASE_URL` and `.env`.
Integration tests without `TEST_DATABASE_URL` fail with an explanation instead
of silently skipping. Do not run concurrent suites against one database.
Unit tests do not connect to PostgreSQL.

## GitHub Actions

Pull requests and pushes to every branch run pytest on a fresh runner with a
temporary PostgreSQL service. A JUnit report is saved even when tests fail.
ECR publishing runs only for `main`/`trial` pushes, after the tests pass. Images carry
their source commit as a tag and OCI revision label so `ec2yml` can verify a pair.
Publishing an image is not deployment approval. Existing AWS repository variables
are needed for publishing; tests need no AWS or Google secrets.

## Combined browser-test interface

Production images exclude `tests/`. The combined workflow checks out this repository
at the image's recorded source commit, then mounts `tests` read-only at
`/test-support/tests` in that image. It sets:

```text
PYTHONPATH=/test-support:/app
TEST_DATABASE_URL=postgresql://persona_test:persona_test@db:5432/persona_test
```

Its backend command is:

```text
uvicorn tests.support.e2e_app:app --host 0.0.0.0 --port 8000
```

That entrypoint runs the real app, resets and seeds the disposable database, and
installs fake Gemini. No fake-model flag or test endpoint exists in the production
application. The test entrypoint uses an HTTP session cookie for the isolated
runner. Production retains its normal `app.main:app` command and environment.

Stable fictional browser fixtures:

| Item | Value |
| --- | --- |
| Owner and chat name | `Test Candidate` |
| Project | `test-project`, label `Test Project` |
| Invite code | `TEST-INVITE` |
| Consent header | `Test consent` |
| Consent condition | `I agree to use fictional messages in this automated test.` |
| Model reply | `I built a fictional portfolio with automated tests.` |
| Message selecting a simulated failure | `Please simulate a model failure.` |

These checks verify application behavior. Real-model answer quality evaluations
remain a separate, explicitly triggered process.
