"""
Loads the JSON files in this directory into an empty database.

    python -m app.models.seed.load            # load anything missing
    python -m app.models.seed.load --dry-run  # validate only, touch nothing

Written for the local `db` service in persona_stand_back/docker-compose.yml,
so a fresh clone comes up with a site that renders and a chat that answers,
without a tunnel to production RDS. It is safe to point at any database: it
never deletes and never updates, and it skips anything already present.

TWO INSERT RULES, matching the two kinds of table:

- The site_* tables are append-only by design -- "to change a section, INSERT
  a new row with the same slug; reads take the newest" (see
  app/models/site_content.py). Re-running the loader against a database that
  already has content would therefore stack a second, identical version on top
  of every section. So these are seeded ONLY when the table is empty.

- consent_policy, code, doc_reference, scenario_reference,
  personality_reference and question_bank have real identity (a version, a
  code, a topic). Each row is skipped individually if its key already exists,
  so the loader can add a newly-written reference topic to a database that
  already has the others.

EVERY payload goes through app/validators/content_validator.py first. That is
the point of routing seeding through code rather than pasting SQL: the JSONB
columns have no CHECK constraint and the frontend casts the payload with `as`
rather than parsing it, so a `"body": 42` reaches the browser and throws
mid-render. --dry-run runs exactly the same validation and writes nothing,
which is also the quickest way to check a hand-edited content file before
pasting it into psql.
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import logging
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import SessionLocal, engine, Base
from app.models.code import InviteCode
from app.models.consent import ConsentPolicy
from app.models.prompt_reference import (
    DocReference,
    PersonalityReference,
    QuestionBank,
    ScenarioReference,
)
from app.models.rate_limit import RateLimitCounter  # noqa: F401  -- registers table for create_all
from app.models.site_content import SiteContent
from app.models.site_image import SiteImage
from app.models.site_journey import SiteJourney
from app.models.site_project import SiteProject
from app.validators.content_validator import (
    ContentValidationError,
    validate_consent_terms,
    validate_image,
    validate_journey_detail,
    validate_project_detail,
    validate_site_content,
)

logger = logging.getLogger("seed")

SEED_DIR = Path(__file__).parent


def _read(name: str) -> Any:
    """
    Reads one seed JSON file from this directory.

    Parameters:
    - name (str): the file name, e.g. "site_content.json" -- comes from the loader below

    Returns:
    - Any: the parsed JSON -- goes to the matching loader function. Returns None
      for a file that is absent, so an optional seed file can simply be deleted.
    """
    path = SEED_DIR / name
    if not path.exists():
        logger.info("skip %s (file not present)", name)
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


async def _is_empty(db: AsyncSession, model) -> bool:
    """
    Reports whether a table has no rows.

    Parameters:
    - db (AsyncSession): the loader's session
    - model: the SQLAlchemy model whose table to check

    Returns:
    - bool: True when the table is empty -- used to decide whether to seed an append-only table
    """
    result = await db.execute(select(func.count()).select_from(model))
    return (result.scalar_one() or 0) == 0


def _check_columns(model, row: dict, source: str, index: int) -> None:
    """
    Checks that a seed row's keys are all real columns on the model it will be constructed from.

    Parameters:
    - model: the SQLAlchemy model the row becomes
    - row (dict): one entry from a seed file
    - source (str): the file name, for the message
    - index (int): the row's position in that file, zero-based

    Returns:
    - None: raises ContentValidationError naming every unknown key, with a
      spelling suggestion where one is close

    Without this, a stale key surfaces as a bare
    `TypeError: 'cluture_background' is an invalid keyword argument for
    PersonalityReference` from deep inside SQLAlchemy's constructor -- which
    says what is wrong but not which FILE or which ROW, and offers nothing
    when the cause is a rename that the seed files were not carried through.
    """
    columns = set(model.__table__.columns.keys())
    unknown = sorted(set(row) - columns)
    if not unknown:
        return
    problems = []
    for key in unknown:
        near = difflib.get_close_matches(key, columns, n=1, cutoff=0.7)
        hint = f" -- did you mean {near[0]!r}?" if near else ""
        problems.append(f"unknown column {key!r}{hint}")
    raise ContentValidationError(f"{source} row {index + 1} ({model.__tablename__})", problems)


# --- Per-table loaders ----------------------------------------------------

async def _load_site_content(db: AsyncSession, dry_run: bool) -> int:
    """
    Validates and inserts the `site_content` sections.

    Parameters:
    - db (AsyncSession): the loader's session
    - dry_run (bool): validate only, insert nothing

    Returns:
    - int: how many rows were inserted (0 on a dry run or when the table already has content)
    """
    payload = _read("site_content.json")
    if payload is None:
        return 0
    if not isinstance(payload, dict):
        raise ContentValidationError("site_content.json", ["top level must be an object of {section: content}"])

    for section, content in payload.items():
        validate_site_content(section, content)
    logger.info("validated %d site_content section(s)", len(payload))

    if dry_run:
        return 0
    if not await _is_empty(db, SiteContent):
        logger.info("skip site_content (table already has rows -- it is append-only, see this module's docstring)")
        return 0

    for section, content in payload.items():
        db.add(SiteContent(section=section, content=content))
    return len(payload)


async def _load_site_journey(db: AsyncSession, dry_run: bool) -> int:
    """
    Validates and inserts the `site_journey` detail sheets.

    Parameters:
    - db (AsyncSession): the loader's session
    - dry_run (bool): validate only, insert nothing

    Returns:
    - int: how many rows were inserted
    """
    payload = _read("site_journey.json")
    if payload is None:
        return 0
    for journey_id, content in payload.items():
        validate_journey_detail(journey_id, content)
    logger.info("validated %d site_journey detail sheet(s)", len(payload))

    if dry_run or not await _is_empty(db, SiteJourney):
        if not dry_run:
            logger.info("skip site_journey (table already has rows)")
        return 0

    for journey_id, content in payload.items():
        db.add(SiteJourney(journey_id=journey_id, content=content))
    return len(payload)


async def _load_site_project(db: AsyncSession, dry_run: bool) -> int:
    """
    Validates and inserts the `site_project` detail sheets.

    Parameters:
    - db (AsyncSession): the loader's session
    - dry_run (bool): validate only, insert nothing

    Returns:
    - int: how many rows were inserted
    """
    payload = _read("site_project.json")
    if payload is None:
        return 0
    for project_id, content in payload.items():
        validate_project_detail(project_id, content)
    logger.info("validated %d site_project detail sheet(s)", len(payload))

    if dry_run or not await _is_empty(db, SiteProject):
        if not dry_run:
            logger.info("skip site_project (table already has rows)")
        return 0

    for project_id, content in payload.items():
        db.add(SiteProject(project_id=project_id, content=content))
    return len(payload)


async def _load_site_image(db: AsyncSession, dry_run: bool) -> int:
    """
    Validates and inserts the `site_image` rows.

    Parameters:
    - db (AsyncSession): the loader's session
    - dry_run (bool): validate only, insert nothing

    Returns:
    - int: how many rows were inserted
    """
    payload = _read("site_image.json")
    if payload is None:
        return 0
    for index, row in enumerate(payload):
        _check_columns(SiteImage, row, "site_image.json", index)
        validate_image(row.get("section"), row.get("description"), row.get("image_path"))
    logger.info("validated %d site_image row(s)", len(payload))

    if dry_run or not await _is_empty(db, SiteImage):
        if not dry_run:
            logger.info("skip site_image (table already has rows)")
        return 0

    for row in payload:
        db.add(SiteImage(**row))
    return len(payload)


async def _load_consent_policy(db: AsyncSession, dry_run: bool) -> int:
    """
    Inserts any consent policy version that is not already stored.

    Parameters:
    - db (AsyncSession): the loader's session
    - dry_run (bool): validate only, insert nothing

    Returns:
    - int: how many versions were inserted

    THIS IS WHY THE SEED MOVED OUT OF app/main.py. It used to run at import
    time, inserting a row whose text says in its own comment that it is a
    placeholder -- and that placeholder then became the live legal notice
    every visitor agreed to and every consent_record row was stamped with,
    silently, on the first boot of any new environment. Seeding is data: a
    policy that has not been written yet should be absent, and an absent
    policy correctly closes the chat gate rather than quietly inventing terms.
    """
    payload = _read("consent_policy.json")
    if payload is None:
        return 0

    inserted = 0
    for index, row in enumerate(payload):
        _check_columns(ConsentPolicy, row, "consent_policy.json", index)
        version = row["version"]
        # Validated like every other JSONB payload, and for a sharper reason:
        # terms ConsentService.normalise_terms cannot read make the consent
        # gate UNAVAILABLE, which refuses every chat turn sitewide without
        # anything crashing. See validate_consent_terms.
        validate_consent_terms(version, row.get("condition_text"))
        if dry_run:
            continue
        existing = await db.execute(
            select(ConsentPolicy.id).where(ConsentPolicy.version == version).limit(1)
        )
        if existing.scalar_one_or_none() is not None:
            logger.info("skip consent_policy %s (already present)", version)
            continue
        db.add(ConsentPolicy(**row))
        inserted += 1
    return inserted


async def _load_keyed(db: AsyncSession, dry_run: bool, *, file_name: str, model, key_field: str) -> int:
    """
    Inserts rows from a seed file, skipping any whose key already exists.

    Parameters:
    - db (AsyncSession): the loader's session
    - dry_run (bool): validate only, insert nothing
    - file_name (str): the seed file to read
    - model: the SQLAlchemy model to insert
    - key_field (str): the column that identifies a row (its unique key)

    Returns:
    - int: how many rows were inserted

    Used for the tables whose rows have real identity -- invite codes,
    reference topics -- so a newly written topic can be added to a database
    that already holds the others, without duplicating them.
    """
    payload = _read(file_name)
    if payload is None or dry_run:
        return 0

    column = getattr(model, key_field)
    inserted = 0
    for index, row in enumerate(payload):
        _check_columns(model, row, file_name, index)
        existing = await db.execute(select(model.id).where(column == row[key_field]).limit(1))
        if existing.scalar_one_or_none() is not None:
            continue
        db.add(model(**row))
        inserted += 1
    if inserted:
        logger.info("inserted %d row(s) into %s", inserted, model.__tablename__)
    return inserted


async def _load_singleton(db: AsyncSession, dry_run: bool, *, file_name: str, model) -> int:
    """
    Inserts rows only if the table is completely empty.

    Parameters:
    - db (AsyncSession): the loader's session
    - dry_run (bool): validate only, insert nothing
    - file_name (str): the seed file to read
    - model: the SQLAlchemy model to insert

    Returns:
    - int: how many rows were inserted

    For personality_reference (ContextGatherer reads the first row, so a
    second one would be silently ignored) and question_bank (no uniqueness on
    a question, so re-running would duplicate every row and skew every BM25
    document frequency).
    """
    payload = _read(file_name)
    if payload is None or dry_run:
        return 0
    for index, row in enumerate(payload):
        _check_columns(model, row, file_name, index)
    if not await _is_empty(db, model):
        logger.info("skip %s (table already has rows)", model.__tablename__)
        return 0
    for row in payload:
        db.add(model(**row))
    logger.info("inserted %d row(s) into %s", len(payload), model.__tablename__)
    return len(payload)


# --- Entry point ----------------------------------------------------------

async def seed(dry_run: bool = False, create_tables: bool = True) -> None:
    """
    Runs every loader in one transaction.

    Parameters:
    - dry_run (bool): validate every payload and write nothing -- comes from the CLI
    - create_tables (bool): run create_all first, so a brand-new database works -- comes from the CLI

    Returns:
    - None: commits once at the end, so a validation failure anywhere leaves the database untouched
    """
    if create_tables and not dry_run:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        logger.info("schema ready")

    async with SessionLocal() as db:
        total = 0
        total += await _load_site_content(db, dry_run)
        total += await _load_site_journey(db, dry_run)
        total += await _load_site_project(db, dry_run)
        total += await _load_site_image(db, dry_run)
        total += await _load_consent_policy(db, dry_run)
        total += await _load_keyed(db, dry_run, file_name="invite_code.json", model=InviteCode, key_field="code")
        total += await _load_keyed(db, dry_run, file_name="doc_reference.json", model=DocReference, key_field="document_topic")
        total += await _load_keyed(db, dry_run, file_name="scenario_reference.json", model=ScenarioReference, key_field="scenario_topic")
        total += await _load_singleton(db, dry_run, file_name="personality_reference.json", model=PersonalityReference)
        total += await _load_singleton(db, dry_run, file_name="question_bank.json", model=QuestionBank)

        if dry_run:
            logger.info("dry run: every payload validated, nothing written")
            return

        await db.commit()
        logger.info("seed complete: %d row(s) inserted", total)

    await engine.dispose()


def main() -> int:
    """
    Command-line entry point.

    Parameters:
    - none (reads sys.argv)

    Returns:
    - int: process exit code -- 0 on success, 1 if any payload failed validation
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="validate every seed file and write nothing")
    parser.add_argument("--no-create-tables", action="store_true", help="assume the schema already exists")
    args = parser.parse_args()

    try:
        asyncio.run(seed(dry_run=args.dry_run, create_tables=not args.no_create_tables))
    except ContentValidationError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
