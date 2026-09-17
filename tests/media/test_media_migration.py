"""Exercise the real SQL in an isolated schema of the disposable test DB."""
import os
from pathlib import Path
from uuid import uuid4

import asyncpg
import pytest

from tests.support.environment import validate_test_database_url

pytestmark = pytest.mark.integration
SQL = (Path(__file__).resolve().parents[2] / 'scripts/migrations/20260917_site_media.sql').read_text()


async def test_rename_preserves_history_defaults_and_is_repeatable():
    url = validate_test_database_url(os.environ['TEST_DATABASE_URL']).replace('postgresql+asyncpg:', 'postgresql:')
    connection = await asyncpg.connect(url)
    schema = 'migration_' + uuid4().hex
    try:
        await connection.execute(f'CREATE SCHEMA {schema}; SET search_path TO {schema}')
        await connection.execute('''
            CREATE TABLE site_image (
                id serial PRIMARY KEY, section varchar NOT NULL,
                description varchar NOT NULL, image_path varchar NOT NULL,
                created_at timestamp NOT NULL DEFAULT now()
            );
            CREATE INDEX ix_site_image_section_description_id_desc
              ON site_image (section, description, id DESC);
            INSERT INTO site_image (section, description, image_path) VALUES
              ('projects', 'demo', 'test/old.mp4'),
              ('projects', 'demo', 'test/new.mp4'),
              ('projects', 'document', 'test/details.pdf');
        ''')
        before = await connection.fetch('SELECT id, section, description, image_path, created_at FROM site_image ORDER BY id')
        await connection.execute(SQL)
        await connection.execute(SQL)
        after = await connection.fetch('SELECT id, section, description, media_path, created_at FROM site_media ORDER BY id')
        assert [tuple(row) for row in before] == [tuple(row) for row in after]
        assert await connection.fetchval("SELECT to_regclass('site_image')") is None
        assert await connection.fetchval("SELECT to_regclass('ix_site_media_section_description_id_desc')")
        assert await connection.fetchval("INSERT INTO site_media (section, description, media_path) VALUES ('projects', 'photo', 'test/photo.jpg') RETURNING id") == 4
    finally:
        await connection.execute('ROLLBACK')
        await connection.execute(f'SET search_path TO public; DROP SCHEMA {schema} CASCADE')
        await connection.close()


async def test_ambiguous_tables_abort_without_losing_records():
    url = validate_test_database_url(os.environ['TEST_DATABASE_URL']).replace('postgresql+asyncpg:', 'postgresql:')
    connection = await asyncpg.connect(url)
    schema = 'migration_' + uuid4().hex
    try:
        await connection.execute(f'CREATE SCHEMA {schema}; SET search_path TO {schema}')
        await connection.execute('CREATE TABLE site_image (id integer); INSERT INTO site_image VALUES (7); CREATE TABLE site_media (id integer)')
        with pytest.raises(asyncpg.RaiseError, match='Both site_image and site_media'):
            await connection.execute(SQL)
        await connection.execute('ROLLBACK')
        assert await connection.fetchval('SELECT id FROM site_image') == 7
        assert await connection.fetchval("SELECT to_regclass('site_media')")
    finally:
        await connection.execute('ROLLBACK')
        await connection.execute(f'SET search_path TO public; DROP SCHEMA {schema} CASCADE')
        await connection.close()
