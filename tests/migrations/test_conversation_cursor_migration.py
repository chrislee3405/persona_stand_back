"""Exercise the last_handled_index migration in an isolated schema of the disposable test DB."""
import os
from pathlib import Path
from uuid import uuid4

import asyncpg
import pytest

from tests.support.environment import validate_test_database_url

pytestmark = pytest.mark.integration
SQL = (Path(__file__).resolve().parents[2] / 'scripts/migrations/20260920_conversation_last_handled_index.sql').read_text()

# The two tables as they stood before the readiness gate: no cursor column.
_PRE_GATE_SCHEMA = '''
    CREATE TABLE conversation (
        conversation_id varchar PRIMARY KEY,
        last_summarized_index integer NOT NULL DEFAULT -1
    );
    CREATE TABLE message (
        message_id serial PRIMARY KEY,
        conversation_id varchar NOT NULL REFERENCES conversation (conversation_id),
        order_index integer NOT NULL,
        sender varchar NOT NULL,
        text text NOT NULL,
        UNIQUE (conversation_id, order_index)
    );
    INSERT INTO conversation (conversation_id) VALUES ('answered'), ('failed'), ('empty');
    INSERT INTO message (conversation_id, order_index, sender, text) VALUES
        ('answered', 0, 'user', 'q1'), ('answered', 1, 'backend', 'a1'),
        ('answered', 2, 'user', 'q2'), ('answered', 3, 'backend', 'a2'),
        ('failed', 0, 'not_saved_user', 'q1'), ('failed', 1, 'error', '[incident x] boom');
'''


async def _isolated_connection():
    url = validate_test_database_url(os.environ['TEST_DATABASE_URL']).replace('postgresql+asyncpg:', 'postgresql:')
    connection = await asyncpg.connect(url)
    schema = 'migration_' + uuid4().hex
    await connection.execute(f'CREATE SCHEMA {schema}; SET search_path TO {schema}')
    return connection, schema


async def _drop(connection, schema):
    await connection.execute('ROLLBACK')
    await connection.execute(f'SET search_path TO public; DROP SCHEMA {schema} CASCADE')
    await connection.close()


async def _cursors(connection) -> dict[str, int]:
    rows = await connection.fetch('SELECT conversation_id, last_handled_index FROM conversation')
    return {row['conversation_id']: row['last_handled_index'] for row in rows}


async def test_existing_conversations_start_fully_handled():
    connection, schema = await _isolated_connection()
    try:
        await connection.execute(_PRE_GATE_SCHEMA)
        await connection.execute(SQL)
        # Nothing an old backend already dealt with may read as pending: every
        # row that exists is at or below its conversation's cursor.
        assert await _cursors(connection) == {'answered': 3, 'failed': 1, 'empty': -1}
        # New conversations still start at the model's -1.
        await connection.execute("INSERT INTO conversation (conversation_id) VALUES ('new')")
        assert (await _cursors(connection))['new'] == -1
    finally:
        await _drop(connection, schema)


async def test_rerun_never_moves_a_live_cursor():
    connection, schema = await _isolated_connection()
    try:
        await connection.execute(_PRE_GATE_SCHEMA)
        await connection.execute(SQL)
        # The gate is live: a message is held (above the cursor) when the file
        # is run a second time. The rerun must leave it pending.
        await connection.execute(
            "INSERT INTO message (conversation_id, order_index, sender, text) VALUES ('answered', 4, 'user', 'held')"
        )
        await connection.execute(SQL)
        assert (await _cursors(connection))['answered'] == 3
    finally:
        await _drop(connection, schema)


async def test_missing_conversation_table_aborts():
    connection, schema = await _isolated_connection()
    try:
        with pytest.raises(asyncpg.RaiseError, match='No conversation table exists'):
            await connection.execute(SQL)
    finally:
        await _drop(connection, schema)
