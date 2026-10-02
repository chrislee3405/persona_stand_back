import json
from pathlib import Path

import asyncpg
import pytest

from tests.migrations.test_conversation_cursor_migration import _isolated_connection, _drop

pytestmark = pytest.mark.integration
SQL = (Path(__file__).resolve().parents[2] / "scripts/migrations/20261002_message_token_usage.sql").read_text()


async def test_add_and_rerun_preserves_messages_and_existing_usage():
    connection, schema = await _isolated_connection()
    try:
        await connection.execute("CREATE TABLE message (message_id integer PRIMARY KEY, text text NOT NULL); "
                                 "INSERT INTO message VALUES (1, 'existing message')")
        await connection.execute(SQL)
        assert await connection.fetchval("SELECT token_usage FROM message") is None
        usage = {"schema_version": 1, "runs": [{"run_id": "preserve-me"}]}
        await connection.execute("UPDATE message SET token_usage=$1::jsonb", json.dumps(usage))
        await connection.execute(SQL)
        row = await connection.fetchrow("SELECT * FROM message")
        assert row["text"] == "existing message" and json.loads(row["token_usage"]) == usage
    finally:
        await _drop(connection, schema)


@pytest.mark.parametrize("column", ["text", "jsonb NOT NULL DEFAULT '{}'::jsonb"])
async def test_wrong_existing_column_rolls_back_without_altering_data(column):
    connection, schema = await _isolated_connection()
    try:
        await connection.execute(f"CREATE TABLE message (token_usage {column})")
        with pytest.raises(asyncpg.RaiseError, match="must be nullable jsonb"):
            await connection.execute(SQL)
    finally:
        await _drop(connection, schema)


async def test_missing_table_fails_with_actionable_message():
    connection, schema = await _isolated_connection()
    try:
        with pytest.raises(asyncpg.RaiseError, match="No message table exists"):
            await connection.execute(SQL)
    finally:
        await _drop(connection, schema)
