from pathlib import Path

import asyncpg
import pytest

from tests.migrations.test_conversation_cursor_migration import _isolated_connection, _drop

pytestmark = pytest.mark.integration
SQL = (Path(__file__).resolve().parents[2] / "scripts/migrations/20261005_model_configuration.sql").read_text()
PRIORITY_SQL = (Path(__file__).resolve().parents[2] / "scripts/migrations/20261006_model_configuration_use_priority.sql").read_text()


async def test_selection_upgrade_allows_multiple_enabled_providers_and_preserves_settings():
    root = Path(__file__).resolve().parents[2] / "scripts/migrations"
    connection, schema = await _isolated_connection()
    try:
        await connection.execute(SQL)
        await connection.execute((root / "20261007_model_configuration_providers.sql").read_text())
        before = [dict(row) for row in await connection.fetch("SELECT * FROM model_configuration ORDER BY id")]
        migration = (root / "20261008_model_configuration_selection.sql").read_text()
        await connection.execute(migration)
        await connection.execute(migration)
        assert [dict(row) for row in await connection.fetch("SELECT * FROM model_configuration ORDER BY id")] == before
        await connection.execute("UPDATE model_configuration SET is_active=true WHERE provider='gemini'")
        assert await connection.fetchval("SELECT count(*) FROM model_configuration WHERE is_active") == 2
        with pytest.raises(asyncpg.UniqueViolationError):
            await connection.execute("UPDATE model_configuration SET use_priority=1 WHERE provider='gemini'")
        await connection.execute("UPDATE model_configuration SET is_active=false WHERE provider='gemini'")
        await connection.execute("UPDATE model_configuration SET use_priority=1 WHERE provider='gemini'")
    finally:
        await _drop(connection, schema)


async def test_seed_and_rerun_preserve_owner_configuration():
    connection, schema = await _isolated_connection()
    try:
        await connection.execute(SQL)
        assert await connection.fetchval("SELECT provider FROM model_configuration") == "openai"
        assert await connection.fetchval("SELECT default_model FROM model_configuration") == "gpt-4.1-mini"
        assert await connection.fetchval("SELECT use_priority FROM model_configuration") == 1
        await connection.execute("UPDATE model_configuration SET provider='grok', default_model='chosen', summary_model=NULL")
        await connection.execute(SQL)
        row = await connection.fetchrow("SELECT * FROM model_configuration")
        assert row["provider"] == "grok" and row["default_model"] == "chosen"
        assert row["summary_model"] is None
        assert await connection.fetchval("SELECT count(*) FROM model_configuration") == 1
    finally:
        await _drop(connection, schema)


@pytest.mark.parametrize("sql", [
    "INSERT INTO model_configuration (id, provider, default_model) VALUES (2, 'gemini', 'x')",
    "UPDATE model_configuration SET provider='unknown'",
    "UPDATE model_configuration SET default_model=' '",
    "UPDATE model_configuration SET response_model=''",
    "UPDATE model_configuration SET use_priority=0",
])
async def test_singleton_and_settings_constraints(sql):
    connection, schema = await _isolated_connection()
    try:
        await connection.execute(SQL)
        with pytest.raises(asyncpg.CheckViolationError):
            await connection.execute(sql)
    finally:
        await _drop(connection, schema)


async def test_priority_upgrade_preserves_configuration_and_reruns():
    connection, schema = await _isolated_connection()
    try:
        await connection.execute("CREATE TABLE model_configuration (id integer PRIMARY KEY, provider text, default_model text);")
        await connection.execute("INSERT INTO model_configuration VALUES (1, 'grok', 'chosen')")
        await connection.execute(PRIORITY_SQL)
        assert await connection.fetchval("SELECT use_priority FROM model_configuration") == 1
        await connection.execute("UPDATE model_configuration SET use_priority=3")
        await connection.execute(PRIORITY_SQL)
        row = await connection.fetchrow("SELECT * FROM model_configuration")
        assert dict(row) == dict(id=1, provider="grok", default_model="chosen", use_priority=3)
        with pytest.raises(asyncpg.CheckViolationError):
            await connection.execute("UPDATE model_configuration SET use_priority=-1")
    finally:
        await _drop(connection, schema)


async def test_provider_upgrade_preserves_selection_and_supports_switching():
    upgrade = (Path(__file__).resolve().parents[2] / "scripts/migrations/20261007_model_configuration_providers.sql").read_text()
    connection, schema = await _isolated_connection()
    try:
        await connection.execute(SQL)
        await connection.execute("UPDATE model_configuration SET provider='gemini', default_model='owner-model', summary_model='owner-summary'")
        await connection.execute(upgrade)
        await connection.execute(upgrade)
        assert await connection.fetchval("SELECT count(*) FROM model_configuration") == 2
        active = await connection.fetchrow("SELECT * FROM model_configuration WHERE is_active")
        assert active["provider"] == "gemini" and active["default_model"] == "owner-model"
        assert active["summary_model"] == "owner-summary"
        with pytest.raises(asyncpg.UniqueViolationError):
            await connection.execute("UPDATE model_configuration SET is_active=true WHERE provider='openai'")
        async with connection.transaction():
            await connection.execute("UPDATE model_configuration SET is_active=false WHERE is_active")
            await connection.execute("UPDATE model_configuration SET is_active=true WHERE provider='openai'")
        await connection.execute(upgrade)
        assert await connection.fetchval("SELECT provider FROM model_configuration WHERE is_active") == "openai"
    finally:
        await _drop(connection, schema)
