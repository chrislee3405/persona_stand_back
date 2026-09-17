from unittest.mock import AsyncMock

import pytest

from app.services.media_schema import require_media_schema


async def test_guard_refuses_legacy_table_before_schema_creation():
    connection = AsyncMock()
    connection.scalar.return_value = 'site_image'
    with pytest.raises(RuntimeError, match='run scripts/migrations|scripts/migrations'):
        await require_media_schema(connection)


async def test_guard_accepts_new_or_fresh_schema():
    connection = AsyncMock()
    connection.scalar.side_effect = [None, False]
    await require_media_schema(connection)


async def test_guard_refuses_unmigrated_column():
    connection = AsyncMock()
    connection.scalar.side_effect = [None, True]
    with pytest.raises(RuntimeError, match='image_path'):
        await require_media_schema(connection)
