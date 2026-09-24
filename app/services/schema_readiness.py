"""Read-only checks for the schema required by this application version."""
from sqlalchemy import text

from app.services.media_schema import require_media_schema


async def require_chat_schema(connection):
    await require_media_schema(connection)
    exists = await connection.scalar(text("SELECT to_regclass('conversation')"))
    if exists and not await connection.scalar(text("""
        SELECT EXISTS (SELECT 1 FROM information_schema.columns
        WHERE table_schema = current_schema() AND table_name = 'conversation'
          AND column_name = 'last_handled_index' AND is_nullable = 'NO')
    """)):
        raise RuntimeError(
            "Conversation cursor migration required. Stop the old backend and run "
            "scripts/migrations/20260920_conversation_last_handled_index.sql."
        )


async def check_readiness(connection):
    await require_chat_schema(connection)
    # Resolve required columns and table permissions even when there are no rows.
    await connection.execute(text("SELECT last_handled_index FROM conversation LIMIT 0"))
    await connection.execute(text("SELECT order_index FROM message LIMIT 0"))
