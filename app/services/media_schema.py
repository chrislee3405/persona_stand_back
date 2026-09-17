"""Prevent create_all from hiding an unmigrated asset table with an empty one."""
from sqlalchemy import text


async def require_media_schema(connection):
    if await connection.scalar(text("SELECT to_regclass('site_image')")):
        raise RuntimeError(
            'Legacy site_image table exists. Stop the backend and run '
            'scripts/migrations/20260917_site_media.sql before starting or seeding.'
        )
    if await connection.scalar(text("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = current_schema() AND table_name = 'site_media'
              AND column_name = 'image_path'
        )
    """)):
        raise RuntimeError('site_media still has image_path; run scripts/migrations/20260917_site_media.sql.')
