# Asset table rename

`20260917_site_media.sql` renames `site_image` to `site_media` and `image_path`
to `media_path`, including the standard index, primary key and sequence names.
It preserves records and tag connections. It is safe to rerun after success;
ambiguous tables or columns cause a rollback instead of merging or dropping data.

For an existing database:

1. Back up the database and stop old backend instances (including workers).
2. Run with the intended database connection:
   `psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f scripts/migrations/20260917_site_media.sql`
   Use a PostgreSQL URI, not a SQLAlchemy `postgresql+asyncpg` URI.
3. Start the new backend, then deploy the frontend. Verify `/api/site-content`
   returns the expected `media` entries and test images, video/posters and PDFs.

Fresh databases use `SiteMedia` through normal schema creation and
`app/models/seed/site_media.json`. Existing site tables are not updated by the
seed loader. Never create a second empty table to replace the old one.

The API temporarily returns both `media` and the legacy `images` alias from
one query for already-open browsers. The new frontend prefers `media` and can
read `images` during deployment. Each entry still has `description` and `path`;
only the database/seed field is `media_path`. Remove the alias in a later
coordinated release. Source/poster/image tags are unchanged.

Application rollback requires stopping the new backend and reversing the table,
column, index, primary-key and sequence renames before starting old code.
The forward SQL has not been applied to a development or production database
by this change.
