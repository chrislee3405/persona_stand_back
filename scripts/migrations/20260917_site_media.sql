-- Stop old backend instances before running. Metadata-only rename; preserves
-- every row, ID, object key, timestamp, default and index definition.
BEGIN;
SET LOCAL lock_timeout = '10s';
DO $$
BEGIN
    IF to_regclass('site_image') IS NOT NULL THEN
        IF to_regclass('site_media') IS NOT NULL THEN
            RAISE EXCEPTION 'Both site_image and site_media exist; reconcile them manually before migration';
        END IF;
        ALTER TABLE site_image RENAME TO site_media;
    END IF;
    IF to_regclass('site_media') IS NULL THEN
        RAISE EXCEPTION 'No asset table exists; use the new application schema for a fresh database';
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_schema = current_schema() AND table_name = 'site_media'
                 AND column_name = 'image_path') THEN
        IF EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema = current_schema() AND table_name = 'site_media'
                     AND column_name = 'media_path') THEN
            RAISE EXCEPTION 'Both path columns exist; reconcile before migration';
        END IF;
        ALTER TABLE site_media RENAME COLUMN image_path TO media_path;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema = current_schema() AND table_name = 'site_media'
                     AND column_name = 'media_path') THEN
        RAISE EXCEPTION 'Asset path column is missing';
    END IF;
    IF to_regclass('ix_site_image_section_description_id_desc') IS NOT NULL THEN
        ALTER INDEX ix_site_image_section_description_id_desc
            RENAME TO ix_site_media_section_description_id_desc;
    END IF;
    IF EXISTS (SELECT 1 FROM pg_constraint
               WHERE conrelid = 'site_media'::regclass AND conname = 'site_image_pkey') THEN
        ALTER TABLE site_media RENAME CONSTRAINT site_image_pkey TO site_media_pkey;
    END IF;
    IF to_regclass('site_image_id_seq') IS NOT NULL THEN
        ALTER SEQUENCE site_image_id_seq RENAME TO site_media_id_seq;
    END IF;
END $$;
COMMIT;
