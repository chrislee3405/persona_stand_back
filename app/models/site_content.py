from sqlalchemy import Column, Integer, String, DateTime, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from app.database import Base


class SiteContent(Base):
    """
    One row per version of a piece of static website copy -- the personal
    statement on the main page, the qualifications blurb, the journey
    timeline, and so on.

    To change a section, INSERT a new row with the same `section` value;
    never edit an existing row in place. Reads always take the newest row
    (highest `created_at`, then highest `id` as a tie-breaker) for a given
    `section`, so the previous version stays in the table as history and
    can be restored by inserting it again.

    `section` is a stable slug the frontend and back-end agree on, e.g.
    "personal_statement", "qualifications", "journey".

    `content` is JSONB and its shape depends on the section -- the section
    slug implies the schema. Nothing in this table enforces that shape;
    keep the writer disciplined (see the seed block in main.py for the
    canonical shapes). Examples:
      - "personal_statement": {"body": "Hi there! ..."}
      - "journey": [
            {"id": "2024-award", "year": "2024", "title": "...",
             "body": "...", "media": "journey/2024-award.jpg",
             "layout": "image-right"},
            ...
        ]
    `media` values are object-storage keys, never URLs or image data.
    `layout` is a hint the frontend maps to a layout/animation template;
    the animation code itself lives only in the frontend.
    """
    __tablename__ = "site_content"
    __table_args__ = (
        # Serves the only query this table has: "newest row for this
        # section" (and the Postgres DISTINCT ON form that fetches the
        # newest row for every section at once).
        Index("ix_site_content_section_created_at", "section", "created_at"),
    )

    id = Column(Integer, primary_key=True, index=True)
    section = Column(String, nullable=False)
    content = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
