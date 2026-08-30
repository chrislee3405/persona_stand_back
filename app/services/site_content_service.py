import logging
from typing import Any

from fastapi import Depends
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.site_content import SiteContent

logger = logging.getLogger(__name__)


class SiteContentService:
    """
    Serves the static website copy (main-page personal statement, the
    qualifications blurb, and so on) out of the site_content table instead
    of it being hardcoded in the frontend, so wording can change without a
    redeploy.

    site_content holds one row per version of each section (see
    app/models/site_content.py): a change is a new INSERT with the same
    `section` slug, never an in-place UPDATE. Every read here takes the
    newest row for a section -- highest `created_at`, with `id` breaking
    ties -- so older versions stay in the table as restorable history.

    `content` is JSONB; its shape depends on the section (an object for
    prose sections, a list for the journey timeline). This service just
    passes it straight through -- SQLAlchemy hands back a dict/list and
    FastAPI serializes it back to JSON, so callers never parse a string.
    """

    def __init__(self, db: Session = Depends(get_db)):
        """
        Stores the injected database session.

        Parameters:
        - db (Session): SQLAlchemy session -- injected by FastAPI via get_db

        Returns:
        - None: sets self.db
        """
        self.db = db

    def get_section(self, section: str) -> SiteContent | None:
        """
        Fetches the current copy for one section.

        Parameters:
        - section (str): the section slug, e.g. "personal_statement" -- comes from the router's path parameter

        Returns:
        - SiteContent | None: the newest row for that section (newest created_at, then highest id), or None if the section has no rows
        """
        return (
            self.db.query(SiteContent)
            .filter(SiteContent.section == section)
            .order_by(desc(SiteContent.created_at), desc(SiteContent.id))
            .first()
        )

    def get_all_current(self) -> dict[str, Any]:
        """
        Fetches the current content for every section in one query -- what the main page loads on first paint.

        Parameters:
        - none

        Returns:
        - dict[str, Any]: section slug -> content (a dict or list, per the section's shape), one entry per distinct section (its newest row). Empty dict if site_content has no rows. Uses Postgres DISTINCT ON (section) with a matching ORDER BY so exactly the newest row per section comes back.
        """
        rows = (
            self.db.query(SiteContent)
            .distinct(SiteContent.section)
            .order_by(SiteContent.section, desc(SiteContent.created_at), desc(SiteContent.id))
            .all()
        )
        return {row.section: row.content for row in rows}
