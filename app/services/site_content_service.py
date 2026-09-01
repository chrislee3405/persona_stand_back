from typing import Any

from fastapi import Depends
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.site_content import SiteContent
from app.models.site_image import SiteImage
from app.models.site_journey import SiteJourney


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

    Images are kept out of `content` entirely -- they live in the
    `site_image` table (app/models/site_image.py), one row per version of
    each (section, description) image slot, and get_all_images() reads them
    with the same "newest row wins" rule. The Journey click-through detail
    sheets live in `site_journey` (app/models/site_journey.py), read by
    get_all_journey_details() the same way.

    The frontend only ever calls GET /api/site-content, so the three
    "fetch everything" methods below are all this service needs.
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

    def get_all_images(self) -> dict[str, list[dict[str, str]]]:
        """
        Fetches the current image for every (section, description) slot, grouped by section -- the picture side of what the main page loads on first paint.

        Parameters:
        - none

        Returns:
        - dict[str, list[dict[str, str]]]: section slug -> list of {"description": ..., "path": ...} (the S3 object key), one entry per distinct (section, description) slot (its newest row). Empty dict if site_image has no rows. Uses Postgres DISTINCT ON (section, description) with a matching ORDER BY so exactly the newest row per slot comes back.
        """
        rows = (
            self.db.query(SiteImage)
            .distinct(SiteImage.section, SiteImage.description)
            .order_by(
                SiteImage.section,
                SiteImage.description,
                desc(SiteImage.created_at),
                desc(SiteImage.id),
            )
            .all()
        )
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            grouped.setdefault(row.section, []).append(
                {"description": row.description, "path": row.image_path}
            )
        return grouped

    def get_all_journey_details(self) -> dict[str, Any]:
        """
        Fetches the current detail sheet for every journey block in one query -- the expanded content the main page shows in the bottom pop-up when a Journey card is clicked.

        Parameters:
        - none

        Returns:
        - dict[str, Any]: journey_id -> content (a dict, per site_journey's shape), one entry per distinct journey_id (its newest row). Empty dict if site_journey has no rows. Uses Postgres DISTINCT ON (journey_id) with a matching ORDER BY so exactly the newest row per block comes back.
        """
        rows = (
            self.db.query(SiteJourney)
            .distinct(SiteJourney.journey_id)
            .order_by(
                SiteJourney.journey_id,
                desc(SiteJourney.created_at),
                desc(SiteJourney.id),
            )
            .all()
        )
        return {row.journey_id: row.content for row in rows}
