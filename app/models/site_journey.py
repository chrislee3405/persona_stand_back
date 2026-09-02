from sqlalchemy import Column, Integer, String, DateTime, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from app.database import Base


class SiteJourney(Base):
    """
    The long-form detail behind ONE journey block -- the content shown in
    the bottom pop-up ("sheet") when a visitor clicks a card in the Journey
    timeline on the main page.

    The timeline itself still comes from `site_content` (section "journey"):
    a short array of {id, year, title, body, image_tag}. That is the summary
    on the card. This table holds the expanded story for a card, keyed by
    that block's `id`, so the two can be edited on different schedules and a
    block can exist on the timeline with no detail sheet yet (the card just
    is not clickable).

    Same "newest row wins, never UPDATE" rule as site_content / site_image:
    to change a block's detail, INSERT a new row with the same `journey_id`;
    reads take the newest row (highest created_at, then id) per journey_id,
    so older versions stay as restorable history.

    Columns
    -------
    id          serial PK.
    journey_id  The `id` of the block in the `site_content` "journey" array
                this detail belongs to, e.g. "2024-master-ai". Not a DB
                foreign key (the journey array is JSONB, not rows) -- the
                writer keeps the two in sync. One logical detail per
                journey_id; its newest row wins.
    content     JSONB. `?` marks optional keys. Templates, not literal JSON:

                {
                  "heading":  "<string>",   ?  # sheet title; falls back to
                                               #   the block's `title`
                  "subtitle": "<string>",   ?  # one italic line under it,
                                               #   e.g. place / role
                  "body":     "<string>",      # main text (required). Blank
                                               #   lines -> paragraphs, same
                                               #   as every other body field.
                  "highlights": ["<string>", ...],  ?  # bullet list under
                                                       #   the body
                  "links": [                  ?  # related links, rendered as
                    { "label": "<string>",       #   buttons at the bottom
                      "href":  "<string>" }
                  ]
                }

    created_at  defaults to now(); newest row per journey_id wins.

    Served to the frontend on first paint inside GET /api/site-content as
    `journeyDetails: { "<journey_id>": <content>, ... }`.

    See also persona_stand_ec2yml/Part_D.md (D.2 seed SQL, D.5 shapes).
    """
    __tablename__ = "site_journey"
    __table_args__ = (
        # Serves the only query this table has: "newest row for this
        # journey_id" and the DISTINCT ON form that fetches the newest row
        # for every journey_id at once.
        Index("ix_site_journey_journey_id_created_at", "journey_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, index=True)
    journey_id = Column(String, nullable=False)
    content = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
