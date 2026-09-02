from sqlalchemy import Column, Integer, String, DateTime, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from app.database import Base


class SiteProject(Base):
    """
    The detail content behind ONE project -- shown in the bottom pop-up
    ("sheet") when a visitor clicks a thumbnail in the Projects banner on
    the main page. The Projects timeline/banner itself still comes from
    `site_content` (section "projects"): a short array of
    {id, label, image_tag}. That is the card. This table holds the expanded
    content for a card, keyed by that project's `id`, so the two can be
    edited on different schedules and a project can exist in the banner with
    no detail sheet yet (its thumbnail just is not clickable).

    Same "newest row wins, never UPDATE" rule as the rest of the site_*
    tables: to change a project's detail, INSERT a new row with the same
    `project_id`; reads take the newest row (highest created_at, then id)
    per project_id.

    Columns
    -------
    id          serial PK.
    project_id  The `id` of the project in the `site_content` "projects"
                array this detail belongs to, e.g. "ransom-simulator". Not a
                DB foreign key (the projects array is JSONB) -- the writer
                keeps the two in sync. One logical detail per project_id;
                its newest row wins.
    content     JSONB. `?` marks optional keys. Templates, not literal JSON:

                {
                  "overview":     "<string>",  ?  # prose; blank lines ->
                                                  #   paragraphs (like every
                                                  #   other body field)
                  "features":     ["<string>", ...],  ?  # "Main features" list
                  "technologies": ["<string>", ...],  ?  # "Technologies used" list
                  "githubUrl":    "<string>",  ?  # repo link
                  "demoUrl":      "<string>",  ?  # live demo link
                  "videos": [                  ?  # feature-demo clips, shown
                                                  #   on the right, one playing
                                                  #   at a time as you scroll
                    { "src_tag":    "<string>",     # -> site_image row
                                                    #    (section "projects",
                                                    #    description == this),
                                                    #    image_path = the .mp4
                                                    #    S3 KEY
                      "poster_tag": "<string>",  ?  # -> site_image row, a .jpg
                                                    #    still shown before play
                      "caption":    "<string>" }  ? # line under the clip
                  ]
                }

                NB: no asset PATH is stored here -- video + poster come from
                site_image rows referenced by tag, same rule as everything
                else (see app/models/site_image.py).

    created_at  defaults to now(); newest row per project_id wins.

    Served to the frontend on first paint inside GET /api/site-content as
    `projectDetails: { "<project_id>": <content>, ... }`.

    See also persona_stand_ec2yml/Part_D.md (D.2 seed SQL, D.5 shapes).
    """
    __tablename__ = "site_project"
    __table_args__ = (
        # Serves the only query this table has: "newest row for this
        # project_id" and the DISTINCT ON form that fetches the newest row
        # for every project_id at once.
        Index("ix_site_project_project_id_created_at", "project_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(String, nullable=False)
    content = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
