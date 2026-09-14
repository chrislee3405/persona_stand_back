from sqlalchemy import Column, Integer, String, DateTime, Index, desc
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from app.database import Base


class SiteProject(Base):
    """
    The detail content behind ONE project -- shown in the bottom pop-up
    ("sheet") when a visitor clicks a thumbnail in the Projects banner on
    the main page. The Projects timeline/banner itself still comes from
    `site_content` (section "projects"): a short array of
    {id, label, overview, image_tag} -- where that `overview` is the card's
    own short point-form blurb, separate from this table's paragraph
    `overview` below. That is the card. This table holds the expanded
    content for a card, keyed by that project's `id`, so the two can be
    edited on different schedules and a project can exist in the banner with
    no detail sheet yet (its thumbnail just is not clickable).

    Same "highest id wins, never UPDATE" rule as the rest of the site_*
    tables: to change a project's detail, INSERT a new row with the same
    `project_id`; reads take the row with the highest id per project_id.

    Columns
    -------
    id          serial PK.
    project_id  The `id` of the project in the `site_content` "projects"
                array this detail belongs to, e.g. "persona-stand". Not a
                DB foreign key (the projects array is JSONB) -- the writer
                keeps the two in sync. One logical detail per project_id;
                its highest-id row wins.
    content     JSONB. `?` marks optional keys. Templates, not literal JSON:

                {
                  "overview":     "<string>",  ?  # the POP-UP overview -- the
                                                  #   fuller PARAGRAPH write-up
                                                  #   shown in the sheet. Blank
                                                  #   lines -> paragraphs, and
                                                  #   "- " / "* " lines -> a
                                                  #   bullet list, via <Prose>
                                                  #   (same as every body
                                                  #   field). The thumbnail
                                                  #   CARD has its own SHORT
                                                  #   point-form overview on the
                                                  #   `site_content` "projects"
                                                  #   row; this one is not shown
                                                  #   on the card unless that
                                                  #   key is missing.
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

    created_at  defaults to now(); record metadata only -- the highest id per
                project_id is the current version.

    Served to the frontend on first paint inside GET /api/site-content as
    `projectDetails: { "<project_id>": <content>, ... }`.

    See also persona_stand_ec2yml/Part_D.md (D.2 seed SQL, D.5 shapes).
    """
    __tablename__ = "site_project"
    __table_args__ = (
        # (key, id DESC): exactly the order the "current version" query reads
        # -- one index scan, no sort. The current version is the highest id,
        # not the newest created_at: these tables are single-owner, low-write
        # and append-only, so id order IS write order on the application's
        # write path, and created_at is kept as record metadata only.
        # EXISTING DATABASES need this by hand -- create_all never adds an
        # index to a table it did not create. See persona_stand_ec2yml/Part_C.md.
        Index("ix_site_project_project_id_id_desc", "project_id", desc("id")),
    )

    id = Column(Integer, primary_key=True)
    project_id = Column(String, nullable=False)
    content = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
