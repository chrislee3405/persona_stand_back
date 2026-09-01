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

    `section` is a stable slug the frontend and back-end agree on. The
    sections and the exact JSON shape the frontend expects for each are
    below. `content` is JSONB; nothing in this table enforces the shape,
    so keep the writer disciplined. `?` marks an optional key. These are
    templates, not literal JSON -- fill the <...> placeholders. See also
    persona_stand_ec2yml/Part_D.md (D.2 seed SQL, D.5 shapes table).

    ------------------------------------------------------------------
    section = "personal_statement"      JSON OBJECT     Home -> About
    ------------------------------------------------------------------
    {
      "heading":   "<string>",    ?   # small heading above the bio
      "body":      "<string>",        # bio paragraph (required)
      "cta": {                    ?   # call-to-action button
        "label":   "<string>",        #   button text
        "href":    "<string>"         #   route, e.g. "/chatroom"
      },
      "hero": {                   ?   # hero-image FRAMING overrides. Any
        "fit": "cover"|"fitHeight",?  #   subset; unset fields use the frontend
        "height":     <number>,  ?   #   defaults. Lets the About photo be
        "heightMin":  <number>,  ?   #   re-framed from the DB with no redeploy.
        "heightMax":  <number>,  ?   #   "fit": "cover" fills+crops the band;
        "focusX":     <number>,  ?   #   "fitHeight" shows the whole photo on
        "focusY":     <number>,  ?   #   the left. Meaning of each: see
        "zoom":       <number>,  ?   #   HERO_DEFAULTS in
        "scrimStart": <number>,  ?   #   persona_stand_front/src/pages/Home.tsx
        "scrimEnd":   <number>,  ?
        "textWidth":  <number>   ?
      },
      "qualHero": { ...same keys as "hero"... }, ?
                                      # framing for the Qualifications & Awards
                                      #   banner (mirror layout: image on the
                                      #   RIGHT, text on the LEFT). Only shows
                                      #   when a site_image ("qualifications",
                                      #   "banner") row exists.
      "certHero": { ...same keys as "hero"... }, ?
                                      # framing for the Certifications banner
                                      #   (image LEFT, text RIGHT, like About).
                                      #   Only shows when a site_image
                                      #   ("certifications", "banner") row exists.
      # NO image key here. Every image on the site -- the hero included --
      # is a site_image row. The hero is section="personal_statement",
      # description="hero" (app/models/site_image.py). A leftover "heroImage"
      # key on an old row is IGNORED by the frontend; migrate it with the
      # one-liner in persona_stand_ec2yml/Part_D.md ("Migrating an existing
      # environment").
    }

    ------------------------------------------------------------------
    section = "qualifications"          JSON ARRAY      Home -> Qualifications & Awards
    ------------------------------------------------------------------
    [
      {
        "id":          "<string>",     # stable key (required)
        "title":       "<string>",     # qualification / award name (required)
        "institution": "<string>", ?   # school / awarding body
        "year":        "<string>", ?   # e.g. "2024" or "2024 - 2026"
        "detail":      "<string>"  ?   # one extra line (or null)
      }
      # ... more items; array order = display order. Degrees and awards
      # share this list -- put degrees first, then awards.
    ]
    # A bare {"body": "<string>"} object is also accepted here for a
    # single free-text paragraph instead of a list.

    ------------------------------------------------------------------
    section = "certifications"          JSON ARRAY      Home -> Certifications
    ------------------------------------------------------------------
    [
      {
        "id":     "<string>",          # stable key (required)
        "title":  "<string>",          # certification name (required)
        "issuer": "<string>",      ?   # awarding body / exam board
        "year":   "<string>",      ?   # e.g. "2023"
        "detail": "<string>"       ?   # one extra line (or null)
      }
      # ... more items; array order = display order
    ]

    ------------------------------------------------------------------
    section = "projects"                JSON ARRAY      Home -> Projects
    ------------------------------------------------------------------
    [
      {
        "id":        "<string>",       # stable key AND route slug (required):
                                       #   the thumbnail and the navbar
                                       #   "Projects" dropdown both link to
                                       #   /projects/<id>, so this must match a
                                       #   project page route.
        "label":     "<string>",       # caption + <img alt> + dropdown text (required)
        "image_tag": "<string>"    ?   # names a site_image row -- section
                                       #   "projects", description == this value
                                       #   (defaults to `id`). The thumbnail URL
                                       #   is built from THAT row's image_path.
                                       #   No image path is stored in this row --
                                       #   every image comes from site_image.
      }
      # ... more items; array order = left-to-right order in the scroller
      #     AND top-to-bottom order in the navbar dropdown
    ]

    ------------------------------------------------------------------
    section = "journey"                 JSON ARRAY      Home -> Journey
    ------------------------------------------------------------------
    [
      {
        "id":        "<string>",       # stable key + scroll anchor (required)
        "year":      "<string>",       # timeline label, e.g. "2018" (required)
        "title":     "<string>",       # block heading (required)
        "body":      "<string>",       # block paragraph (required)
        "image_tag": "<string>"    ?   # names a site_image row -- section
                                       #   "journey", description == this value.
                                       #   That row's image_path shows on the
                                       #   OPPOSITE side of the card. Omit, or
                                       #   leave a "<...>" placeholder, for no
                                       #   image. NOT an S3 key itself -- the key
                                       #   lives in site_image.image_path.
      }
      # ... more blocks; array order = top-to-bottom order
    ]
    # The card shows this summary. The long-form story behind a block --
    # shown in a bottom pop-up when the card is clicked -- lives in the
    # separate `site_journey` table (app/models/site_journey.py), keyed by
    # the block's `id`. A block with no site_journey row just has a
    # non-clickable card.

    ------------------------------------------------------------------
    section = "contact"                 JSON OBJECT     Contact page + footer
    ------------------------------------------------------------------
    {
      "email":    "<string>",          # rendered as a mailto link (required)
      "intro":    "<string>",     ?    # short line above the details
      "location": "<string>",     ?    # e.g. "Brisbane, Australia"
      "links": [                  ?    # social links; the footer shows the
        { "label": "<string>",         #   LinkedIn / GitHub ones, matched by
          "href":  "<string>" }        #   label containing "linkedin" /
      ]                                #   "github" (case-insensitive)
    }

    Images are NEVER stored in this table (nor in `site_journey`). EVERY
    image the site renders -- hero, section banners, project thumbnails,
    journey block pictures -- is a `site_image` row (app/models/
    site_image.py), one per version of a (section, description) slot,
    holding the S3 object key. The frontend gets them alongside this content
    from GET /api/site-content ("images" key) and resolves each key against
    the CDN base. A leftover "heroImage" key on an old personal_statement
    row is ignored; migrate it to a site_image row (Part_D.md).
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
