from sqlalchemy import Column, Integer, String, DateTime, Index, desc
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from app.database import Base


class SiteContent(Base):
    """
    One row per version of a piece of static website copy -- the personal
    statement on the main page, the qualifications blurb, the journey
    timeline, and so on.

    To change a section, INSERT a new row with the same `section` value;
    never edit an existing row in place. Reads always take the row with the
    highest `id` for a given `section`, so the previous version stays in the
    table as history and can be restored by inserting it again. `created_at`
    is record metadata; it does not decide which version is current.

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
      "owner":     "<string>",    ?   # the site owner's NAME. Rendered as the
                                      #   page's only <h1>, above the title.
      "title":     "<string>",    ?   # professional title / role shown under
                                      #   the name, e.g. "Full-stack Engineer".
                                      #   Renamed from "heading"; the frontend
                                      #   still falls back to "heading" when
                                      #   "title" is absent, so an un-migrated
                                      #   row keeps rendering. Write "title" on
                                      #   every new row.
      "heading":   "<string>",    ?   # DEPRECATED -- former name of "title".
      "body":      "<string>",        # bio (required). Blank lines ->
                                      #   paragraphs; lines starting "- " /
                                      #   "* " -> a bullet list. Rendered by
                                      #   <Prose>, like every other body field.
      "resume": {                 ?   # the CV button beside the chat icon
        "label":   "<string>"     ?   #   its accessible name and tooltip
      },                              #   (default "Download CV"). NO key here:
                                      #   the PDF is a site_media row --
                                      #   section "personal_statement",
                                      #   description "resume", media_path =
                                      #   the PDF's S3 key. No row, no button.
                                      #   A "key" left on an older row is
                                      #   IGNORED.
      "skills": [                 ?   # skill pills under the role line
        {
          "group":  "<string>",       #   row label, e.g. "Frontend" (required).
                                      #   Announced to screen readers, not drawn.
          "colour": "<string>",   ?   #   DEPRECATED -- IGNORED. Pills used to
                                      #   take one of five named colours per
                                      #   group; that put up to five hues three
                                      #   lines from the top of the page and the
                                      #   grouping it encoded was never reliable
                                      #   (colours repeat once there are more
                                      #   groups than hues). Groups now simply
                                      #   ALTERNATE between the two brand
                                      #   colours, which shows where one group
                                      #   ends without pretending to identify
                                      #   it. Old rows carrying this key still
                                      #   parse; the value does nothing. Do not
                                      #   write it on new rows.
          "items":  ["<string>", ...] #   the pills themselves (required)
        }
      ],
      "cta": {                    ?   # call-to-action button
        "label":   "<string>",        #   button text
        "href":    "<string>"         #   route, e.g. "/chatroom"
      },
      "hero": {                     ? # hero-image FRAMING overrides. Any
        "fit": "cover"|"fitHeight", ? #   subset; unset fields use the frontend
        "height":       <number>,   ? #   defaults. Lets the About photo be
        "heightMin":    <number>,   ? #   re-framed from the DB with no redeploy.
        "heightMax":    <number>,   ? #   "fit": "cover" fills+crops the band;
        "focusX":       <number>,   ? #   "fitHeight" shows the whole photo on
        "focusY":       <number>,   ? #   the left.
        "zoom":         <number>,   ?
        "scrimFade":    <number>,   ? #   length of the photo -> page-background
                                      #   blend, as a % of the PHOTO's width.
        "textWidth":    <number>,   ?
        "mobileFocusX": <number>,   ? #   backdrop horizontal slice at <= 900px
        "tinyFocusX":   <number>    ? #   backdrop horizontal slice at <= 480px
                                      # Meaning of each, and the defaults:
                                      #   HeroOverrides / HERO_DEFAULTS in
                                      #   persona_stand_front/src/lib/knobs.ts
                                      #   -- THE authoritative list. Keep this
                                      #   block in step with it.
                                      # NOTE "scrimStart" and "scrimEnd" were
                                      #   documented here long after the
                                      #   frontend stopped reading them. They
                                      #   are IGNORED: the scrim's position is
                                      #   now derived from where the photo
                                      #   actually ends, and only "scrimFade"
                                      #   (how long the blend is) is
                                      #   configurable. Old rows carrying them
                                      #   still parse; the values do nothing.
      },
      "qualHero": { ...same keys as "hero"... }, ?
                                      # framing for the Qualifications & Awards
                                      #   banner (mirror layout: image on the
                                      #   RIGHT, text on the LEFT). Only shows
                                      #   when a site_media ("qualifications",
                                      #   "banner") row exists.
      "certHero": { ...same keys as "hero"... }, ?
                                      # framing for the Certifications banner
                                      #   (image LEFT, text RIGHT, like About).
                                      #   Only shows when a site_media
                                      #   ("certifications", "banner") row exists.
      # NO image key here. Every image on the site -- the hero included --
      # is a site_media row. The hero is section="personal_statement" with
      # TWO slots: description="hero_desk" (framed for the side-by-side
      # layout, >=900px) and "hero_mob" (the stacked one, <900px). The
      # browser fetches only the one its width matches. Either may be
      # omitted and the other stands in. (An earlier single "hero" slot is
      # no longer read.) A leftover "heroImage"
      # key on an old row is IGNORED by the frontend; migrate it with the
      # one-liner in persona_stand_ec2yml/Part_D.md ("Migrating an existing
      # environment").
    }

    ------------------------------------------------------------------
    section = "qualifications"          JSON ARRAY      Home -> About (Education)
    ------------------------------------------------------------------
    [
      {
        "id":          "<string>",     # stable key (required)
        "title":       "<string>",     # qualification / award name (required)
        "institution": "<string>", ?   # school / awarding body
        "year":        "<string>", ?   # e.g. "2024" or "2024 - 2026"
        "detail":      "<string>"  ?   # one extra line (or null)
      }
      # ... more items; array order = display order.
    ]
    # NOW EDUCATION ONLY. This section used to be its own full-bleed band
    # titled "Qualifications & Awards"; it renders inside About as a short
    # "Education" list, so a degree sits beside who you are instead of in a
    # band of its own. Awards and prizes moved to "certifications", which is
    # titled "Certification & Award" on the page -- put anything that is not
    # a degree there. /qualifications and /#qualifications now redirect to
    # that section (see RETIRED_SECTIONS in the frontend's lib/knobs.ts).
    #
    # A bare {"body": "<string>"} object is also accepted here for a
    # single free-text paragraph instead of a list -- it renders in the
    # "Certification & Award" section, above that list.

    ------------------------------------------------------------------
    section = "certifications"          JSON ARRAY      Home -> Certification & Award
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
        "id":        "<string>",       # stable key (required). Also the
                                       #   site_project.project_id that holds
                                       #   this project's pop-up detail.
        "label":     "<string>",       # caption + sheet heading (required);
                                       #   also the thumbnail alt when
                                       #   image_description is absent
        "image_description": "<string>", ?
                                       # ALT TEXT for the thumbnail: what the
                                       #   screenshot shows, not the project's
                                       #   name (the label under it already
                                       #   says that). Falls back to `label`.
        "overview":  "<string>",   ?   # the CARD's overview -- the short blurb
                                       #   revealed on hover. Meant as POINT
                                       #   FORM: write each line as "- ..."
                                       #   (rendered by <Prose>, same "- " /
                                       #   "* " -> bullets rule as every body
                                       #   field). This is DISTINCT from the
                                       #   pop-up's overview, which is a fuller
                                       #   PARAGRAPH write-up on the
                                       #   `site_project` row keyed by `id`.
                                       #   Omit it and the card falls back to
                                       #   that paragraph overview.
        "image_tag": "<string>"    ?   # names a site_media row -- section
                                       #   "projects", description == this value
                                       #   (defaults to `id`). The thumbnail URL
                                       #   is built from THAT row's media_path.
                                       #   No image path is stored in this row --
                                       #   every image comes from site_media.
      }
      # ... more items; array order = left-to-right order in the scroller
    ]
    # This is the card. Clicking a thumbnail opens a bottom pop-up whose
    # content lives in the separate `site_project` table
    # (app/models/site_project.py), keyed by this `id`. A project with no
    # site_project row just has a non-clickable thumbnail. The card and the
    # pop-up each have their OWN overview -- a point-form summary here, a
    # paragraph write-up there.

    ------------------------------------------------------------------
    section = "journey"                 JSON ARRAY      Home -> Journey
    ------------------------------------------------------------------
    [
      {
        "id":        "<string>",       # stable key + scroll anchor (required).
                                       #   MUST be unique within the array: it
                                       #   is the React key AND the element's
                                       #   DOM id, so a duplicate silently
                                       #   breaks the /#<id> anchor link.
        "year":      "<string>",       # timeline label, e.g. "2018" (required)
        "title":     "<string>",       # block heading (required)
        "institution": "<string>", ?   # school / company / organisation, shown
                                       #   in italics under the title.
        "body":      "<string>",       # block text (required). MUST be a
                                       #   JSON string -- a number or null here
                                       #   raises in the frontend's renderer.
                                       #   Blank lines -> paragraphs; lines
                                       #   starting "- " / "* " -> a bullet
                                       #   list ("point form"). Same renderer
                                       #   (<Prose>) as every other body field.
        "image_tag": "<string>"    ?   # names a site_media row -- section
                                       #   "journey", description == this value.
                                       #   That row's media_path shows on the
                                       #   OPPOSITE side of the card. Omit, or
                                       #   leave a "<...>" placeholder, for no
                                       #   image. NOT an S3 key itself -- the key
                                       #   lives in site_media.media_path.
        "image_description": "<string>", ?
                                       # ALT TEXT for that photo: what it shows,
                                       #   not what it is called ("Graduating
                                       #   from QUT", not "qut_img"). Omit and
                                       #   the image is marked decorative and
                                       #   skipped by screen readers.
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
      "links": [                  ?    # social links, rendered in the
        { "label": "<string>",         #   Contact section. A link whose
          "href":  "<string>" }        #   label contains "linkedin" or
      ]                                #   "github" also gets its icon.
    }

    ------------------------------------------------------------------
    section = "navbar"                  JSON OBJECT     Site header
    ------------------------------------------------------------------
    {
      "name": "<string>"          ?   # shown beside the brand mark in the
                                      #   navigation bar. Falls back to
                                      #   personal_statement."owner", so this
                                      #   row is only needed to show something
                                      #   different there.
    }
    # The mark itself is NOT in this row. It comes from the `site_media`
    # slot section="navbar", description="tab_logo"; its `media_path` is
    # resolved against the CDN base by the frontend.

    ------------------------------------------------------------------
    section = "footer"                  JSON OBJECT     Site footer
    ------------------------------------------------------------------
    {
      "owner": "<string>",        ?   # name in the copyright line. Falls back
                                      #   to personal_statement."owner".
      "note":  "<string>",        ?   # one short line beside it, e.g.
                                      #   "Built with React, FastAPI and AWS".
      "links": [                  ?   # links on the right. Falls back to
        { "label": "<string>",         #   contact."links", so social links do
          "href":  "<string>" }        #   not have to be written twice.
      ]
    }
    # The YEAR is never stored -- it is computed at render time. The footer
    # previously carried a hardcoded "2025 Company, Inc" that went stale.
    # Every field here is optional: with no row at all the footer still
    # renders a correct copyright line from the owner's name.

    ------------------------------------------------------------------
    section = "chatroom"                JSON OBJECT     Chatroom header
    ------------------------------------------------------------------
    {
      "name": "<string>"          ?   # the persona's display name, shown in
                                      #   the chat header. Falls back to
                                      #   personal_statement."owner", then to
                                      #   a generic label, so this row is
                                      #   optional -- add it only to show a
                                      #   name different from the site
                                      #   owner's.
    }
    # Header-only. The line under the name is live status
    # (online / offline / "Typing...") and is NOT content. The header
    # avatar is a fixed CDN object, not a site_media row (see AVATAR_URL
    # in persona_stand_front/src/pages/Chatroom.tsx).

    Media is NEVER stored in this table (nor in `site_journey` /
    `site_project`). EVERY image AND video the site renders -- hero, section
    banners, project thumbnails, journey block pictures, project demo clips
    -- is a `site_media` row (app/models/site_media.py; it holds any S3
    object key, .mp4 included), one per version of a (section, description)
    slot. The frontend gets them alongside this content from GET
    /api/site-content ("images" key) and resolves each key against the CDN
    base. A leftover "heroImage" key on an old personal_statement row is
    ignored; migrate it to a site_media row (Part_D.md).
    """
    __tablename__ = "site_content"
    __table_args__ = (
        # (key, id DESC): exactly the order the "current version" query reads
        # -- one index scan, no sort. The current version is the highest id,
        # not the newest created_at: these tables are single-owner, low-write
        # and append-only, so id order IS write order on the application's
        # write path, and created_at is kept as record metadata only.
        # EXISTING DATABASES need this by hand -- create_all never adds an
        # index to a table it did not create. See persona_stand_ec2yml/Part_C.md.
        Index("ix_site_content_section_id_desc", "section", desc("id")),
    )

    id = Column(Integer, primary_key=True)
    section = Column(String, nullable=False)
    content = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
