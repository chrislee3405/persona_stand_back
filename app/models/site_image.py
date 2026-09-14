from sqlalchemy import Column, Integer, String, DateTime, Index, desc
from sqlalchemy.sql import func
from app.database import Base


class SiteImage(Base):
    """
    One row per version of an image used somewhere in the static website
    copy. This is the image counterpart of `site_content` (app/models/
    site_content.py): `site_content` holds the text of each section as
    JSONB, `site_image` holds the picture(s) that section shows.

    Why a separate table instead of an "image" key inside the section JSON:
    images change on their own schedule (a new photo, a re-crop), they are
    referenced by more than one section shape (an object section like
    `personal_statement`, an array section like `qualifications`), and the
    S3 key is a single flat string that does not need the freedom of JSONB.

    Columns
    -------
    id           serial PK.
    section      Which `site_content` section this image belongs to -- the
                 same slug set the frontend already knows:
                 "personal_statement", "qualifications", "certifications",
                 "journey", "contact".
    description  A stable slot label for the image within its section, and
                 the text used as the <img alt>. e.g. "hero",
                 "qualifications banner". (section, description) together
                 identify one logical image slot; insert a new row with the
                 same pair to swap the picture in that slot.
                 For section "journey" the description is the `image_tag`
                 named by a block in the journey site_content array.
    image_path   S3 object KEY only, e.g. "about_me/main_img.png" -- never a
                 full URL, never the image bytes. The frontend resolves it
                 against the CloudFront base in src/lib/assetUrl.ts.
                 Despite the name it holds any asset key, not only pictures:
                 project demo clips are .mp4 keys, and the CV is a .pdf key
                 (section "personal_statement", description "resume").
    created_at   defaults to now(); record metadata only. Reads take the row
                 with the highest id for a given (section, description), so
                 old rows stay as restorable history, exactly like
                 site_content. Never UPDATE a row; INSERT a new one.

    See also persona_stand_ec2yml/Part_D.md (D.2 seed SQL, D.5 shapes).
    """
    __tablename__ = "site_image"
    __table_args__ = (
        # (section, description, id DESC) -- the slot, then id: exactly the
        # order the "current version" query reads -- one index scan, no sort. The current version is the highest id,
        # not the newest created_at: these tables are single-owner, low-write
        # and append-only, so id order IS write order on the application's
        # write path, and created_at is kept as record metadata only.
        # EXISTING DATABASES need this by hand -- create_all never adds an
        # index to a table it did not create. See persona_stand_ec2yml/Part_C.md.
        Index("ix_site_image_section_description_id_desc", "section", "description", desc("id")),
    )

    id = Column(Integer, primary_key=True)
    section = Column(String, nullable=False)
    description = Column(String, nullable=False)
    image_path = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
