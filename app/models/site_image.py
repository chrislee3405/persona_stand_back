from sqlalchemy import Column, Integer, String, DateTime, Index
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
    created_at   defaults to now(). Reads take the newest row for a given
                 (section, description) -- newest created_at, then highest
                 id -- so old rows stay as restorable history, exactly like
                 site_content. Never UPDATE a row; INSERT a new one.

    See also persona_stand_ec2yml/Part_D.md (D.2 seed SQL, D.5 shapes).
    """
    __tablename__ = "site_image"
    __table_args__ = (
        # Serves the only query this table has: "newest row for this
        # (section, description)" and "newest row per slot for a section".
        Index("ix_site_image_section_description_created_at", "section", "description", "created_at"),
    )

    id = Column(Integer, primary_key=True, index=True)
    section = Column(String, nullable=False)
    description = Column(String, nullable=False)
    image_path = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
