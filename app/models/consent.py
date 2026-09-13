from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from app.database import Base


class ConsentPolicy(Base):
    """
    One row per version of the consent terms -- what ConsentRecord rows are
    agreeing to. Add a new row (with a new, higher `version`) to change the
    wording; don't edit an existing row's condition_text in place, or past
    ConsentRecord rows silently stop meaning what they originally recorded.
    ConsentService treats the highest-id row as "the current policy" that
    the popup shows and new consents are recorded against.

    condition_text is JSONB with this shape (`?` = optional):

        {
          "header":    "<string>",  ?  # one-line purpose statement shown
                                       #   above the terms, e.g. "How your
                                       #   chat messages are handled"
          "condition": "<string>"      # the detailed terms, shown in a box
                                       #   under the header (required).
                                       #   Rendered by <Prose>: blank lines ->
                                       #   paragraphs, "- " lines -> bullets.
        }

    The column used to be plain text. Rows from then are converted to
    {"header": "", "condition": <old text>}, and ConsentService's
    normalise_terms reads either form, so the code works before and after
    that conversion runs.
    """
    __tablename__ = "consent_policy"

    id = Column(Integer, primary_key=True, index=True)
    version = Column(String, nullable=False, unique=True)
    condition_text = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ConsentRecord(Base):
    __tablename__ = "consent_record"
    __table_args__ = (
        # A session can end up consenting again after the current policy
        # version changes (see ConsentService.get_current_policy) -- one
        # row per (session, version) rather than one row per session overall.
        UniqueConstraint("session_id", "policy_version", name="uq_session_policy_version"),
    )

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, nullable=False, index=True)
    policy_version = Column(String, ForeignKey("consent_policy.version"), nullable=False)
    # The {header, condition} terms the client submitted (validated to match
    # ConsentPolicy.condition_text at the time -- see
    # ConsentService.record_consent) captured onto the record itself, so
    # this row is self-contained proof of what was agreed to even if the
    # consent_policy row it points at is ever edited afterward.
    condition_text = Column(JSONB, nullable=False)
    consented_at = Column(DateTime(timezone=True), server_default=func.now())
