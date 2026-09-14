from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Index, text
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

    id = Column(Integer, primary_key=True)
    version = Column(String, nullable=False, unique=True)
    condition_text = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ConsentRecord(Base):
    __tablename__ = "consent_record"
    __table_args__ = (
        # At most ONE ACTIVE agreement per (session, policy version) -- a
        # PARTIAL unique index, not a plain unique constraint.
        #
        # A session can agree, withdraw, and agree again (see
        # ConsentService.withdraw_consent). With a plain (session_id,
        # policy_version) constraint the second agreement could only UPDATE
        # the first row, overwriting the fact that consent was ever withdrawn
        # -- and a withdrawal is exactly the kind of event that needs to stay
        # provable. So a withdrawn row keeps its `withdrawn_at` and a new
        # agreement is a NEW row; the index only forbids two live rows at once.
        #
        # EXISTING DATABASES need this by hand: create_all never alters a
        # table it did not create. See persona_stand_ec2yml/Part_C.md.
        Index(
            "uq_consent_record_active",
            "session_id",
            "policy_version",
            unique=True,
            postgresql_where=text("withdrawn_at IS NULL"),
        ),
    )

    id = Column(Integer, primary_key=True)
    session_id = Column(String, nullable=False, index=True)
    policy_version = Column(String, ForeignKey("consent_policy.version"), nullable=False)
    # The {header, condition} terms the client submitted (validated to match
    # ConsentPolicy.condition_text at the time -- see
    # ConsentService.record_consent) captured onto the record itself, so
    # this row is self-contained proof of what was agreed to even if the
    # consent_policy row it points at is ever edited afterward.
    condition_text = Column(JSONB, nullable=False)
    consented_at = Column(DateTime(timezone=True), server_default=func.now())
    # Set when the session withdraws this agreement ("Disagree with consent"
    # under the chatroom). NULL means the agreement is in force. The row is
    # never deleted: that consent was given, and messages collected under it
    # were collected lawfully, so the record of it -- and of its withdrawal --
    # outlives the withdrawal itself.
    withdrawn_at = Column(DateTime(timezone=True), nullable=True)
