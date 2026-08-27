import logging

from fastapi import Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import consent as consent_models

logger = logging.getLogger(__name__)


class NoConsentPolicyConfiguredError(Exception):
    """Raised when consent_policy has no rows -- there's nothing yet to show the user or record consent against."""
    pass


class ConsentRequiredError(Exception):
    """Raised when a session hasn't consented to the current consent_policy version."""

    def __init__(self, session_id: str, policy_version: str | None):
        self.session_id = session_id
        self.policy_version = policy_version
        super().__init__(f"session {session_id} has not consented to policy version {policy_version}")


class ConsentService:
    """
    Records and checks whether a session has agreed to the current consent
    text before its messages are collected. Backed by two tables (see
    app/models/consent.py):
    - consent_policy: the actual wording, one row per version -- the
      frontend's popup pulls its text from here instead of hardcoding it,
      so wording can change without a redeploy.
    - consent_record: proof that a given session agreed to a given
      version, so agreement is provable after the fact, not just a claim
      the frontend made -- see PrivacyGateService/RateControlService for
      the same "gate before chat_service does anything with user_text"
      pattern this follows.
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

    def get_current_policy(self) -> consent_models.ConsentPolicy | None:
        """
        Fetches the current consent policy -- the most recently added consent_policy row.

        Parameters:
        - none

        Returns:
        - ConsentPolicy | None: the highest-id row, or None if consent_policy is empty (nothing configured yet)
        """
        return (
            self.db.query(consent_models.ConsentPolicy)
            .order_by(consent_models.ConsentPolicy.id.desc())
            .first()
        )

    def is_consented(self, session_id: str) -> bool:
        """
        Checks whether this session has already consented to the current policy version.

        Parameters:
        - session_id (str): the caller's session -- comes from the router (get_or_create_session_id)

        Returns:
        - bool: True if a consent_record row exists for (session_id, current policy's version); False if no policy is configured at all
        """
        policy = self.get_current_policy()
        if policy is None:
            return False
        return (
            self.db.query(consent_models.ConsentRecord)
            .filter(
                consent_models.ConsentRecord.session_id == session_id,
                consent_models.ConsentRecord.policy_version == policy.version
            )
            .first()
        ) is not None

    def check(self, session_id: str) -> None:
        """
        Guards a chat turn on prior consent.

        Parameters:
        - session_id (str): the caller's session -- comes from ChatService.handle_chat_turn, checked before anything else touches user_text

        Returns:
        - None: raises ConsentRequiredError if this session hasn't consented to the current policy version yet (including if no policy is configured at all), otherwise returns silently
        """
        if not self.is_consented(session_id):
            policy = self.get_current_policy()
            raise ConsentRequiredError(session_id, policy.version if policy else None)

    def record_consent(self, session_id: str) -> None:
        """
        Records that a session has agreed to the current policy version.

        Parameters:
        - session_id (str): the consenting session -- comes from the router (get_or_create_session_id)

        Returns:
        - None: inserts a consent_record row, or does nothing if this session already consented to the current version (idempotent). Raises NoConsentPolicyConfiguredError if consent_policy is empty.
        """
        policy = self.get_current_policy()
        if policy is None:
            raise NoConsentPolicyConfiguredError()
        if self.is_consented(session_id):
            return
        entry = consent_models.ConsentRecord(session_id=session_id, policy_version=policy.version)
        self.db.add(entry)
        self.db.commit()
