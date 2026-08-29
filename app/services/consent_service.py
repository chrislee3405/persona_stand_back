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


class ConsentTextMismatchError(Exception):
    """Raised when record_consent's submitted_text doesn't match the current policy's condition_text -- either a stale/wrong value, or a scripted call that never actually fetched the real text."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        super().__init__(f"session {session_id} submitted consent text that doesn't match the current policy")


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
      pattern this follows. record_consent requires the caller to submit
      the exact current policy text (not just a bare confirmation) and
      stores it on the row -- a scripted call has to actually fetch (GET
      /api/consent) and forward the real text rather than blindly
      POSTing, and the stored copy makes each record self-contained even
      if consent_policy is edited later.
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

    def record_consent(self, session_id: str, submitted_text: str) -> None:
        """
        Records that a session has agreed to the current policy version -- requires the caller to submit the exact current policy text, not a bare confirmation.

        Parameters:
        - session_id (str): the consenting session -- comes from the router (get_or_create_session_id)
        - submitted_text (str): the condition text the client claims to be agreeing to -- comes from the router's request body (populated from GET /api/consent's conditionText in the real popup flow). Must match the current policy's condition_text (compared with leading/trailing whitespace ignored) or the call is rejected -- this is what makes a scripted bypass need to actually fetch and forward the real text rather than blindly POSTing.

        Returns:
        - None: raises NoConsentPolicyConfiguredError if consent_policy is empty, ConsentTextMismatchError if submitted_text doesn't match the current policy's condition_text. Otherwise inserts a consent_record row (storing submitted_text on it), or does nothing if this session already consented to the current version (idempotent).
        """
        policy = self.get_current_policy()
        if policy is None:
            raise NoConsentPolicyConfiguredError()
        if submitted_text.strip() != policy.condition_text.strip():
            logger.info("rejected consent for session=%s: submitted text didn't match the current policy", session_id)
            raise ConsentTextMismatchError(session_id)
        if self.is_consented(session_id):
            return
        entry = consent_models.ConsentRecord(session_id=session_id, policy_version=policy.version, condition_text=submitted_text)
        self.db.add(entry)
        self.db.commit()
