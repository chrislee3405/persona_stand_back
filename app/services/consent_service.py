import logging
from typing import Any, TypedDict

from fastapi import Depends
from sqlalchemy import select, update
from sqlalchemy.sql import func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import consent as consent_models

logger = logging.getLogger(__name__)


class ConsentTerms(TypedDict):
    """
    The normalised shape of a consent policy's terms.

    `header` is a one-line purpose statement shown above the box; it may be
    empty, and the popup then names itself by the terms instead. `condition`
    is the detailed wording actually being agreed to, and is what the client
    echoes back on submission.
    """

    header: str
    condition: str


def normalise_terms(raw: Any) -> ConsentTerms | None:
    """
    Reads a consent_policy.condition_text value in either of its stored forms and returns it as {header, condition}.

    Parameters:
    - raw (Any): the JSONB value straight off the row -- comes from ConsentService

    Returns:
    - ConsentTerms | None: the normalised terms, or None when the value cannot
      be used at all (not a string or object, or a blank `condition`). None is
      a real, expected outcome and every caller treats it the same as "no
      policy configured": the frontend shows its inert "terms unavailable"
      card and the chat gate stays closed.

    TWO FORMS, on purpose. The column used to be plain text, so rows written
    before the JSONB conversion hold a bare string; those read as
    {"header": "", "condition": <the old text>}. Rows written since hold
    {"header"?, "condition"}. Accepting both means the code is correct before,
    during and after the conversion runs, rather than only on one side of it.

    NOTHING IS INVENTED HERE. A row whose `condition` is missing or blank
    yields None rather than a default, for the same reason app/main.py no
    longer seeds a placeholder policy at startup: a consent notice the
    application made up is not a consent notice, and one that silently becomes
    the live legal text is worse than none at all.
    """
    if isinstance(raw, str):
        condition = raw.strip()
        return {"header": "", "condition": condition} if condition else None

    if isinstance(raw, dict):
        condition = raw.get("condition")
        if not isinstance(condition, str) or not condition.strip():
            return None
        header = raw.get("header")
        return {
            "header": header.strip() if isinstance(header, str) else "",
            "condition": condition.strip(),
        }

    return None


class NoConsentPolicyConfiguredError(Exception):
    """
    Raised when there is no usable consent policy to show or record against.

    Covers both "consent_policy has no rows" and "the current (highest-id) row's terms are
    unusable" (see normalise_terms) -- the two are the same fact from the
    visitor's side, and the same fact from the gate's side.
    """
    pass


class ConsentRequiredError(Exception):
    """Raised when a session hasn't consented to the current consent_policy version."""

    def __init__(self, session_id: str, policy_version: str | None):
        self.session_id = session_id
        self.policy_version = policy_version
        super().__init__(f"session {session_id} has not consented to policy version {policy_version}")


class ConsentTextMismatchError(Exception):
    """Raised when record_consent's submitted condition doesn't match the current policy's -- either a stale/wrong value, or a scripted call that never actually fetched the real terms."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        super().__init__(f"session {session_id} submitted consent text that doesn't match the current policy")


class ConsentService:
    """
    Records and checks whether a session has agreed to the current consent
    terms before its messages are collected. Backed by two tables (see
    app/models/consent.py):

    - consent_policy: the actual wording, one row per version, stored as
      JSONB {header?, condition} -- the frontend's popup pulls it from here
      instead of hardcoding it, so wording can change without a redeploy.
      SEEDED AS DATA, not by application code: see app/models/seed/. With no
      row at all -- or a row whose terms are unusable -- GET /api/chatroom_initialize
      reports the terms as unavailable and check() refuses every chat turn,
      which is the correct fail-closed behaviour for a consent gate and is
      now a state the app is designed to have rather than one it papers over.
    - consent_record: proof that a given session agreed to a given version,
      so agreement is provable after the fact rather than being a claim the
      frontend made. record_consent requires the caller to submit the exact
      current `condition` text and stores the whole normalised terms object
      on the row, which is what makes each record self-contained evidence of
      what was agreed to even if consent_policy is edited later.
    """

    def __init__(self, db: AsyncSession = Depends(get_db)):
        """
        Stores the injected database session.

        Parameters:
        - db (AsyncSession): SQLAlchemy async session -- injected by FastAPI via get_db

        Returns:
        - None: sets self.db and the per-request policy cache
        """
        self.db = db
        # Per-request memo. check() used to reach get_current_policy three
        # times for one boolean -- once directly and twice inside
        # is_consented -- which is three round trips on the hot path of every
        # chat turn. Scoped to this instance, and FastAPI builds a fresh one
        # per request, so it can never serve a stale policy across requests.
        self._policy_cache: consent_models.ConsentPolicy | None = None
        self._policy_cached = False

    async def get_current_policy(self) -> consent_models.ConsentPolicy | None:
        """
        Fetches the current consent policy -- the most recently added consent_policy row.

        Parameters:
        - none

        Returns:
        - ConsentPolicy | None: the highest-id row, or None if consent_policy is empty (nothing configured yet). Memoised for the life of this request.
          Note this returns the ROW, whose terms may still be unusable -- use
          get_current_terms when what you want is displayable wording.
        """
        if self._policy_cached:
            return self._policy_cache

        result = await self.db.execute(
            select(consent_models.ConsentPolicy).order_by(consent_models.ConsentPolicy.id.desc()).limit(1)
        )
        self._policy_cache = result.scalar_one_or_none()
        self._policy_cached = True
        return self._policy_cache

    async def get_current_terms(self) -> tuple[str, ConsentTerms] | None:
        """
        Fetches the current policy's version and displayable terms together.

        Parameters:
        - none

        Returns:
        - tuple[str, ConsentTerms] | None: (version, {header, condition}), or None
          when there is no policy row OR the current (highest-id) row's terms cannot be used.
          The two collapse into one None deliberately: both mean "there is
          nothing a visitor could read and agree to", which is the only
          distinction the popup and the chat gate care about.
        """
        policy = await self.get_current_policy()
        if policy is None:
            return None
        terms = normalise_terms(policy.condition_text)
        if terms is None:
            # A row exists but is unusable -- a malformed edit, or a partially
            # converted legacy row. Worth a log line because, unlike an empty
            # table, this one is a mistake somebody made and can fix.
            logger.error(
                "consent_policy version %s has unusable condition_text -- treating the terms as unavailable",
                policy.version,
            )
            return None
        return policy.version, terms

    async def is_consented(self, session_id: str) -> bool:
        """
        Checks whether this session has already consented to the current policy version.

        Parameters:
        - session_id (str): the caller's session -- comes from the router (get_or_create_session_id)

        Returns:
        - bool: True if an ACTIVE consent_record row exists for (session_id, current policy's version) -- one that has not been withdrawn; False if no usable policy is configured at all
        """
        current = await self.get_current_terms()
        if current is None:
            return False
        version, _terms = current
        result = await self.db.execute(
            select(consent_models.ConsentRecord.id).where(
                consent_models.ConsentRecord.session_id == session_id,
                consent_models.ConsentRecord.policy_version == version,
                # A withdrawn agreement is history, not consent.
                consent_models.ConsentRecord.withdrawn_at.is_(None),
            ).limit(1)
        )
        return result.scalar_one_or_none() is not None

    async def check(self, session_id: str) -> None:
        """
        Guards a chat turn on prior consent.

        Parameters:
        - session_id (str): the caller's session -- comes from ChatService.handle_chat_turn, checked before anything else touches user_text

        Returns:
        - None: raises ConsentRequiredError if this session hasn't consented to the current policy version yet -- INCLUDING when no usable policy is configured at all, because a session cannot have agreed to terms that do not exist. Otherwise returns silently.
        """
        if not await self.is_consented(session_id):
            current = await self.get_current_terms()
            raise ConsentRequiredError(session_id, current[0] if current else None)

    async def record_consent(self, session_id: str, submitted_condition: str) -> str:
        """
        Records that a session has agreed to the current policy version -- requires the caller to submit the exact current condition text, not a bare confirmation.

        Parameters:
        - session_id (str): the consenting session -- comes from the router (get_or_create_session_id)
        - submitted_condition (str): the `condition` the client claims to be agreeing to -- comes from the router's request body (populated from GET /api/chatroom_initialize's consent.conditionTerms.condition in the real popup flow). Must match the current policy's `condition` (compared with leading/trailing whitespace ignored) or the call is rejected.

        Returns:
        - str: the policy version consented to -- goes back to the router for the response body

        Raises:
        - NoConsentPolicyConfiguredError: there is no usable policy, so there is nothing to agree to. The submission is refused rather than recorded: a consent_record whose policy_version points at nothing would look like proof and be none.
        - ConsentTextMismatchError: submitted_condition doesn't match the current policy's condition.

        The WHOLE normalised terms object is stored on the record, not just
        the condition the client sent, so the row carries the header the
        visitor actually saw above the box as well as the wording itself.
        """
        current = await self.get_current_terms()
        if current is None:
            raise NoConsentPolicyConfiguredError()
        version, terms = current

        if submitted_condition.strip() != terms["condition"]:
            logger.info("rejected consent for session=%s: submitted text didn't match the current policy", session_id)
            raise ConsentTextMismatchError(session_id)

        if await self.is_consented(session_id):
            return version

        entry = consent_models.ConsentRecord(
            session_id=session_id,
            policy_version=version,
            condition_text=dict(terms),
        )
        self.db.add(entry)
        await self.db.commit()
        return version

    async def withdraw_consent(self, session_id: str) -> int:
        """
        Withdraws every agreement this session currently has in force.

        Parameters:
        - session_id (str): the withdrawing session -- comes from consent_router.withdraw_consent (get_or_create_session_id)

        Returns:
        - int: how many active records were withdrawn -- goes back to the router. 0 is not
          an error: withdrawing when nothing is in force is simply already the case, and
          the endpoint stays idempotent.

        MARKED, NOT DELETED. Each active row gets `withdrawn_at` set and stays
        in the table. The agreement was really given, and whatever was
        collected while it was in force was collected under it -- deleting the
        row would destroy the only proof of that. The same row now also proves
        when consent ended.

        From the moment this commits, ConsentService.check refuses the
        session's chat turns with 403, so nothing further is processed or
        stored until it agrees again -- which inserts a NEW row (see the
        partial unique index on ConsentRecord), leaving this withdrawal on
        record rather than overwriting it.

        Every policy version is withdrawn, not only the current one: a session
        holding an older agreement has nothing in force under the current
        policy anyway, and leaving stale rows active would let a policy
        rollback quietly revive them.
        """
        result = await self.db.execute(
            update(consent_models.ConsentRecord)
            .where(
                consent_models.ConsentRecord.session_id == session_id,
                consent_models.ConsentRecord.withdrawn_at.is_(None),
            )
            .values(withdrawn_at=func.now())
        )
        await self.db.commit()
        withdrawn = result.rowcount or 0
        logger.info("session=%s withdrew consent (%d active record(s))", session_id, withdrawn)
        return withdrawn

    async def transfer_records(self, old_session_id: str, new_session_id: str) -> int:
        """
        Re-points every consent record held by one session id at another.

        Parameters:
        - old_session_id (str): the session id being retired -- comes from codes_router's session rotation
        - new_session_id (str): the freshly-minted session id -- comes from codes_router

        Returns:
        - int: how many records moved -- goes to codes_router for logging

        Does NOT commit: one step of the rotation transaction, alongside
        ConversationService.transfer_ownership. Rotating the session id
        without bringing the consent record along would leave the visitor
        consented under an id their browser no longer holds, so their very
        next message would be refused with 403 and the popup would reappear
        on a session that had already agreed.

        Withdrawn records move too, so the new id carries the session's full
        consent history, not just what is in force. The partial unique index
        on active (session_id, policy_version) cannot collide here: the new id
        is a freshly generated UUID that has never been seen before.
        """
        result = await self.db.execute(
            select(consent_models.ConsentRecord).where(
                consent_models.ConsentRecord.session_id == old_session_id
            )
        )
        records = list(result.scalars().all())
        for record in records:
            record.session_id = new_session_id
        return len(records)
