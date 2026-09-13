from sqlalchemy import Column, Date, DateTime, Integer, String, UniqueConstraint
from sqlalchemy.sql import func

from app.database import Base


class RateLimitCounter(Base):
    """
    One row per (rate-limit key, calendar day), holding that day's count.

    Replaces the in-process dictionaries RateControlService used to keep
    (`_daily_counts`, `_ip_daily_counts`). Those had three problems this table
    fixes:

    - They were UNBOUNDED. `key` includes session ids, and a caller mints a
      fresh session id by simply dropping its cookie, so every request from a
      cookie-less client cost a permanent dict entry until midnight.
    - They did not survive a restart, so a redeploy handed everybody a fresh
      day's allowance -- and with `uvicorn --reload` previously in the
      production CMD, so did any write into /app.
    - They were per-process, so the counters silently stopped meaning anything
      the moment the app ran more than one worker or more than one replica.

    Counts are incremented with a single atomic
    `INSERT ... ON CONFLICT DO UPDATE SET count = count + 1 RETURNING count`,
    so two concurrent requests for the same key cannot both read the old value
    and write the same new one. That statement is the whole concurrency story;
    there is no read-then-write anywhere in RateControlService.

    Rows are KEPT after their day rolls over, as rate-limit history for the
    owner to review -- which is why this is a table of days rather than a
    single mutable counter per key. `RATE_LIMIT_RETENTION_DAYS` bounds that
    history; RateControlService sweeps anything older on the first request of
    each new day.

    Columns
    -------
    id       serial PK.
    key      What is being counted, as `<scope>:<value>`. The scopes in use:
               session:<session_id>       per-session chat turns
               ip:<client_ip>             per-IP chat turns (guest tier only)
               code_fail_ip:<client_ip>   failed invite-code verifications, per IP
               code_fail:global           failed invite-code verifications, all IPs
               consent_ip:<client_ip>     POST /api/consent submissions, per IP
             Prefixed rather than split into a `scope` column so one uniqueness
             constraint and one index serve every counter, and so adding a
             scope needs no migration.
    day      The calendar day the count belongs to (server local date, same
             rule the in-memory version used -- a portfolio's visitors are not
             synchronised to any timezone and it only has to reset once a day).
    count    How many units of this key were consumed that day.
    """

    __tablename__ = "rate_limit_counter"
    __table_args__ = (
        # The uniqueness that makes ON CONFLICT work at all -- the increment
        # statement names this constraint's columns as its conflict target.
        UniqueConstraint("key", "day", name="uq_rate_limit_key_day"),
    )

    id = Column(Integer, primary_key=True)
    key = Column(String(256), nullable=False)
    day = Column(Date, nullable=False, index=True)  # indexed for the retention sweep
    count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
