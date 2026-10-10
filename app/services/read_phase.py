"""Explicit boundary for read phases using the request's expire_on_commit=False session."""


async def finish_read_phase(db):
    # These phases own reads only. Refuse to accidentally flush unrelated writes.
    # Commit ends the read transaction without expiring the history/user ORM rows
    # that the rest of the request still needs (rollback would expire them).
    if db.new or db.dirty or db.deleted:
        raise RuntimeError("Unexpected pending writes at read-phase boundary")
    await db.commit()
