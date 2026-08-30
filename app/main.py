import logging
import os

from fastapi import FastAPI

from app.middleware import setup_middleware
from app.routers import codes_router, conversations_router, consent_router, site_content_router
from app.database import engine, Base, SessionLocal
from app.models.consent import ConsentPolicy
from app.models.site_content import SiteContent  # noqa: F401  -- registers table for create_all


logging.basicConfig(
    level=logging.DEBUG if os.environ.get("ENV", "development") != "production" else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


Base.metadata.create_all(bind=engine) # create tables in models folder if they don't exist in db

# Seed an initial consent policy if none exists yet -- otherwise
# consent_policy starts empty and ConsentService.check would block every
# session with no way to actually agree to anything. Placeholder wording;
# add a new row (don't edit this one) with real text and a new version
# once it's ready -- see ConsentPolicy/ConsentService.get_current_policy.
with SessionLocal() as _seed_db:
    if _seed_db.query(ConsentPolicy).first() is None:
        _seed_db.add(ConsentPolicy(
            version="v1-placeholder",
            condition_text=(
                "This chatroom collects and stores the messages you send so the "
                "conversation can function and be reviewed. By clicking \"I Agree\", "
                "you consent to your messages being collected for this purpose."
            )
        ))
        _seed_db.commit()

app = FastAPI(title="My Backend API")



setup_middleware(app)

app.include_router(conversations_router.router, tags=["conversations"])
app.include_router(codes_router.router, tags=["codes"])
app.include_router(consent_router.router, tags=["consent"])
app.include_router(site_content_router.router, tags=["site-content"])
