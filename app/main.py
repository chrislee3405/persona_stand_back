import logging
import os

from fastapi import FastAPI

from app.middleware import setup_middleware
from app.routers import codes_router, conversations_router
from app.database import engine, Base


logging.basicConfig(
    level=logging.DEBUG if os.environ.get("ENV", "development") != "production" else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


Base.metadata.create_all(bind=engine) # create tables in models folder if they don't exist in db

app = FastAPI(title="My Backend API")



setup_middleware(app)

app.include_router(conversations_router.router, tags=["conversations"])
app.include_router(codes_router.router, tags=["codes"])
