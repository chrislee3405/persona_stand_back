from fastapi import FastAPI
# #RUN THE APP LOCALALLY WITHOUT DOCKER
# .venv\Scripts\activate
# uvicorn app.main:app --reload

# # UPDATE THE APP VERSION IN DOCKER
# docker compose -f docker-compose.yml up --build


# #IF ACCIDENTIALLY RUN THE WRONG docker-compose.yml IN EC2 
# docker system prune -f

from app.middleware import setup_middleware
from app.routers import codes_router, dialogues_router
from app.database import engine, Base

Base.metadata.create_all(bind=engine)

app = FastAPI(title="My Backend API")



setup_middleware(app)

app.include_router(dialogues_router.router, tags=["dialogues"])
app.include_router(codes_router.router, tags=["codes"])
