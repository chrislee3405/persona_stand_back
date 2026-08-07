from fastapi import FastAPI
# #RUN THE DOCKER WEBSAITE IN LOCAL DEVICE
# .venv\Scripts\activate
# uvicorn app.main:app --reload

# #IF ACCIDENTIALLY RUN THE WRONG docker-compose.yml IN EC2 
# docker system prune -f

from app.middleware import setup_middleware
from app.routers import codes_router, dialogues_router

app = FastAPI(title="My Backend API")

setup_middleware(app)

app.include_router(dialogues_router.router, tags=["dialogues"])
app.include_router(codes_router.router, tags=["codes"])
