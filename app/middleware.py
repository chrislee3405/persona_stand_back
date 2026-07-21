from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

def setup_middleware(app: FastAPI) -> None:
    # Define the exact frontend URLs allowed to make API calls to this backend
    origins = [
        "http://localhost:5173",  # Local Vite React frontend
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,      # Allows cookies/session tracking if needed later
        allow_methods=["*"],         # Allows GET, POST, OPTIONS, etc.
        allow_headers=["*"],         # Allows headers like Content-Type
    )