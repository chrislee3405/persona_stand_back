from fastapi import FastAPI

app = FastAPI(title="My Backend API")

@app.get("/")
def read_root():
    return {"status": "healthy", "message": "FastAPI backend is running!"}