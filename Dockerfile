# 1. Official lightweight Python base image
FROM python:3.11-slim

# 2. Set working directory inside container
WORKDIR /app

# 3. Environment flags to keep Python output fast and clean
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 4. Copy requirements first & install (optimizes Docker build caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your app source code
COPY . .

# 6. Document the port FastAPI listens on
EXPOSE 8000

# 7. Start Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]