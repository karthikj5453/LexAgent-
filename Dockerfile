# Stage 1: Build Frontend React App
FROM node:20-alpine as frontend-builder
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Build Python Backend dependencies
FROM python:3.11-slim as backend-builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 3: Final Runtime image
FROM python:3.11-slim
WORKDIR /app

# Copy installed python packages from backend-builder
COPY --from=backend-builder /root/.local /root/.local

# Copy frontend static build assets from frontend-builder
COPY --from=frontend-builder /frontend/dist ./frontend/dist

# Copy backend files
COPY . .

# Ensure paths and environment are set correctly
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PORT=8000

# Expose port
EXPOSE 8000

# Run the FastAPI application (serving both REST endpoints and the UI)
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
