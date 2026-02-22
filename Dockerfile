# M2: Containerization
FROM python:3.10-slim

WORKDIR /app

# Install dependencies with version pinning via requirements.txt
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir --default-timeout=600 --retries 10 -r requirements.txt

# Copy source code into container
COPY . .
COPY models/ ./models/

EXPOSE 8000

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
