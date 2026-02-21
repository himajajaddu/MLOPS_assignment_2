from fastapi import FastAPI, File, UploadFile
from prometheus_fastapi_instrumentator import Instrumentator
import random
import time
import logging

# M2: Inference Service wrapped with FastAPI
app = FastAPI(title="Cats vs Dogs Inference Service")

# M5: Basic Monitoring & Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# M5: Prometheus metrics tracking for request count and latency
Instrumentator().instrument(app).expose(app)

# M2: Endpoint 1 - Health Check
@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "cats-vs-dogs-classifier"}

# M2: Endpoint 2 - Prediction
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    logger.info(f"Received prediction request for file: {file.filename}")
    
    # Dummy prediction logic (Replace with actual model inference steps)
    # 1. Load image using PIL
    # 2. Resize to 224x224 RGB
    # 3. Convert to tensor and run through model
    
    class_name = random.choice(["Cat", "Dog"])
    probability = random.uniform(0.75, 0.99)
    
    latency = time.time() - start_time
    # M5: Track latency and logs
    logger.info(f"Prediction: {class_name} ({probability:.4f}) - Latency: {latency:.4f}s")
    
    return {
        "filename": file.filename,
        "prediction": class_name,
        "probability": float(f"{probability:.4f}"),
        "latency_seconds": float(f"{latency:.4f}")
    }
