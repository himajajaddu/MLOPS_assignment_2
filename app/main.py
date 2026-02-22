from fastapi import FastAPI, File, UploadFile, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from PIL import Image
import io
import time
import logging
import os

from app.model import load_model, pick_device, predict_image

app = FastAPI(title="Cats vs Dogs Inference Service")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Instrumentator().instrument(app).expose(app)

MODEL_PATH = os.getenv("MODEL_PATH", "models/cats_dogs_model.pt")
DEVICE_STR = os.getenv("DEVICE", "")  # cpu / mps / cuda / empty(auto)

# Globals used by endpoints
model = None
meta = None
device = None


@app.on_event("startup")
def startup():
    global model, meta, device
    try:
        device = pick_device(DEVICE_STR)
        logger.info(f"Loading model from: {os.path.abspath(MODEL_PATH)} on device={device}")
        model, meta = load_model(MODEL_PATH, device)
        logger.info(f"Model loaded ✅ classes={meta.get('class_names')} img_size={meta.get('img_size')}")
    except Exception as e:
        logger.exception(f"Model not loaded at startup: {e}")
        model, meta, device = None, None, None


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "cats-vs-dogs-classifier",
        "model_loaded": model is not None,
        "model_path": os.path.abspath(MODEL_PATH),
        "device": str(device) if device else None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    logger.info(f"Received prediction request for file: {file.filename}")

    if model is None or meta is None or device is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Train and place model at '{os.path.abspath(MODEL_PATH)}' "
                   f"(or set MODEL_PATH env var), then restart service."
        )

    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file. Please upload a valid JPG/PNG image.")

    result = predict_image(model, meta, image, device)

    latency = time.time() - start_time
    logger.info(f"Prediction: {result['predicted_class']} - Latency: {latency:.4f}s")

    return {
        "filename": file.filename,
        "prediction": result["predicted_class"],
        "class_probabilities": {k: float(f"{v:.4f}") for k, v in result["probabilities"].items()},
        "latency_seconds": float(f"{latency:.4f}"),
    }