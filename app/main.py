from fastapi import FastAPI, File, UploadFile, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from PIL import Image
import io
import time
import logging
import os

from app.model import load_model, pick_device, build_preprocess, predict_image

app = FastAPI(title="Cats vs Dogs Inference Service")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Instrumentator().instrument(app).expose(app)

MODEL_PATH = os.getenv("MODEL_PATH", "models/cats_dogs_model.pt")
DEVICE = pick_device(os.getenv("DEVICE", "cpu"))

_model = None
_preprocess = None
_class_names = None
_device = None


@app.on_event("startup")
def startup():
    global _model, _preprocess, _class_names, _device, MODEL_PATH, DEVICE

    _device = DEVICE
    try:
        model, meta = load_model(MODEL_PATH, _device)

        # meta should contain class_names and img_size
        _model = model
        _class_names = meta.get("class_names", ["Cat", "Dog"])
        img_size = int(meta.get("img_size", 224))

        _preprocess = build_preprocess(img_size)

        logger.info(
            f"✅ Model loaded at startup. path={MODEL_PATH} device={_device} classes={_class_names} img_size={img_size}"
        )
    except Exception as e:
        _model = None
        _preprocess = None
        _class_names = None
        logger.warning(f"Model not loaded at startup: {e}")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "cats-vs-dogs-classifier",
        "model_loaded": _model is not None,
        "model_path": MODEL_PATH,
        "device": str(_device) if _device is not None else None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    logger.info(f"Received prediction request for file: {file.filename}")

    if _model is None or _preprocess is None or _class_names is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Train and place model at '{MODEL_PATH}' (or set MODEL_PATH env var), then restart service."
        )

    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file. Please upload a valid JPG/PNG image.")

    out = predict_image(_model, {"class_names": _class_names, "img_size": 224}, image, device=_device)
    pred = out["predicted_class"]
    probs = out["probabilities"]
    prob = max(probs.values())

    latency = time.time() - start_time
    logger.info(f"Prediction: {pred} ({prob:.4f}) - Latency: {latency:.4f}s")

    return {
        "filename": file.filename,
        "prediction": pred,
        "probability": float(f"{prob:.4f}"),
        "class_probabilities": {k: float(f"{v:.4f}") for k, v in probs.items()},
        "latency_seconds": float(f"{latency:.4f}")
    }