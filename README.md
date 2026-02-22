# MLOps Assignment 2: End-to-End MLOps Pipeline (Cats vs Dogs)

This project implements an end-to-end MLOps workflow for **Cats vs Dogs** image classification:

- **M1**: Real model training + MLflow experiment tracking
- **M2**: FastAPI inference service (loads the trained model and predicts from an uploaded image)
- **M3**: Automated tests with Pytest
- **M4**: Containerization + Docker Compose deployment
- **M5**: Basic monitoring (Prometheus metrics) + logging

---

## Folder Structure (important)

- `data/raw/` → raw Kaggle download/extract (DO NOT commit large files)
- `data/processed/` → ImageFolder structured dataset (optional helper script)
- `models/` → trained model artifact saved as `cats_dogs_model.pt`
- `src/train.py` → real training script
- `app/main.py` → FastAPI inference service (uses the trained model)

---

## M1: Data + Training + MLflow

### 1) Put Kaggle dataset in `data/raw/`

Use **any** Kaggle Cats vs Dogs dataset, but the code supports these common layouts:

**Layout A (already ImageFolder style)**:
```
data/raw/train/cats/*.jpg
data/raw/train/dogs/*.jpg
```

**Layout B (Kaggle Dogs vs Cats competition)**:
```
data/raw/train/cat.0.jpg, dog.0.jpg, ...
data/raw/test/...
```

If your dataset is Layout B, run the converter:

```bash
python -m src.prepare_data --raw_dir data/raw --out_dir data/processed --subset 2500
```

Then you will train from `data/processed/`.

> Tip: start with `--subset 2500` for faster training, then increase later.

### 2) Track data with DVC (optional but recommended)

```bash
dvc init
dvc add data/raw
git add data/raw.dvc .gitignore
git commit -m "Track raw data with DVC"
```

### 3) Install requirements

```bash
pip install -r requirements.txt
```

### 4) Train (REAL training)

If your data is in `data/raw/` (Layout A):
```bash
python -m src.train --data_dir data/raw --epochs 3 --img_size 128
```

If you used the converter (Layout B → processed):
```bash
python -m src.train --data_dir data/processed --epochs 3 --img_size 128
```

This will:
- train a small CNN
- log params/metrics to **MLflow**
- save the best model to: `models/cats_dogs_model.pt`

### 5) View MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open: http://127.0.0.1:5000

---

## M2: FastAPI Inference (uses trained model)

### 1) Run locally

Train first, then:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Health check:
```bash
curl http://localhost:8000/health
```

Predict (upload any jpg/png):
```bash
curl -X POST -F "file=@path/to/sample.jpg" http://localhost:8000/predict
```

### 2) Model path via environment variable (optional)

By default, API loads:
- `models/cats_dogs_model.pt`

Override:
```bash
MODEL_PATH=models/cats_dogs_model.pt DEVICE=cpu uvicorn app.main:app --reload
```

---

## M2: Docker build/run

Build:
```bash
docker build -t cats-dogs-classifier:latest .
```

Run:
```bash
docker run -p 8000:8000 -e MODEL_PATH=models/cats_dogs_model.pt cats-dogs-classifier:latest
```

> NOTE: Your trained model must be inside the image. The simplest approach is: train first, then build the image (so `models/` is included).

---

## M3: Tests

```bash
pytest app/tests/
```

Tests are designed to pass even if the model is not trained yet (the API returns 503 until model exists).

---

## M4: Deployment (Docker Compose)

```bash
cd deployment
docker-compose up -d
```

---

## M5: Monitoring

Prometheus metrics:
- http://localhost:8000/metrics

Logs:
- standard Python logging in `app/main.py`

---

## Submission Tip

Zip the project **without large data**:

```bash
zip -r mlops_assignment_submission.zip MLOPS_assignment_2/ \
  -x "*/__pycache__/*" -x "*/data/raw/*" -x "*/data/processed/*"
```
