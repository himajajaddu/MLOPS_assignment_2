# 🐱🐶 MLOps Pipeline: Cats vs Dogs Image Classification

An end-to-end **MLOps pipeline** for binary image classification (Cats vs Dogs) designed for a pet adoption platform. The project demonstrates model training, experiment tracking, data versioning, containerization, CI/CD automation, deployment, and monitoring.

---

## 📌 Project Objective

Build and deploy a reproducible ML system that:

- Trains a CNN model for image classification
- Tracks experiments & metrics
- Versions datasets & artifacts
- Packages model into a REST API
- Automates build & deployment
- Monitors predictions & service health

---

## 🧩 Dataset

**Source:** Kaggle Cats vs Dogs Dataset  

**Preprocessing:**

- Resize images to RGB format  
- Split into Train / Validation / Test  
- Apply data augmentation for better generalization  

---

## 🏗️ MLOps Architecture

| Module | Purpose | Tools Used |
|--------|--------|------------|
| **M1** Model Development & Tracking | Train & track experiments | PyTorch, MLflow |
| **M2** Packaging & Containerization | REST API service | FastAPI, Docker |
| **M3** Continuous Integration | Automated testing & build | GitHub Actions, Pytest |
| **M4** Continuous Deployment | Service deployment | Docker Compose |
| **M5** Monitoring & Logging | Metrics & logs | Prometheus metrics + logging |

---

## 📁 Project Structure

```
├── .dvc/                   # Versioning
├── .github/                # github worflows of CI/CD
├── app/                    # FastAPI inference service
├── data/                   # Dataset (DVC tracked)
├── deployment/             # Docker Compose deployment
├── models/                 # Saved trained model
├── reports/                # Plots & evaluation outputs
├── src/                    # Training & preprocessing code
├── .github/workflows/      # CI pipeline
├── Dockerfile              # Container configuration
├── requirements.txt        # Dependencies
└── README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repository

```bash
git clone https://github.com/himajajaddu/MLOPS_assignment_2.git
cd MLOPS_assignment_2
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate     # Mac/Linux
# .venv\Scripts\activate      # Windows
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Dataset Versioning with DVC

Initialize DVC:

```bash
dvc init
```

Track dataset:

```bash
dvc add data/raw
git add data/raw.dvc .gitignore
git commit -m "Track dataset with DVC"
```

(Optional) Add remote storage:

```bash
dvc remote add -d storage <REMOTE_URL>
dvc push
```

Pull dataset later:

```bash
dvc pull
```

View pipeline graph:

```bash
dvc dag
```

---

## 🧠 Model Training & Experiment Tracking

Train model:

```bash
python -m src.train --data_dir data/raw --epochs 3
```

Outputs:

- Saved model → `models/cats_dogs_model.pt`
- Metrics & logs → MLflow

Start MLflow UI:

```bash
mlflow ui
```

Open: http://localhost:5000

---

## 🚀 Running the API Locally

Start FastAPI server:

```bash
uvicorn app.main:app --reload
```

### Health Check

```bash
curl http://localhost:8000/health
```

### Prediction

```bash
curl -X POST -F "file=@data/splits/test/Cat/0.jpg" http://localhost:8000/predict 
```

### Metrics Endpoint

```bash
curl http://localhost:8000/metrics
```

---

## 🐳 Docker Containerization

### Build Docker Image

```bash
docker build -t cats-dogs-classifier:latest .
```

### Run Container

```bash
docker run -p 8000:8000 -e MODEL_PATH=models/cats_dogs_model.pt cats-dogs-classifier:latest
```

Test:

```bash
curl http://localhost:8000/health
```

---

## 🚢 Deployment with Docker Compose

```bash
cd deployment
docker-compose up -d
```

Verify:

```bash
curl http://localhost:8000/health
```

---

## 🧪 Running Tests

```bash
pytest
```

---

## 🔁 CI Pipeline (GitHub Actions)

On every push:

✔ Install dependencies  
✔ Run tests  
✔ Build Docker image  
✔ Verify build success  

View pipeline → GitHub → **Actions tab**

---

## 🔄 CD Deployment Flow

On main branch updates:

✔ Pull latest image  
✔ Restart service  
✔ Run health check  
✔ Fail deployment if service unhealthy  

---

## 📊 Monitoring & Logging

### Logging
The API logs:

- incoming requests  
- prediction latency  
- errors  

### Metrics
Prometheus metrics available at:

```
/metrics
```

Includes:

- request count  
- response time  
- error rate  

---

## 🌐 API Endpoints

### ✅ Health Check
`GET /health`

Response:
```
{"status": "ok"}
```

### ✅ Prediction
`POST /predict`

Response:
```
{
  "label": "dog",
  "probability": 0.94
}
```

### ✅ Metrics
`GET /metrics`

Returns service metrics.

---

## 🎥 Demonstration Workflow

This project demonstrates:

1️⃣ Code & data versioning  
2️⃣ Model training & MLflow tracking  
3️⃣ Containerized API service  
4️⃣ CI pipeline automation  
5️⃣ Deployment & health checks  
6️⃣ Live prediction & monitoring  


