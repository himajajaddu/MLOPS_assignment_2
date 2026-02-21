# MLOps Assignment 2: End-to-End MLOps Pipeline Guide

This folder contains all the necessary boilerplate scripts, configuration files, and workflows required to complete your MLOps assignment for the Cats vs Dogs binary classification. 

Below are the step-by-step instructions to fulfill **all** the assignment tasks.

---

## M1: Model Development & Experiment Tracking

### 1. Data & Code Versioning
1. **Initialize Git:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit of MLOps assignment boilerplate"
   ```
2. **Initialize DVC for dataset versioning:**
   ```bash
   dvc init
   mkdir -p data/raw
   # Download Cats vs Dogs dataset from Kaggle and place it in data/raw/
   # Track the data with DVC
   dvc add data/raw
   git add data/raw.dvc .gitignore
   git commit -m "Add raw dataset via DVC"
   ```

### 2. Model Building & Experiment Tracking
The baseline CNN model and MLflow experiment tracking are pre-configured in `src/train.py`.
1. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run training:**
   ```bash
   python src/train.py
   ```
   *This script simulates training, logs metrics (loss, accuracy) and parameters to MLflow, and saves `cats_dogs_model.pt` in the `models/` folder.*
3. **View MLflow UI:**
   ```bash
   mlflow ui --backend-store-uri sqlite:///mlflow.db
   ```
   *Open your browser to http://127.0.0.1:5000 to see your tracked experiments.*

---

## M2: Model Packaging & Containerization

### 1. Inference Service
The FastAPI app is already created at `app/main.py` with two endpoints (`/health` and `/predict`).

### 2. Run Locally & Verify
1. **Start the FastAPI service locally:**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
2. **Test via Curl/Postman:**
   ```bash
   # Health check
   curl http://localhost:8000/health
   
   # Predict
   curl -X POST -F "file=@data/test_cat.jpg" http://localhost:8000/predict
   ```

### 3. Containerization
1. **Build the Docker Image:**
   ```bash
   docker build -t cats-dogs-classifier:latest .
   ```
2. **Run the container:**
   ```bash
   docker run -p 8000:8000 cats-dogs-classifier:latest
   ```

---

## M3: CI Pipeline for Build, Test & Image Creation

1. **Automated Testing:** Pytest unit tests for the endpoints are in `app/tests/test_api.py`.
   ```bash
   pytest app/tests/
   ```
2. **GitHub Actions CI Setup:** 
   The pipeline is defined in `.github/workflows/ci.yml`. It automatically checks out code, installs dependencies, runs pytest, and builds the Docker image on every push to the `main` branch.
3. **Publishing Artifacts:**
   Update your GitHub Repository Secrets (`DOCKER_USERNAME` and `DOCKER_PASSWORD`) and uncomment the Docker Push steps in the `ci.yml` file to push images to Docker Hub.

---

## M4: CD Pipeline & Deployment

1. **Docker Compose Target:** 
   We chose Docker Compose as the deployment target. The configuration is at `deployment/docker-compose.yml`.
   ```bash
   cd deployment
   docker-compose up -d
   ```
2. **Smoke Tests:**
   A smoke test script (`deployment/smoke_test.py`) has been provided. This script calls the health check and prediction endpoints. If either fails, the script exits with an error code, which can be configured to fail a CI/CD pipeline.
   ```bash
   python deployment/smoke_test.py
   ```

---

## M5: Monitoring, Logs & Final Submission

1. **Monitoring & Logging:**
   - Standard Python logging has been implemented in `app/main.py`.
   - **Prometheus** metrics are exposed automatically via `prometheus-fastapi-instrumentator`. You can access the metrics at `http://localhost:8000/metrics`.

2. **Final Submission Preparation:**
   - Ensure all code is pushed to your Git repository.
   - Record a 5-minute screen recording showcasing:
     1. Making a code change.
     2. Running `git push` which triggers GitHub Actions.
     3. Showing the completed GitHub Actions pipeline.
     4. Sending a sample prediction request to your running deployed container.
   - Zip this entire folder (excluding virtual environments and large datasets) to submit to your assignment portal.

```bash
# Example zip command excluding large folders
zip -r mlops_assignment_submission.zip mlops_assignment/ -x "*/__pycache__/*" -x "*/data/raw/*"
```