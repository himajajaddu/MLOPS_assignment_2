# deployment/smoke_test.py
import os
import sys
import time
import requests

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

def wait_for_service(timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=5)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("Service did not become ready in time")

def main():
    health = wait_for_service()
    if not health.get("model_loaded"):
        raise RuntimeError(f"Model not loaded: {health}")

    # Use a small sample image shipped with repo (you should keep 1 tiny image in repo)
    sample_path = os.getenv("SMOKE_IMAGE", "app/tests/assets/sample_cat.jpg")
    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"Smoke image not found: {sample_path}")

    with open(sample_path, "rb") as f:
        files = {"file": ("sample_cat.jpg", f, "image/jpeg")}
        r = requests.post(f"{BASE_URL}/predict", files=files, timeout=30)

    if r.status_code != 200:
        raise RuntimeError(f"/predict failed: {r.status_code}, {r.text}")

    out = r.json()
    if "prediction" not in out and "predicted_class" not in out:
        raise RuntimeError(f"Unexpected response: {out}")

    print("✅ Smoke test passed")
    print("Health:", health)
    print("Predict:", out)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("❌ Smoke test failed:", e)
        sys.exit(1)