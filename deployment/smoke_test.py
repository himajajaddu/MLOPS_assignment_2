import io
import sys
import time
import requests
from PIL import Image

BASE_URL = "http://localhost:8000"

def wait_for_health(timeout_s: int = 30) -> dict:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=3)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("Service did not become healthy in time")

def main():
    health = wait_for_health()
    print("Health:", health)

    # Create a tiny valid jpeg
    img = Image.new("RGB", (64, 64))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    files = {"file": ("smoke.jpg", buf.getvalue(), "image/jpeg")}
    r = requests.post(f"{BASE_URL}/predict", files=files, timeout=30)

    # If the model is not trained/mounted, API returns 503 (acceptable for smoke test if you haven't trained yet)
    if r.status_code == 503:
        print("Predict returned 503 (model not loaded). Train the model and retry.")
        return 0

    r.raise_for_status()
    print("Predict:", r.json())
    return 0

if __name__ == "__main__":
    sys.exit(main())
