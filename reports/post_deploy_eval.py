# reports/post_deploy_eval.py
import os
import json
import random
from pathlib import Path
import requests

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
TEST_DIR = Path(os.getenv("TEST_DIR", "/Users/apple/Documents/MLOPS_assignment_2/data/splits/test"))  # expects test/Cat and test/Dog
SAMPLE_N = int(os.getenv("SAMPLE_N", "50"))

def collect_images():
    items = []
    for cls in ["Cat", "Dog"]:
        folder = TEST_DIR / cls
        print(f"Collecting from {folder}...")
        for p in folder.glob("*"):
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                items.append((p, cls))
                print(f"  Found: {p} -> {cls}")
    random.shuffle(items)
    return items[:SAMPLE_N]

def predict_api(img_path: Path):
    with open(img_path, "rb") as f:
        files = {"file": (img_path.name, f, "image/jpeg")}
        r = requests.post(f"{BASE_URL}/predict", files=files, timeout=30)
    r.raise_for_status()
    return r.json()

def main():
    data = collect_images()
    print(data[:5])  # Show some samples
    if not data:
        raise RuntimeError(f"No test images found at {TEST_DIR}")

    y_true = []
    y_pred = []
    for path, label in data:
        out = predict_api(path)
        pred = out.get("prediction") or out.get("predicted_class")
        y_true.append(label)
        y_pred.append(pred)

    correct = sum(t == p for t, p in zip(y_true, y_pred))
    acc = correct / len(y_true)

    results = {
        "base_url": BASE_URL,
        "test_dir": str(TEST_DIR),
        "sample_n": len(y_true),
        "accuracy": acc,
        "examples": [
            {"image": str(p), "true": t, "pred": pr}
            for (p, _), t, pr in zip(data, y_true, y_pred)
        ][:10]
    }

    Path("reports").mkdir(exist_ok=True)
    with open("reports/post_deploy_eval.json", "w") as f:
        json.dump(results, f, indent=2)

    print("✅ Post-deploy evaluation saved to reports/post_deploy_eval.json")
    print("Accuracy:", acc)

if __name__ == "__main__":
    main()