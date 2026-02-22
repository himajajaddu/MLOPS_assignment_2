# app/tests/test_inference_utils.py
from PIL import Image
import torch

from app.model import SmallCNN, predict_image

def test_predict_image_returns_probabilities_sum_to_1():
    device = torch.device("cpu")
    model = SmallCNN(num_classes=2).to(device).eval()

    meta = {"class_names": ["Cat", "Dog"], "img_size": 224}
    img = Image.new("RGB", (224, 224), color=(0, 255, 0))

    out = predict_image(model, meta, img, device)

    assert "predicted_class" in out
    assert "probabilities" in out
    probs = out["probabilities"]
    assert set(probs.keys()) == {"Cat", "Dog"}

    s = float(probs["Cat"]) + float(probs["Dog"])
    assert abs(s - 1.0) < 1e-5