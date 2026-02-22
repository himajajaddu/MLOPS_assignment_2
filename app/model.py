from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def pick_device(device_str: str = "") -> torch.device:
    d = (device_str or "").lower().strip()
    if d == "cpu":
        return torch.device("cpu")
    if d == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if d == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_preprocess(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


def load_model(model_path: str, device: torch.device) -> Tuple[nn.Module, Dict]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    payload = torch.load(str(path), map_location=device)
    class_names = payload.get("class_names", ["cats", "dogs"])
    img_size = int(payload.get("img_size", 128))

    model = SmallCNN(num_classes=len(class_names))
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()

    meta = {"class_names": class_names, "img_size": img_size}
    return model, meta


@torch.no_grad()
def predict_image(model: nn.Module, meta: Dict, image: Image.Image, device: torch.device) -> Dict:
    tfm = build_preprocess(meta["img_size"])
    x = tfm(image.convert("RGB")).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    class_names = meta["class_names"]
    pred_idx = int(torch.tensor(probs).argmax().item())
    return {
        "predicted_class": class_names[pred_idx],
        "probabilities": {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    }