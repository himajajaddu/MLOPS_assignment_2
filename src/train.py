import argparse
import os
from pathlib import Path
import random
from typing import Tuple

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image, ImageFile, UnidentifiedImageError
ImageFile.LOAD_TRUNCATED_IMAGES = True

# A small CNN that trains quickly (uses AdaptiveAvgPool to avoid huge FC layers)
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
class SafeImageFolder(datasets.ImageFolder):
    """
    ImageFolder that skips corrupted/unreadable images instead of crashing.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Filter out broken images once at init (fast, avoids worker crashes)
        good_samples = []
        bad = 0
        for path, cls in self.samples:
            try:
                with Image.open(path) as img:
                    img.verify()
                good_samples.append((path, cls))
            except Exception:
                bad += 1

        if bad > 0:
            print(f"[SafeImageFolder] Skipped {bad} corrupted images.")
        self.samples = good_samples
        self.imgs = self.samples  # torchvision compatibility

    def __getitem__(self, index):
        # Extra safety at runtime
        try:
            return super().__getitem__(index)
        except (UnidentifiedImageError, OSError) as e:
            # if something slips through, try next sample
            new_index = (index + 1) % len(self.samples)
            return self.__getitem__(new_index)

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(
    data_dir: str,
    img_size: int,
    batch_size: int,
    val_split: float,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, list]:
    """
    Expects ImageFolder structure:
      data_dir/
        cats/
        dogs/
    or
      data_dir/train/
        cats/
        dogs/
    """
    data_path = Path(data_dir)

    # Allow passing either the class root or a folder that contains train/
    if (data_path / "train").exists() and (data_path / "train").is_dir():
        data_path = data_path / "train"

    if not (data_path.exists() and data_path.is_dir()):
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    #dataset = datasets.ImageFolder(root=str(data_path), transform=tfm)
    dataset = SafeImageFolder(root=str(data_path), transform=tfm)
    class_names = dataset.classes

    if len(class_names) != 2:
        raise ValueError(
            f"Expected exactly 2 classes (cats & dogs). Found: {class_names}. "
            "Ensure folders are named like 'cats' and 'dogs'."
        )

    val_len = int(len(dataset) * val_split)
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))
    #train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    #val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
  
    return train_loader, val_loader, class_names


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total, correct = 0, 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)

    #device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if not args.cpu and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif not args.cpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")




    # M1: Experiment Tracking using MLflow
    os.makedirs("mlruns", exist_ok=True)
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment_name)

    train_loader, val_loader, class_names = build_dataloaders(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
    )

    model = SmallCNN(num_classes=len(class_names)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / args.model_name

    with mlflow.start_run():
        mlflow.log_params({
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "img_size": args.img_size,
            "val_split": args.val_split,
            "model": "SmallCNN",
            "device": str(device),
            "data_dir": str(Path(args.data_dir).resolve()),
        })

        best_val_acc = 0.0
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            seen = 0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * x.size(0)
                seen += x.size(0)

            train_loss = running_loss / max(seen, 1)
            val_loss, val_acc = evaluate(model, val_loader, device)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

            print(f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                payload = {
                    "state_dict": model.state_dict(),
                    "class_names": class_names,
                    "img_size": args.img_size,
                    "architecture": "SmallCNN",
                }
                torch.save(payload, model_path)

        mlflow.log_metric("best_val_accuracy", best_val_acc)
        mlflow.log_artifact(str(model_path))

        print(f"\nSaved best model to: {model_path}")
        print(f"Classes: {class_names} | Best val acc: {best_val_acc:.4f}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Cats vs Dogs classifier (real training)")
    p.add_argument("--data_dir", default="data/raw", help="Path to dataset root (expects ImageFolder structure)")
    p.add_argument("--output_dir", default="models", help="Where to save trained model")
    p.add_argument("--model_name", default="cats_dogs_model.pt", help="Model filename")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    p.add_argument("--mlflow_uri", default="file:./mlruns")
    p.add_argument("--experiment_name", default="cats-vs-dogs-classification")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
