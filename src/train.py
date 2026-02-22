import argparse
import os
import random
import shutil
from pathlib import Path
from typing import Tuple, List

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image, ImageFile, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = True


# -----------------------------
# Model
# -----------------------------
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


# -----------------------------
# Safe dataset (skips broken images)
# -----------------------------
class SafeImageFolder(datasets.ImageFolder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        self.imgs = self.samples

    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except (UnidentifiedImageError, OSError):
            new_index = (index + 1) % len(self.samples)
            return self.__getitem__(new_index)


# -----------------------------
# Utils
# -----------------------------
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(force_cpu: bool) -> torch.device:
    if not force_cpu and torch.backends.mps.is_available():
        return torch.device("mps")
    if not force_cpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _list_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return [p for p in folder.rglob("*") if p.suffix.lower() in exts and p.is_file()]


def split_raw_to_folders(
    raw_dir: str,
    out_dir: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    copy: bool = True,
) -> Path:
    """
    raw_dir expected:
      raw_dir/Cat/*.jpg
      raw_dir/Dog/*.jpg

    Creates:
      out_dir/train/Cat, out_dir/train/Dog
      out_dir/val/Cat,   out_dir/val/Dog
      out_dir/test/Cat,  out_dir/test/Dog
    """
    raw = Path(raw_dir)
    out = Path(out_dir)

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    classes = ["Cat", "Dog"]
    for c in classes:
        if not (raw / c).exists():
            raise FileNotFoundError(f"Expected folder not found: {raw/c}")

    # Create folders
    for split in ["train", "val", "test"]:
        for c in classes:
            (out / split / c).mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    for c in classes:
        imgs = _list_images(raw / c)
        rng.shuffle(imgs)

        n = len(imgs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        train_imgs = imgs[:n_train]
        val_imgs = imgs[n_train:n_train + n_val]
        test_imgs = imgs[n_train + n_val:]

        def _move_or_copy(paths: List[Path], dst: Path):
            for p in paths:
                target = dst / p.name
                if copy:
                    shutil.copy2(p, target)
                else:
                    shutil.move(p, target)

        _move_or_copy(train_imgs, out / "train" / c)
        _move_or_copy(val_imgs, out / "val" / c)
        _move_or_copy(test_imgs, out / "test" / c)

        print(f"[Split] {c}: total={n} train={len(train_imgs)} val={len(val_imgs)} test={len(test_imgs)}")

    return out


def build_loaders_from_splits(
    split_root: Path,
    img_size: int,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    train_ds = SafeImageFolder(root=str(split_root / "train"), transform=tfm)
    val_ds = SafeImageFolder(root=str(split_root / "val"), transform=tfm)
    test_ds = SafeImageFolder(root=str(split_root / "test"), transform=tfm)

    class_names = train_ds.classes
    if len(class_names) != 2:
        raise ValueError(f"Expected exactly 2 classes. Found {class_names}")

    # For Mac stability: num_workers=0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    return train_loader, val_loader, test_loader, class_names


@torch.no_grad()
def evaluate_full(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, List[int], List[int]]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total = 0
    correct = 0
    y_true_all: List[int] = []
    y_pred_all: List[int] = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)

        correct += (preds == y).sum().item()
        total += x.size(0)

        y_true_all.extend(y.detach().cpu().tolist())
        y_pred_all.extend(preds.detach().cpu().tolist())

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc, y_true_all, y_pred_all


def save_confusion_matrix_png(y_true: List[int], y_pred: List[int], class_names: List[str], out_path: Path) -> None:
    # Pure matplotlib (no seaborn)
    import matplotlib.pyplot as plt

    # Build 2x2 confusion matrix
    cm = [[0, 0], [0, 0]]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    fig = plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix (Test)")
    plt.xticks([0, 1], class_names)
    plt.yticks([0, 1], class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i][j]), ha="center", va="center")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


# -----------------------------
# Training
# -----------------------------
def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = pick_device(args.cpu)

    # MLflow
    os.makedirs("mlruns", exist_ok=True)
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment_name)

    # 1) Split raw -> train/val/test folders
    split_root = split_raw_to_folders(
        raw_dir=args.data_dir,
        out_dir=args.split_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        copy=True,   # keep raw intact
    )

    # 2) Build dataloaders from splits
    train_loader, val_loader, test_loader, class_names = build_loaders_from_splits(
        split_root=split_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
    )

    # 3) Model
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
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "model": "SmallCNN",
            "device": str(device),
            "raw_dir": str(Path(args.data_dir).resolve()),
            "split_dir": str(Path(args.split_dir).resolve()),
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
            val_loss, val_acc, _, _ = evaluate_full(model, val_loader, device)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

            print(f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

            # Save best by val accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                payload = {
                    "state_dict": model.state_dict(),
                    "class_names": class_names,
                    "img_size": args.img_size,
                    "architecture": "SmallCNN",
                }
                torch.save(payload, model_path)

        # 4) Load best model for test evaluation
        best_payload = torch.load(str(model_path), map_location=device)
        model.load_state_dict(best_payload["state_dict"])
        model.eval()

        test_loss, test_acc, y_true, y_pred = evaluate_full(model, test_loader, device)

        mlflow.log_metric("best_val_accuracy", best_val_acc)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)

        # Confusion matrix artifact
        cm_path = Path("reports") / "confusion_matrix_test.png"
        save_confusion_matrix_png(y_true, y_pred, class_names, cm_path)
        mlflow.log_artifact(str(cm_path))

        # Log model artifact
        mlflow.log_artifact(str(model_path))

        print(f"\nSaved best model to: {model_path}")
        print(f"Classes: {class_names} | Best val acc: {best_val_acc:.4f} | Test acc: {test_acc:.4f}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Cats vs Dogs classifier with raw->split folders + MLflow logging")
    p.add_argument("--data_dir", default="data/raw", help="Raw dataset root: data/raw/Cat and data/raw/Dog")
    p.add_argument("--split_dir", default="data/splits", help="Where to write train/val/test folders")
    p.add_argument("--output_dir", default="models", help="Where to save trained model")
    p.add_argument("--model_name", default="cats_dogs_model.pt", help="Model filename")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img_size", type=int, default=224)  # Assignment wants 224x224
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA/MPS is available")
    p.add_argument("--mlflow_uri", default="file:./mlruns")
    p.add_argument("--experiment_name", default="cats-vs-dogs-classification")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)