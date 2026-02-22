# app/tests/test_preprocess_split.py
from pathlib import Path
from PIL import Image
import pytest

from src.train import split_raw_to_folders

def _make_dummy_image(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (50, 50), color=(255, 0, 0))
    img.save(path)

@pytest.mark.parametrize("n_cat,n_dog", [(10, 10), (25, 30)])
def test_split_raw_to_folders_creates_expected_structure(tmp_path: Path, n_cat: int, n_dog: int):
    raw = tmp_path / "raw"
    (raw / "Cat").mkdir(parents=True)
    (raw / "Dog").mkdir(parents=True)

    for i in range(n_cat):
        _make_dummy_image(raw / "Cat" / f"cat_{i}.jpg")
    for i in range(n_dog):
        _make_dummy_image(raw / "Dog" / f"dog_{i}.jpg")

    out = tmp_path / "splits"
    split_root = split_raw_to_folders(
        raw_dir=str(raw),
        out_dir=str(out),
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
        copy=True,
    )

    # Check directories exist
    for split in ["train", "val", "test"]:
        for cls in ["Cat", "Dog"]:
            assert (split_root / split / cls).exists()

    # Check counts sum up
    cat_total = 0
    dog_total = 0
    for split in ["train", "val", "test"]:
        cat_total += len(list((split_root / split / "Cat").glob("*.jpg")))
        dog_total += len(list((split_root / split / "Dog").glob("*.jpg")))

    assert cat_total == n_cat
    assert dog_total == n_dog