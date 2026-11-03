"""TrashNet indexing, transforms, and DataLoader utilities."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_index(root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Index class folders, remove exact duplicates and flag cross-class collisions."""
    root = Path(root)
    rows = []
    for label, class_name in enumerate(CLASSES):
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        paths = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        for path in paths:
            rows.append(
                {
                    "path": str(path.relative_to(root)),
                    "label": label,
                    "class": class_name,
                    "md5": _md5(path),
                }
            )
    if not rows:
        raise ValueError(f"No images found under {root}; expected class subfolders {CLASSES}")

    df = pd.DataFrame(rows)
    by_hash: dict[str, set[str]] = defaultdict(set)
    for digest, class_name in zip(df.md5, df["class"]):
        by_hash[digest].add(class_name)
    conflict = df.md5.map(lambda digest: len(by_hash[digest]) > 1)
    same_duplicate = df.duplicated("md5", keep="first") & ~conflict
    removed = df[conflict | same_duplicate].copy()
    removed["reason"] = removed.md5.map(
        lambda digest: "cross-class duplicate" if len(by_hash[digest]) > 1 else "same-class duplicate"
    )
    clean = df[~(conflict | same_duplicate)].reset_index(drop=True)
    return clean, removed.reset_index(drop=True)


def make_split(
    df: pd.DataFrame, val: float = 0.15, test: float = 0.15, seed: int = 42
) -> pd.DataFrame:
    """Create a stratified train/validation/test split."""
    if not 0 < val < 1 or not 0 < test < 1 or val + test >= 1:
        raise ValueError("val and test must be positive fractions whose sum is < 1")
    trainval, test_df = train_test_split(
        df, test_size=test, stratify=df.label, random_state=seed
    )
    train_df, val_df = train_test_split(
        trainval,
        test_size=val / (1 - test),
        stratify=trainval.label,
        random_state=seed,
    )
    out = pd.concat(
        [train_df.assign(split="train"), val_df.assign(split="val"), test_df.assign(split="test")]
    )
    return out.sort_values(["split", "path"]).reset_index(drop=True)


def get_transforms(train: bool, size: int = 224):
    if size <= 0:
        raise ValueError(f"Image size must be positive, got {size}")
    if train:
        return T.Compose(
            [
                T.RandomResizedCrop(size, scale=(0.7, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandomRotation(15),
                T.ColorJitter(0.2, 0.2, 0.2, 0.02),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    # Keep the original 256:224 resize-to-crop ratio, scaled with img_size.
    resize_to = int(round(size * 256 / 224))
    return T.Compose(
        [
            T.Resize(resize_to),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class TrashNetDataset(Dataset):
    def __init__(self, root: str | Path, index: pd.DataFrame, transform=None):
        self.root = Path(root)
        self.items = list(zip(index.path.astype(str), index.label.astype(int)))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        relative_path, label = self.items[index]
        image_path = self.root / relative_path
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform:
                image = self.transform(image)
        return image, label


def class_weights(labels, n_classes: int = len(CLASSES)) -> torch.Tensor:
    """Inverse-frequency class weights normalized to mean one."""
    counts = torch.bincount(torch.as_tensor(list(labels), dtype=torch.long), minlength=n_classes).float()
    if (counts == 0).any():
        missing = [CLASSES[i] for i, value in enumerate(counts) if value == 0]
        raise ValueError(f"Cannot compute class weights; no training examples for: {missing}")
    weights = counts.sum() / (n_classes * counts)
    return weights / weights.mean()


def make_loaders(
    root,
    split_csv,
    batch_size=32,
    num_workers=2,
    size=224,
    persistent_workers=False,
):
    """Create train/val/test loaders from the existing split CSV."""
    root, split_csv = Path(root), Path(split_csv)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {root}. Download/unpack TrashNet or set data_root in the config."
        )
    if not split_csv.is_file():
        raise FileNotFoundError(
            f"Split CSV not found: {split_csv}. Run `python scripts/prepare_data.py --root <dataset>` first."
        )
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be >= 1 and num_workers must be >= 0")

    df = pd.read_csv(split_csv)
    required = {"path", "label", "split"}
    if not required.issubset(df.columns):
        raise ValueError(f"Split CSV must contain columns {sorted(required)}; got {list(df.columns)}")

    loaders = {}
    for split in ("train", "val", "test"):
        part = df[df.split == split].reset_index(drop=True)
        if part.empty:
            raise ValueError(f"No rows found for split={split!r} in {split_csv}")
        dataset = TrashNetDataset(root, part, get_transforms(split == "train", size))
        keep_workers = persistent_workers and num_workers > 0 and split == "train"
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=bool(keep_workers),
        )
    return loaders, df
