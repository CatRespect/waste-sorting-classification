"""Hash, deduplicate and split TrashNet.

Usage:
    python scripts/prepare_data.py --root data/raw/dataset-resized --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import CLASSES, build_index, make_split  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deduplicated stratified TrashNet splits")
    parser.add_argument("--root", default="data/raw/dataset-resized", help="Dataset root with one folder per class")
    parser.add_argument("--out", default="data", help="Directory for splits.csv and removed_duplicates.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    args = parser.parse_args()

    clean, removed = build_index(args.root)
    split = make_split(clean, val=args.val, test=args.test, seed=args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    split.to_csv(out_dir / "splits.csv", index=False)
    removed.to_csv(out_dir / "removed_duplicates.csv", index=False)

    print(f"kept {len(clean)} images; removed {len(removed)} duplicates")
    if not removed.empty:
        print(removed[["path", "reason"]].to_string(index=False))
    distribution = (
        split.pivot_table(index="class", columns="split", values="path", aggfunc="count")
        .reindex(CLASSES)
        .fillna(0)
    )
    print(distribution[["train", "val", "test"]].assign(total=lambda frame: frame.sum(axis=1)).to_string())


if __name__ == "__main__":
    main()
