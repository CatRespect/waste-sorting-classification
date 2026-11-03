"""Validation/test metrics, confusion matrices and inference benchmark.

Examples:
    python -m src.evaluate --config configs/baseline_effb0_fe.yaml --split val
    python -m src.evaluate --config configs/baseline_effb0_fe.yaml --split test --tta
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .data import CLASSES, make_loaders
from .models import build_model
from .train import autocast_context, predict


def _load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch versions before weights_only was added.
        return torch.load(path, map_location=device)


@torch.no_grad()
def benchmark(
    model,
    device: torch.device,
    size: int = 224,
    batch: int = 1,
    warmup: int = 5,
    iters: int = 20,
    amp: bool = False,
    channels_last: bool = False,
    tta: bool = False,
):
    model.eval()
    images = torch.randn(batch, 3, size, size, device=device)
    if channels_last:
        images = images.contiguous(memory_format=torch.channels_last)
    def infer(batch_images):
        with autocast_context(device, amp):
            model(batch_images)
            if tta:
                model(torch.flip(batch_images, dims=[3]))

    for _ in range(warmup):
        infer(images)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iters):
        infer(images)
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = (time.perf_counter() - started) / max(1, iters)
    return {
        "batch": batch,
        "precision": "fp16" if amp and device.type == "cuda" else "fp32",
        "channels_last": bool(channels_last),
        "ms_per_batch": seconds * 1e3,
        "ms_per_image": seconds * 1e3 / batch,
        "img_per_sec": batch / max(seconds, 1e-12),
    }


def plot_cm(cm: np.ndarray, path: Path, title: str) -> None:
    row_totals = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, row_totals, out=np.zeros_like(cm, dtype=float), where=row_totals != 0)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    for row in range(len(CLASSES)):
        for col in range(len(CLASSES)):
            ax.text(
                col, row, str(cm[row, col]), ha="center", va="center",
                color="white" if normalized[row, col] > 0.5 else "black", fontsize=9,
            )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained TrashNet checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--tta", action="store_true", help="average image and horizontal-mirror probabilities")
    parser.add_argument("--no-benchmark", action="store_true", help="skip latency measurements")
    parser.add_argument("--benchmark-iters", type=int, default=20)
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    seed = args.seed if args.seed is not None else int(cfg["seed"])
    run_dir = Path(cfg["output_dir"]) / f"{cfg['name']}_seed{seed}"
    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Run `python -m src.train --config {args.config}` first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.get("amp", False)) and device.type == "cuda"
    channels_last = bool(cfg.get("channels_last", False)) and device.type == "cuda"
    loaders, split_df = make_loaders(
        cfg["data_root"], cfg["split_csv"], cfg.get("batch_size", 32),
        cfg.get("num_workers", 2), cfg.get("img_size", 224),
        cfg.get("persistent_workers", False),
    )

    # The project checkpoint is complete; evaluation must not try to download
    # ImageNet/CLIP weights first. The checkpoint replaces this random init.
    model = build_model(
        cfg["backbone"],
        num_classes=6,
        strategy=cfg["strategy"],
        dropout=float(cfg.get("dropout", 0.2)),
        img_size=int(cfg.get("img_size", 224)),
        pretrained=False,
    ).to(device)
    model.load_state_dict(_load_checkpoint(checkpoint_path, device))
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    y_true, y_pred, probabilities = predict(
        model, loaders[args.split], device, amp=use_amp,
        channels_last=channels_last, tta=args.tta,
    )
    report = classification_report(
        y_true, y_pred, labels=list(range(len(CLASSES))), target_names=CLASSES,
        output_dict=True, zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))

    suffix = "_tta" if args.tta else ""
    split_name = args.split
    plot_cm(matrix, run_dir / f"{split_name}_confusion_matrix{suffix}.png", f"{cfg['name']} ({split_name})")
    rows = split_df[split_df.split == args.split].reset_index(drop=True)
    if len(rows) != len(y_pred):
        raise RuntimeError("Prediction count does not match the split CSV row count")
    wrong = rows.assign(
        true=[CLASSES[int(value)] for value in y_true],
        pred=[CLASSES[int(value)] for value in y_pred],
        confidence=probabilities.max(axis=1),
    )
    wrong = wrong[wrong.true != wrong.pred].sort_values("confidence", ascending=False)
    wrong.to_csv(run_dir / f"{split_name}_misclassified{suffix}.csv", index=False)

    benchmarks = []
    if not args.no_benchmark and args.split == "test":
        batch_sizes = sorted({1, min(16, int(cfg.get("batch_size", 16)))})
        for batch_size in batch_sizes:
            benchmarks.append(
                benchmark(
                    model, device, int(cfg.get("img_size", 224)), batch_size,
                    iters=max(1, args.benchmark_iters), amp=False, channels_last=False, tta=args.tta,
                )
            )
        if device.type == "cuda":
            benchmarks.append(
                benchmark(
                    model, device, int(cfg.get("img_size", 224)),
                    min(16, int(cfg.get("batch_size", 16))),
                    iters=max(1, args.benchmark_iters), amp=use_amp, channels_last=channels_last, tta=args.tta,
                )
            )

    metrics = {
        "config_name": cfg["name"],
        "backbone": cfg["backbone"],
        "strategy": cfg["strategy"],
        "seed": seed,
        "split": split_name,
        "tta": bool(args.tta),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": {class_name: report[class_name] for class_name in CLASSES},
        "confusion_matrix": matrix.tolist(),
        "n_samples": int(len(y_true)),
        "n_misclassified": int((y_true != y_pred).sum()),
        "device": str(device),
        "torch_threads": torch.get_num_threads(),
        "amp": use_amp,
        "channels_last": channels_last,
        "benchmark": benchmarks,
    }
    metrics_path = run_dir / f"{split_name}_metrics{suffix}.json"
    with metrics_path.open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, ensure_ascii=False, indent=2)

    print(json.dumps({k: metrics[k] for k in ("config_name", "split", "accuracy", "macro_f1", "n_samples")}, indent=2))
    print(pd.DataFrame({name: report[name] for name in CLASSES}).T.round(3))
    print(matrix)
    for item in benchmarks:
        print(
            f"batch={item['batch']:>2} {item['precision']}: "
            f"{item['ms_per_image']:.2f} ms/image, {item['img_per_sec']:.1f} img/s"
        )
    


if __name__ == "__main__":
    main()
