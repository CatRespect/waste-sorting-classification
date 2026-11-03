"""Train one model/configuration.

Examples:
    python -m src.train --config configs/baseline_effb0_fe.yaml
    python -m src.train --config configs/effb0_finetune.yaml --epochs 2 --limit-batches 5
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score

from .data import class_weights, make_loaders
from .models import (
    build_model,
    count_params,
    layer_decay_param_groups,
    param_groups,
    set_unfrozen_stages,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device, enabled: bool):
    """CUDA fp16 autocast; disabled/no-op on CPU."""
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def freeze_bn_in_frozen_modules(model: nn.Module) -> None:
    """Keep running statistics fixed only for frozen BatchNorm modules."""
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            own_params = list(module.parameters(recurse=False))
            if not own_params or all(not p.requires_grad for p in own_params):
                module.eval()


@torch.no_grad()
def predict(
    model: nn.Module,
    loader,
    device: torch.device,
    amp: bool = False,
    channels_last: bool = False,
    tta: bool = False,
):
    """Return labels, predictions and softmax probabilities."""
    model.eval()
    ys, predictions, probabilities = [], [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        with autocast_context(device, amp):
            probs = model(images).float().softmax(dim=1)
            if tta:
                mirror_probs = model(torch.flip(images, dims=[3])).float().softmax(dim=1)
                probs = (probs + mirror_probs) / 2
        probs = probs.cpu()
        ys.append(labels)
        predictions.append(probs.argmax(dim=1))
        probabilities.append(probs)
    return (
        torch.cat(ys).numpy(),
        torch.cat(predictions).numpy(),
        torch.cat(probabilities).numpy(),
    )


def _load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    required = {
        "name", "backbone", "strategy", "data_root", "split_csv", "output_dir",
        "img_size", "batch_size", "epochs", "seed", "head_lr", "backbone_lr",
        "weight_decay", "warmup_epochs", "patience",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Config {path} is missing required keys: {missing}")
    if config["strategy"] not in {"feature_extraction", "full_finetune", "layer_decay_finetune", "gradual_unfreeze"}:
        raise ValueError(f"Unsupported strategy {config['strategy']!r}")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a single TrashNet model configuration")
    parser.add_argument("--config", required=True, help="YAML experiment config")
    parser.add_argument("--seed", type=int, default=None, help="Override the seed in the YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override the epoch budget")
    parser.add_argument(
        "--limit-batches", type=int, default=None,
        help="Smoke test: run at most N training batches per epoch",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    if args.epochs is not None:
        if args.epochs < 1:
            parser.error("--epochs must be >= 1")
        cfg["epochs"] = args.epochs
    if args.limit_batches is not None and args.limit_batches < 1:
        parser.error("--limit-batches must be >= 1")

    seed = args.seed if args.seed is not None else int(cfg["seed"])
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, int(cfg.get("torch_threads", torch.get_num_threads()))))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", True))

    use_amp = bool(cfg.get("amp", False)) and device.type == "cuda"
    channels_last = bool(cfg.get("channels_last", False)) and device.type == "cuda"
    run_dir = Path(cfg["output_dir"]) / f"{cfg['name']}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    loaders, split_df = make_loaders(
        cfg["data_root"],
        cfg["split_csv"],
        int(cfg.get("batch_size", 32)),
        int(cfg.get("num_workers", 2)),
        int(cfg.get("img_size", 224)),
        bool(cfg.get("persistent_workers", False)),
    )
    model = build_model(
        cfg["backbone"],
        num_classes=6,
        strategy=cfg["strategy"],
        dropout=float(cfg.get("dropout", 0.2)),
        weights_file=cfg.get("weights_file"),
        img_size=int(cfg.get("img_size", 224)),
        pretrained=bool(cfg.get("pretrained", True)),
    ).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    params_total, params_trainable_initial = count_params(model)
    print(
        f"model={cfg['backbone']} strategy={cfg['strategy']} "
        f"params={params_total / 1e6:.2f}M/{params_trainable_initial / 1e6:.2f}M trainable "
        f"device={device} amp={use_amp} channels_last={channels_last}",
        flush=True,
    )

    weights = None
    if bool(cfg.get("class_weighted_loss", False)):
        weights = class_weights(split_df.loc[split_df.split == "train", "label"]).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=float(cfg.get("label_smoothing", 0.0)),
    )

    steps_per_epoch = len(loaders["train"])
    if args.limit_batches is not None:
        steps_per_epoch = min(steps_per_epoch, args.limit_batches)
    total_steps = int(cfg["epochs"]) * steps_per_epoch
    warmup_steps = int(cfg.get("warmup_epochs", 0)) * steps_per_epoch

    # Epoch (zero-based) -> number of top-level backbone stages to unfreeze.
    unfreeze_schedule = {
        int(epoch): int(stages)
        for epoch, stages in cfg.get("unfreeze_schedule", {0: 0}).items()
    }
    if cfg["strategy"] == "gradual_unfreeze" and not unfreeze_schedule:
        unfreeze_schedule = {0: 0}

    def rebuild_optimizer(step_offset: int):
        if cfg["strategy"] == "layer_decay_finetune":
            groups = layer_decay_param_groups(
                model,
                float(cfg["head_lr"]),
                float(cfg["backbone_lr"]),
                float(cfg.get("layer_decay", 0.75)),
                float(cfg["weight_decay"]),
            )
        else:
            groups = param_groups(model, float(cfg["head_lr"]), float(cfg["backbone_lr"]))
        optimizer = torch.optim.AdamW(groups, weight_decay=float(cfg["weight_decay"]))

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return (step + 1) / warmup_steps
            return 0.5 * (
                1 + math.cos(math.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps))
            )

        if step_offset > 0:
            # A fresh optimizer is created after unfreezing; retain the global
            # warmup/cosine step rather than restarting the schedule.
            for group in optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda, last_epoch=step_offset - 1
        )
        return optimizer, scheduler

    if cfg["strategy"] == "gradual_unfreeze":
        initial_stages = unfreeze_schedule.get(0, 0)
        if initial_stages == -1:
            for parameter in model.parameters():
                parameter.requires_grad = True
        else:
            set_unfrozen_stages(model, initial_stages)

    optimizer, scheduler = rebuild_optimizer(step_offset=0)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):  # compatibility with older supported PyTorch
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_val_f1 = -1.0
    bad_epochs = 0
    global_step = 0
    history = []
    params_trainable_final = params_trainable_initial

    for epoch in range(1, int(cfg["epochs"]) + 1):
        started = time.time()
        if cfg["strategy"] == "gradual_unfreeze" and (epoch - 1) in unfreeze_schedule:
            n_stages = unfreeze_schedule[epoch - 1]
            if n_stages == -1:
                for parameter in model.parameters():
                    parameter.requires_grad = True
            else:
                set_unfrozen_stages(model, n_stages)
            optimizer, scheduler = rebuild_optimizer(step_offset=global_step)
            _, params_trainable_final = count_params(model)
            print(
                f"epoch={epoch}: unfreezing={n_stages} stage(s), "
                f"trainable={params_trainable_final / 1e6:.2f}M",
                flush=True,
            )

        model.train()
        if cfg["strategy"] in ("feature_extraction", "gradual_unfreeze"):
            freeze_bn_in_frozen_modules(model)

        total_loss, sample_count, correct = 0.0, 0, 0
        for batch_idx, (images, labels) in enumerate(loaders["train"]):
            if args.limit_batches is not None and batch_idx >= args.limit_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if channels_last:
                images = images.contiguous(memory_format=torch.channels_last)

            with autocast_context(device, use_amp):
                logits = model(images)
                loss = criterion(logits.float(), labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            batch_size = int(labels.size(0))
            total_loss += float(loss.detach().item()) * batch_size
            sample_count += batch_size
            correct += int((logits.detach().argmax(dim=1) == labels).sum().item())

        if sample_count == 0:
            raise RuntimeError("Training loader produced no batches. Check the dataset and split CSV.")

        y_val, pred_val, _ = predict(
            model, loaders["val"], device, amp=use_amp, channels_last=channels_last
        )
        val_f1 = float(f1_score(y_val, pred_val, average="macro", zero_division=0))
        val_acc = float(accuracy_score(y_val, pred_val))
        head_parameter_ids = {id(parameter) for parameter in model.get_classifier().parameters()}
        head_lr = next(
            (float(group["lr"]) for group in optimizer.param_groups
             if any(id(parameter) in head_parameter_ids for parameter in group["params"])),
            None,
        )
        record = {
            "epoch": epoch,
            "train_loss": total_loss / sample_count,
            "train_acc": correct / sample_count,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
            "lr_head": head_lr,
            "sec": time.time() - started,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

        torch.save(model.state_dict(), run_dir / "last.pt")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            bad_epochs = 0
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= int(cfg.get("patience", 7)):
                print(f"early stopping after epoch {epoch}", flush=True)
                break

    train_log = {
        "config": cfg,
        "seed": seed,
        "backbone": cfg["backbone"],
        "strategy": cfg["strategy"],
        "params_total": params_total,
        "params_trainable_initial": params_trainable_initial,
        "params_trainable_final": params_trainable_final,
        "device": str(device),
        "amp": use_amp,
        "channels_last": channels_last,
        "train_seconds": sum(item["sec"] for item in history),
        "best_val_macro_f1": best_val_f1,
        "history": history,
    }
    with (run_dir / "train_log.json").open("w", encoding="utf-8") as stream:
        json.dump(train_log, stream, ensure_ascii=False, indent=2)
    print(f"best val macro-F1={best_val_f1:.4f}; checkpoint={run_dir / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
