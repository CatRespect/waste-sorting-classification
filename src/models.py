"""Model factory for TrashNet image classification.

Supports any timm model name plus the aliases below. Strategies:
- feature_extraction: freeze the pretrained backbone and train the classifier.
- full_finetune: train the complete network with discriminative learning rates.
- layer_decay_finetune: full fine-tuning with a per-layer LR decay (when timm
  exposes the required grouping hooks; otherwise falls back to two LR groups).
- gradual_unfreeze: begin with a frozen backbone and progressively unfreeze
  top-level backbone stages according to the schedule in the YAML config.
"""
from __future__ import annotations

from typing import Any

import timm
import torch.nn as nn

STRATEGIES = {
    "feature_extraction",
    "full_finetune",
    "layer_decay_finetune",
    "gradual_unfreeze",
}

BACKBONE_ALIASES = {
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
    "resnet50": "resnet50.a1_in1k",
    "convnext_tiny": "convnextv2_tiny.fcmae_ft_in22k_in1k",
    "effnetv2_s": "tf_efficientnetv2_s.in1k",
    "clip_vit_b32": "vit_base_patch32_clip_224.openai",
    "clip_vit_b16": "vit_base_patch16_clip_224.openai",
}


def resolve_backbone_name(backbone: str) -> str:
    """Resolve a short alias or return an arbitrary timm model name unchanged."""
    return BACKBONE_ALIASES.get(backbone, backbone)


def _instantiate(timm_name: str, **kwargs: Any) -> nn.Module:
    """Create a timm model, retrying without ``img_size`` only when unsupported."""
    try:
        return timm.create_model(timm_name, **kwargs)
    except TypeError as exc:
        # CNN backbones generally infer their input size from the input tensor.
        # ViT/CLIP models use img_size to initialize/interpolate position
        # embeddings, so preserve the argument whenever the model accepts it.
        if "img_size" not in kwargs or "img_size" not in str(exc):
            raise
        kwargs = dict(kwargs)
        kwargs.pop("img_size", None)
        return timm.create_model(timm_name, **kwargs)


def _initialize_classifier(head: nn.Module) -> None:
    """Initialize the final linear classifier to small, near-zero logits."""
    linear_layers = [m for m in head.modules() if isinstance(m, nn.Linear)]
    if not linear_layers:
        raise TypeError(
            "The selected timm model has no Linear classifier. "
            "Choose a classification model or provide a custom classifier."
        )
    final = linear_layers[-1]
    nn.init.normal_(final.weight, std=0.01)
    if final.bias is not None:
        nn.init.zeros_(final.bias)


def build_model(
    backbone: str,
    num_classes: int = 6,
    strategy: str = "feature_extraction",
    dropout: float = 0.3,
    weights_file: str | None = None,
    img_size: int = 224,
    pretrained: bool = True,
) -> nn.Module:
    """Build a pretrained (or randomly initialized) timm classifier.

    ``pretrained=False`` is intended for validation/evaluation after the
    project's trained ``state_dict`` has been loaded; it avoids a needless
    network download of ImageNet weights at evaluation time.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {sorted(STRATEGIES)}")
    if img_size <= 0:
        raise ValueError(f"img_size must be positive, got {img_size}")

    timm_name = resolve_backbone_name(backbone)
    kwargs: dict[str, Any] = {
        "pretrained": pretrained,
        "num_classes": num_classes,
        "drop_rate": dropout,
        "img_size": img_size,
    }
    if weights_file and pretrained:
        kwargs["pretrained_cfg_overlay"] = {"file": weights_file}
    model = _instantiate(timm_name, **kwargs)

    head = model.get_classifier()
    if not isinstance(head, nn.Module):
        raise TypeError(f"timm model {timm_name!r} did not expose a valid classifier module")
    _initialize_classifier(head)

    if strategy in ("feature_extraction", "gradual_unfreeze"):
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in head.parameters():
            parameter.requires_grad = True
    else:
        for parameter in model.parameters():
            parameter.requires_grad = True
    return model


def param_groups(model: nn.Module, head_lr: float, backbone_lr: float) -> list[dict[str, Any]]:
    """Discriminative LR groups for the head and all other trainable weights."""
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    head = [p for p in model.parameters() if id(p) in head_ids and p.requires_grad]
    body = [p for p in model.parameters() if id(p) not in head_ids and p.requires_grad]
    if not head and not body:
        raise ValueError("No trainable parameters found. Check the selected strategy.")
    groups: list[dict[str, Any]] = []
    if head:
        groups.append({"params": head, "lr": head_lr})
    if body:
        groups.append({"params": body, "lr": backbone_lr})
    return groups


def layer_decay_param_groups(
    model: nn.Module,
    head_lr: float,
    backbone_lr: float,
    layer_decay: float = 0.75,
    weight_decay: float = 0.05,
) -> list[dict[str, Any]]:
    """Use timm's layer-wise LR decay, with a safe flat-LR fallback."""
    try:
        from timm.optim import param_groups_layer_decay

        groups = param_groups_layer_decay(
            model, weight_decay=weight_decay, layer_decay=layer_decay
        )
    except (ImportError, AttributeError, TypeError, RuntimeError, ValueError):
        return param_groups(model, head_lr, backbone_lr)

    head_ids = {id(p) for p in model.get_classifier().parameters()}
    out: list[dict[str, Any]] = []
    for group in groups:
        params = [p for p in group["params"] if id(p) not in head_ids and p.requires_grad]
        if params:
            out.append(
                {
                    "params": params,
                    "lr": backbone_lr * group.get("lr_scale", 1.0),
                    "weight_decay": group.get("weight_decay", weight_decay),
                }
            )

    head_params = [p for p in model.get_classifier().parameters() if p.requires_grad]
    if head_params:
        out.append({"params": head_params, "lr": head_lr, "weight_decay": 0.0})
    if not out:
        return param_groups(model, head_lr, backbone_lr)
    return out


def backbone_stage_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Return top-level parameterized modules, excluding the classifier head."""
    head_param_ids = {id(p) for p in model.get_classifier().parameters()}
    stages = []
    for name, module in model.named_children():
        module_param_ids = {id(p) for p in module.parameters()}
        if module_param_ids and module_param_ids <= head_param_ids:
            continue
        if module_param_ids:
            stages.append((name, module))
    return stages


def set_unfrozen_stages(model: nn.Module, n_stages_from_head: int) -> None:
    """Freeze all weights, then unfreeze the head and the last N top-level stages.

    ``n_stages_from_head=0`` is head-only. ``-1`` is intentionally handled by
    the caller as a request to unfreeze every parameter.
    """
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.get_classifier().parameters():
        parameter.requires_grad = True
    if n_stages_from_head <= 0:
        return
    stages = backbone_stage_modules(model)
    for _, module in stages[max(0, len(stages) - n_stages_from_head) :]:
        for parameter in module.parameters():
            parameter.requires_grad = True


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
