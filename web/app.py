from __future__ import annotations

import io
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from src.data import CLASSES, get_transforms
from src.models import build_model

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("WASTE_CONFIG", ROOT / "configs/clip_vitb32_full_finetune.yaml"))
CHECKPOINT_PATH = Path(
    os.environ.get(
        "WASTE_CHECKPOINT", ROOT / "runs/clip_vitb32_full_finetune_seed87578/best.pt"
    )
)
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = FastAPI(title="Material recognition classifier")
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")


@lru_cache
def load_predictor():
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Configuration was not found: {CONFIG_PATH}")
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Model checkpoint was not found: {CHECKPOINT_PATH}. "
            "Train a model first or set WASTE_CHECKPOINT."
        )

    with CONFIG_PATH.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        config["backbone"], len(CLASSES), config["strategy"], config["dropout"],
        img_size=config.get("img_size", 224), pretrained=False,
    )
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device).eval()
    return model, get_transforms(train=False, size=config.get("img_size", 224)), device


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "static/index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "checkpoint": str(CHECKPOINT_PATH), "ready": CHECKPOINT_PATH.is_file()}


@app.post("/api/predict")
async def predict(image: UploadFile = File(...)):
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Upload a JPG, PNG, WEBP, or BMP image.")
    raw = await image.read()
    if not raw:
        raise HTTPException(400, "The uploaded image is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image exceeds the 10 MB limit.")
    try:
        source = Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(400, "The uploaded file is not a valid image.") from exc

    try:
        model, transform, device = load_predictor()
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc

    tensor = transform(source).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(tensor).softmax(dim=1)[0].cpu().numpy()
    top_indices = np.argsort(-probabilities)[:3]
    predictions = [
        {
            "label": CLASSES[int(index)],
            "class_name": CLASSES[int(index)],
            "confidence": round(float(probabilities[index]) * 100, 1),
        }
        for index in top_indices
    ]
    return {"prediction": predictions[0], "alternatives": predictions[1:]}
