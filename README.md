# Waste Sorting Image Classification

6-class waste classifier (cardboard, glass, metal, paper, plastic, trash) on [TrashNet](https://github.com/garythung/trashnet),
comparing two ImageNet-pretrained CNN backbones (EfficientNet-B0, ResNet-50) and two fine-tuning strategies
(feature extraction vs. full fine-tuning).

Team: Svetlana Maltseva, Valeria Kolpakova, Alexey Feofanov, Vadim Gafarov 

## Layout

```
configs/                 one YAML per experiment (baseline: baseline_effb0_fe.yaml)
scripts/prepare_data.py  hash, deduplicate and split the dataset -> data/splits.csv
src/data.py              Dataset, transforms, stratified split, class weights
src/models.py            timm backbone + new head, freezing, LR groups
src/train.py             training loop (AdamW, warm-up + cosine, early stopping on val macro-F1)
src/evaluate.py          test metrics, confusion matrix, misclassified list, latency benchmark
runs/<name>_seed<k>/     checkpoints and results (git-ignored except small json/png)
```

## Quick start on Linux or MacOS

```bash
pip install -r requirements.txt

# 1. data 
mkdir -p data/raw && cd data/raw
curl -L -o ds.zip https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip
unzip -q ds.zip && rm -rf __MACOSX ds.zip && cd ../..
python scripts/prepare_data.py --root data/raw/dataset-resized

# 2. ImageNet weights 
mkdir -p weights
curl -L -o weights/efficientnet_b0_ra-3dd342df.pth https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b0_ra-3dd342df.pth

# 3. baseline
python -m src.train    --config configs/baseline_effb0_fe.yaml
python -m src.evaluate --config configs/baseline_effb0_fe.yaml
```
## **Alternative quick start**
If you have **pixi** installed on your system,

1. Downloand and unzip dataset:
```bash
curl -L -o ds.zip https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip
```
Create necessary folders and place it into  ./data/raw/dataset-resized

2. Download weights and place them into  ./weights folder

```bash
curl -L -o weights/efficientnet_b0_ra-3dd342df.pth https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b0_ra-3dd342df.pth
```

3. Install dependencies, prepare data and run the baseline:
```bash
pixi install
pixi run prepare
pixi run train
pixi run evaluate
```

## Improvements: higher accuracy and faster inference

| Change | Where | Expected effect |
|---|---|---|
| Full fine-tuning of EfficientNet-B0 (all layers, backbone LR 10x lower than the head, label smoothing 0.1, weight decay 0.01) | `configs/effb0_finetune.yaml` | Higher accuracy: the backbone adapts to TrashNet textures (glass vs. plastic) instead of using fixed ImageNet features |
| ResNet-50, frozen and fine-tuned | `configs/resnet50_fe.yaml`, `configs/resnet50_finetune.yaml` | Second backbone from the project plan |
| Mixed precision (fp16) and channels-last memory layout on GPU | `amp`, `channels_last` in the configs | Faster training and GPU inference, lower GPU memory |
| Persistent data-loading workers, cuDNN autotuning | `src/data.py`, `src/train.py` | Less time between epochs |
| Test-time augmentation (image + mirror) | `python -m src.evaluate ... --tta` | Small accuracy gain without retraining |
| ONNX export + ONNX Runtime | `python -m src.export ...` | Faster CPU inference (about 3-4x in our test) |
| Prediction on your own photos | `python -m src.predict ...` | Classify any image or folder, prints top-2 classes and time |
| Comparison table of all runs | `python scripts/compare_runs.py` | One table with accuracy, macro-F1, per-class F1 and speed |

Extra weights for ResNet-50:

```bash
curl -L -o weights/resnet50_a1_0-14fe96d1.pth https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1_0-14fe96d1.pth
```

Run the experiments (GPU recommended; every command also works on CPU):

```bash
# quick check that everything runs (2 tiny epochs, about a minute)
python -m src.train --config configs/effb0_finetune.yaml --epochs 2 --limit-batches 5

# full runs
python -m src.train    --config configs/effb0_finetune.yaml
python -m src.evaluate --config configs/effb0_finetune.yaml
python -m src.evaluate --config configs/effb0_finetune.yaml --tta
python -m src.train    --config configs/resnet50_fe.yaml
python -m src.evaluate --config configs/resnet50_fe.yaml
python -m src.train    --config configs/resnet50_finetune.yaml
python -m src.evaluate --config configs/resnet50_finetune.yaml

# faster CPU inference and your own photos
python -m src.export  --config configs/effb0_finetune.yaml
python -m src.predict --config configs/effb0_finetune.yaml path/to/photo.jpg --onnx

# compare everything
python scripts/compare_runs.py --csv runs/comparison.csv
```

Run the quick check first: if it fails, the full runs fail the same way. Each run writes its
results to `runs/<name>_seed<seed>/`; `--seed N` repeats a run with another seed.
If memory runs out or Windows reports `Access is denied` for data-loader workers,
lower `num_workers` in the config (0 is always safe, only slower).

## Status

- [x] Data pipeline (deduplication, stratified 70/15/15 split, augmentation)
- [x] Baseline: EfficientNet-B0, frozen backbone (E-FE) — test accuracy 87.1%, macro-F1 0.847 (GPU run, seed 6545656)
- [ ] E-FT, R-FE, R-FT runs (configs ready: `effb0_finetune`, `resnet50_fe`, `resnet50_finetune`)
- [x] TTA, ONNX export, prediction script, run comparison table
- [ ] Grad-CAM failure-case analysis

## Web interface

The browser interface uploads an image to a local API endpoint and displays the
top prediction and alternatives. Install the extra dependencies and start it
from the repository root:

```bash
pip install -r requirements.txt
uvicorn web.app:app --reload
```

Open `http://127.0.0.1:8000`. The default checkpoint path is
`runs/effb0_feature_extraction_seed87578/best.pt`. Point the site at another
trained model by setting `WASTE_CONFIG` and `WASTE_CHECKPOINT` before starting
Uvicorn (PowerShell example):

```powershell
$env:WASTE_CONFIG = "configs/effb0_finetune.yaml"
$env:WASTE_CHECKPOINT = "runs/effb0_finetune_seed87578/best.pt"
uvicorn web.app:app --reload
```
