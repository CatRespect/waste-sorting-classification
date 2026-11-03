"""Aggregate validation/test artifacts into a CSV and a compact Markdown report.

Usage:
    python scripts/compare_runs.py --output-csv reports/comparison.csv --output-md reports/report.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd



def _read_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError):
        return None


def collect_rows(runs_dir: Path) -> list[dict]:
    rows = []
    for test_path in sorted(runs_dir.glob("**/test_metrics*.json")):
        test_metrics = _read_json(test_path)
        if not test_metrics:
            continue
        run_dir = test_path.parent
        val_path = run_dir / "val_metrics.json"
        val_metrics = _read_json(val_path) or {}
        train_log = _read_json(run_dir / "train_log.json") or {}
        row = {
            "config": test_metrics.get("config_name", run_dir.name),
            "backbone": test_metrics.get("backbone", ""),
            "strategy": test_metrics.get("strategy", ""),
            "seed": test_metrics.get("seed", ""),
            "tta": bool(test_metrics.get("tta", False)),
            "val_accuracy": val_metrics.get("accuracy"),
            "val_macro_f1": val_metrics.get("macro_f1"),
            "test_accuracy": test_metrics.get("accuracy"),
            "test_balanced_accuracy": test_metrics.get("balanced_accuracy"),
            "test_macro_f1": test_metrics.get("macro_f1"),
            "n_test": test_metrics.get("n_samples"),
            "n_misclassified": test_metrics.get("n_misclassified"),
            "train_seconds": train_log.get("train_seconds"),
            "device": test_metrics.get("device", ""),
        }
        for class_name, class_metrics in test_metrics.get("per_class", {}).items():
            row[f"f1_{class_name}"] = class_metrics.get("f1-score")
            row[f"precision_{class_name}"] = class_metrics.get("precision")
            row[f"recall_{class_name}"] = class_metrics.get("recall")
        benchmark = test_metrics.get("benchmark") or []
        if benchmark:
            row["latency_ms_per_image_bs1"] = next(
                (item.get("ms_per_image") for item in benchmark if item.get("batch") == 1), None
            )
            row["throughput_img_s"] = next(
                (item.get("img_per_sec") for item in benchmark if item.get("batch", 0) > 1), None
            )
        rows.append(row)
    return rows


def write_markdown(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        path.write_text("# Experiment report\n\nNo test metrics were found.\n", encoding="utf-8")
        return
    standard = df[~df.tta].copy() if "tta" in df else df.copy()
    if standard.empty:
        standard = df.copy()
    standard = standard.sort_values("test_macro_f1", ascending=False)
    fields = [
        "config", "backbone", "strategy", "seed", "val_macro_f1",
        "test_accuracy", "test_macro_f1", "latency_ms_per_image_bs1",
    ]
    fields = [field for field in fields if field in standard.columns]
    lines = [
        "# Waste Sorting — experiment report",
        "",
        "This report is generated from `runs/<run>/val_metrics.json`, `test_metrics.json`, `train_log.json`.",
        "Validation metrics are used for model selection; test metrics are for the final held-out comparison.",
        "",
        "## Test-set comparison",
        "",
        standard[fields].to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    best = standard.iloc[0]
    lines.extend(
        [
            "## Highest test macro-F1 (descriptive)",
            "",
            f"- Run: `{best['config']}` (seed `{best['seed']}`)",
            f"- Backbone / strategy: `{best['backbone']}` / `{best['strategy']}`",
            f"- Test accuracy: {best['test_accuracy']:.4f}",
            f"- Test macro-F1: {best['test_macro_f1']:.4f}",
            "",
            "## Files",
            "",
            "- `comparison.csv` — complete tabular metrics, including per-class precision/recall/F1 and TTA runs.",
            "- Each run folder — confusion matrices, misclassified examples, model checkpoints, training history and metrics JSON.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an experiment comparison table/report")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--output-csv", default="reports/comparison.csv")
    parser.add_argument("--output-md", default="reports/report.md")
    args = parser.parse_args()

    rows = collect_rows(Path(args.runs_dir))
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["test_macro_f1", "config"], ascending=[False, True])
    df.to_csv(output_csv, index=False)
    write_markdown(df, output_md)
    print(f"Wrote {len(df)} run(s) to {output_csv} and {output_md}")


if __name__ == "__main__":
    main()
