"""Run train -> validation -> test evaluation for every YAML config.

Examples:
    python scripts/run_all.py
    python scripts/run_all.py --seeds 42 2026
    python scripts/run_all.py --epochs 2 --limit-batches 5 --dry-run
    python scripts/run_all.py --force --tta

Existing best checkpoints are reused by default. Use --force to retrain them.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def read_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream) or {}
    for key in ("name", "seed", "output_dir"):
        if key not in cfg:
            raise ValueError(f"{path} is missing required key {key!r}")
    return cfg


def run_command(command: list[str], dry_run: bool) -> tuple[int, float]:
    print("\n$ " + " ".join(str(part) for part in command), flush=True)
    if dry_run:
        return 0, 0.0
    started = time.time()
    completed = subprocess.run(command, cwd=ROOT, check=False)
    return completed.returncode, time.time() - started


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all experiment configs end-to-end")
    parser.add_argument("--configs", nargs="*", default=None, help="Optional YAML paths; default: all configs/*.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Override seeds for every config")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count (useful for smoke tests)")
    parser.add_argument("--limit-batches", type=int, default=None, help="Limit training batches per epoch")
    parser.add_argument("--force", action="store_true", help="Retrain even when best.pt already exists")
    parser.add_argument("--tta", action="store_true", help="Also run test-time augmentation evaluation")
    parser.add_argument("--no-benchmark", action="store_true", help="Skip test inference latency benchmark")
    parser.add_argument("--benchmark-iters", type=int, default=20)
    parser.add_argument("--stop-on-error", action="store_true", help="Stop at the first failed step")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    parser.add_argument("--max-configs", type=int, default=None, help="Only run the first N configs")
    args = parser.parse_args()

    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if args.limit_batches is not None and args.limit_batches < 1:
        parser.error("--limit-batches must be >= 1")

    if args.configs:
        config_paths = [Path(path) if Path(path).is_absolute() else ROOT / path for path in args.configs]
    else:
        config_paths = sorted((ROOT / "configs").glob("*.yaml"))
    config_paths = [path for path in config_paths if path.is_file()]
    if args.max_configs is not None:
        config_paths = config_paths[: max(0, args.max_configs)]
    if not config_paths:
        raise FileNotFoundError("No YAML configs found. Expected one or more files in configs/.")

    suite_started = time.time()
    results = []
    had_error = False
    for config_path in config_paths:
        cfg = read_config(config_path)
        seeds = args.seeds if args.seeds is not None else [int(cfg["seed"])]
        for seed in seeds:
            run_dir = ROOT / cfg["output_dir"] / f"{cfg['name']}_seed{seed}"
            checkpoint = run_dir / "best.pt"
            run_result = {
                "config": str(config_path.relative_to(ROOT)),
                "name": cfg["name"],
                "seed": seed,
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "train": "skipped_existing" if checkpoint.exists() and not args.force else "pending",
                "validate": "pending",
                "evaluate": "pending",
                "tta": "not_requested",
            }

            if not (checkpoint.exists() and not args.force):
                train_cmd = [sys.executable, "-m", "src.train", "--config", str(config_path.relative_to(ROOT)), "--seed", str(seed)]
                if args.epochs is not None:
                    train_cmd += ["--epochs", str(args.epochs)]
                if args.limit_batches is not None:
                    train_cmd += ["--limit-batches", str(args.limit_batches)]
                code, duration = run_command(train_cmd, args.dry_run)
                run_result["train"] = "ok" if code == 0 else f"failed:{code}"
                run_result["train_seconds"] = round(duration, 2)
                if code != 0:
                    had_error = True
                    results.append(run_result)
                    if args.stop_on_error:
                        break
                    continue

            validate_cmd = [
                sys.executable, "-m", "src.evaluate", "--config",
                str(config_path.relative_to(ROOT)), "--seed", str(seed), "--split", "val", "--no-benchmark",
            ]
            code, duration = run_command(validate_cmd, args.dry_run)
            run_result["validate"] = "ok" if code == 0 else f"failed:{code}"
            run_result["validate_seconds"] = round(duration, 2)
            if code != 0:
                had_error = True
                results.append(run_result)
                if args.stop_on_error:
                    break
                continue

            evaluate_cmd = [
                sys.executable, "-m", "src.evaluate", "--config",
                str(config_path.relative_to(ROOT)), "--seed", str(seed), "--split", "test",
                "--benchmark-iters", str(args.benchmark_iters),
            ]
            if args.no_benchmark:
                evaluate_cmd.append("--no-benchmark")
            code, duration = run_command(evaluate_cmd, args.dry_run)
            run_result["evaluate"] = "ok" if code == 0 else f"failed:{code}"
            run_result["evaluate_seconds"] = round(duration, 2)
            if code != 0:
                had_error = True
                results.append(run_result)
                if args.stop_on_error:
                    break
                continue

            if args.tta:
                tta_cmd = evaluate_cmd + ["--tta"]
                code, duration = run_command(tta_cmd, args.dry_run)
                run_result["tta"] = "ok" if code == 0 else f"failed:{code}"
                run_result["tta_seconds"] = round(duration, 2)
                if code != 0:
                    had_error = True

            results.append(run_result)
            if had_error and args.stop_on_error:
                break
        if had_error and args.stop_on_error:
            break

    report_dir = ROOT / "runs" / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    suite_summary = {
        "started_at_unix": suite_started,
        "finished_at_unix": time.time(),
        "duration_seconds": round(time.time() - suite_started, 2),
        "dry_run": args.dry_run,
        "configs": [str(path.relative_to(ROOT)) for path in config_paths],
        "results": results,
    }
    summary_path = report_dir / "run_all_summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(suite_summary, stream, ensure_ascii=False, indent=2)
    print(f"\nExperiment orchestration summary: {summary_path.relative_to(ROOT)}")

    if not args.dry_run:
        compare_cmd = [
            sys.executable, "scripts/compare_runs.py",
            "--runs-dir", "runs",
            "--output-csv", "runs/report/comparison.csv",
            "--output-md", "runs/report/report.md",
        ]
        compare_code, _ = run_command(compare_cmd, dry_run=False)
        if compare_code != 0:
            had_error = True
        print("Report files: runs/report/comparison.csv, runs/report/report.md")

    if had_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
