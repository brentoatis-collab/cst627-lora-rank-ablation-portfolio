#!/usr/bin/env python3
import argparse
import csv
import json
import time
from copy import deepcopy
from pathlib import Path

import torch
import yaml

from src.data import generate_dataset
from src.experiment import adapt_and_evaluate, capture_environment, pretrain
from src.reproducibility import resolve_path


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Controlled LoRA-rank ablation for demand forecasting")
    parser.add_argument("--config", default="configs/ablation.yaml")
    parser.add_argument("--mode", choices=["pretrain", "single", "all", "aggregate"], default="all")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seeds", choices=["3", "5"], default="3")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def smoke_config(config):
    config = deepcopy(config)
    config["data"].update({"hours": 720, "source_coops": 1})
    config["model"].update({"d_model": 32, "n_heads": 4, "n_layers": 1, "d_ff": 64})
    config["pretraining"].update({"epochs": 1, "batch_size": 64})
    config["adaptation"].update({"epochs": 1, "batch_size": 64})
    config["data"]["path"] = "artifacts/smoke/data/synthetic_demand.npz"
    config["data"]["metadata_path"] = "artifacts/smoke/data/synthetic_demand_metadata.json"
    config["pretraining"]["checkpoint"] = "artifacts/smoke/checkpoints/shared_backbone.pt"
    config["project"]["output_dir"] = "artifacts/smoke"
    return config


def aggregate(results_dir: Path, plots_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    records = []
    for path in sorted(results_dir.glob("rank*_seed*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.append({key: payload[key] for key in (
            "rank", "seed", "mae_kw", "rmse_kw", "mape_percent", "training_seconds", "evaluation_seconds",
            "adaptation_and_evaluation_seconds",
            "peak_gpu_memory_mb", "trainable_parameters", "device")})
    if not records:
        print("No completed result files to aggregate.")
        return
    fieldnames = list(records[0])
    with (results_dir / "seed_level_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(records)
    summary = []
    for rank in sorted({row["rank"] for row in records}):
        rows = [row for row in records if row["rank"] == rank]
        values = np.array([row["mape_percent"] for row in rows])
        summary.append({"rank": rank, "n_seeds": len(rows), "mean_mape_percent": values.mean(),
                        "std_mape_percent": values.std(ddof=1) if len(values) > 1 else float("nan"),
                        "min_mape_percent": values.min(), "max_mape_percent": values.max(),
                        "mean_training_seconds": np.mean([row["training_seconds"] for row in rows]),
                        "mean_run_seconds": np.mean([row["adaptation_and_evaluation_seconds"] for row in rows]),
                        "mean_peak_gpu_memory_mb": np.mean([row["peak_gpu_memory_mb"] for row in rows])
                        if all(row["peak_gpu_memory_mb"] is not None for row in rows) else float("nan")})
    with (results_dir / "summary_by_rank.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for rank in sorted({row["rank"] for row in records}):
        rows = [row for row in records if row["rank"] == rank]
        axis.scatter([rank] * len(rows), [row["mape_percent"] for row in rows], s=55, label=f"rank {rank}")
    axis.set(xlabel="LoRA rank", ylabel="Test MAPE (%)", title="Seed-level LoRA rank ablation")
    axis.grid(alpha=0.25); fig.tight_layout(); fig.savefig(plots_dir / "seed_level_mape.png", dpi=180); plt.close(fig)
    print(f"Aggregated {len(records)} runs into {results_dir}")


def main():
    args = parse_args()
    config_path = resolve_path(ROOT, args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.smoke:
        config = smoke_config(config)
    output_dir = resolve_path(ROOT, config["project"]["output_dir"])
    data_path = resolve_path(ROOT, config["data"]["path"])
    metadata_path = resolve_path(ROOT, config["data"]["metadata_path"])
    checkpoint_path = resolve_path(ROOT, config["pretraining"]["checkpoint"])
    results_dir, plots_dir = output_dir / "results", output_dir / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Aggregation is intentionally evidence-preserving: it rebuilds only the
    # CSV summaries and plot from committed per-run JSON records. It must not
    # capture the reviewer's environment, generate data, or replace benchmark
    # records from the confirmatory GPU session.
    if args.mode == "aggregate":
        aggregate(results_dir, plots_dir)
        return

    capture_environment(output_dir / "environment" / "environment.json")
    if not data_path.exists():
        metadata = generate_dataset(config["data"], data_path, metadata_path)
        print(f"Generated dataset: {data_path} (sha256={metadata['sha256']})")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    pretrain_seconds = 0.0
    if args.mode in {"pretrain", "all", "single"} and (not checkpoint_path.exists() or args.smoke):
        info = pretrain(config, data_path, checkpoint_path, device)
        pretrain_seconds = info["pretrain_seconds"]
        (checkpoint_path.parent / "pretraining_log.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        print(f"Pretraining complete in {info['pretrain_seconds']:.2f}s on {device}")
    if args.mode == "pretrain":
        return
    if args.mode == "single":
        if args.rank not in config["model"]["ranks"] or args.seed is None:
            raise SystemExit("--mode single requires --rank from config and --seed")
        pairs = [(args.rank, args.seed)]
    elif args.mode == "all":
        seeds = list(config["reproducibility"]["study_seeds"])
        if args.seeds == "5":
            seeds += list(config["reproducibility"]["optional_extended_seeds"])
        if args.smoke:
            seeds = [seeds[0]]
        pairs = [(rank, seed) for rank in config["model"]["ranks"] for seed in seeds]
    for rank, seed in pairs:
        destination = results_dir / f"rank{rank}_seed{seed}.json"
        if destination.exists() and not args.smoke:
            print(f"Skipping existing {destination.name}")
            continue
        result = adapt_and_evaluate(config, data_path, checkpoint_path, rank, seed, device)
        destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"rank={rank} seed={seed} engineering result: MAPE={result['mape_percent']:.3f}% time={result['training_seconds']:.2f}s")
    aggregate(results_dir, plots_dir)
    benchmark = {"mode": args.mode, "smoke": args.smoke, "wall_clock_seconds": time.perf_counter() - started,
                 "pretrain_seconds": pretrain_seconds, "device": str(device), "completed_runs": len(pairs)}
    (output_dir / "benchmark.json").write_text(json.dumps(benchmark, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
