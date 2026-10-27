import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset


def _coop_trace(hours: int, rng: np.random.Generator, coop_id: int, target: bool) -> np.ndarray:
    t = np.arange(hours, dtype=np.float64)
    hour = t % 24
    weekday = (t // 24) % 7
    year_phase = 2 * np.pi * t / (24 * 365.25)
    phase = 0.35 * coop_id + (0.25 if target else 0.0)
    temperature = (
        12.0
        + 14.0 * np.sin(year_phase - 1.2 + phase * 0.1)
        + 3.0 * np.sin(2 * np.pi * hour / 24 + 0.5)
        + rng.normal(0, 1.8, hours)
    )
    daily = 18.0 * np.sin(2 * np.pi * (hour - 7) / 24) + 8.0 * np.sin(4 * np.pi * (hour - 4) / 24)
    weekend = np.where(weekday >= 5, -10.0, 0.0)
    heating = np.maximum(12.0 - temperature, 0.0) * (2.1 + 0.12 * coop_id)
    cooling = np.maximum(temperature - 22.0, 0.0) * (1.7 + (0.25 if target else 0.0))
    trend = 0.0008 * t
    innovations = rng.normal(0, 3.5, hours)
    ar_noise = np.zeros(hours, dtype=np.float64)
    for idx in range(1, hours):
        ar_noise[idx] = 0.72 * ar_noise[idx - 1] + innovations[idx]
    base = 118.0 + 6.0 * coop_id + (9.0 if target else 0.0)
    load = np.maximum(base + daily + weekend + heating + cooling + trend + ar_noise, 5.0)
    return np.column_stack(
        [load, temperature, np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)]
    ).astype(np.float32)


def _windows(series: np.ndarray, lookback: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    count = len(series) - lookback - horizon + 1
    x = np.empty((count, lookback, series.shape[1]), dtype=np.float32)
    y = np.empty((count, horizon), dtype=np.float32)
    for idx in range(count):
        x[idx] = series[idx : idx + lookback]
        y[idx] = series[idx + lookback : idx + lookback + horizon, 0]
    return x, y


def generate_dataset(config: Dict, data_path: Path, metadata_path: Path) -> Dict:
    seed = int(config["generation_seed"])
    hours = int(config["hours"])
    lookback = int(config["lookback"])
    horizon = int(config["horizon"])
    source_arrays = []
    for coop_id in range(int(config["source_coops"])):
        rng = np.random.default_rng(seed + coop_id)
        source_arrays.append(_windows(_coop_trace(hours, rng, coop_id, False), lookback, horizon))
    source_x = np.concatenate([item[0] for item in source_arrays])
    source_y = np.concatenate([item[1] for item in source_arrays])

    target_rng = np.random.default_rng(seed + 10_000)
    target_x, target_y = _windows(_coop_trace(hours, target_rng, 3, True), lookback, horizon)
    n = len(target_x)
    train_end = int(n * float(config["train_fraction"]))
    val_end = int(n * (float(config["train_fraction"]) + float(config["validation_fraction"])))

    feature_mean = target_x[:train_end].reshape(-1, target_x.shape[-1]).mean(axis=0)
    feature_std = target_x[:train_end].reshape(-1, target_x.shape[-1]).std(axis=0)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
    target_mean = float(target_y[:train_end].mean())
    target_std = float(target_y[:train_end].std())

    normalize_x = lambda value: ((value - feature_mean) / feature_std).astype(np.float32)
    normalize_y = lambda value: ((value - target_mean) / target_std).astype(np.float32)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        data_path,
        source_x=normalize_x(source_x),
        source_y=normalize_y(source_y),
        train_x=normalize_x(target_x[:train_end]),
        train_y=normalize_y(target_y[:train_end]),
        val_x=normalize_x(target_x[train_end:val_end]),
        val_y=normalize_y(target_y[train_end:val_end]),
        test_x=normalize_x(target_x[val_end:]),
        test_y=normalize_y(target_y[val_end:]),
        feature_mean=feature_mean.astype(np.float32),
        feature_std=feature_std.astype(np.float32),
        target_mean=np.array(target_mean, dtype=np.float32),
        target_std=np.array(target_std, dtype=np.float32),
    )
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    metadata = {
        "generator": "src/data.py",
        "generation_seed": seed,
        "hours_per_coop": hours,
        "source_coops": int(config["source_coops"]),
        "target_coops": 1,
        "lookback_hours": lookback,
        "forecast_horizon_hours": horizon,
        "chronological_split": {"train_end": train_end, "validation_end": val_end, "total_windows": n},
        "features": ["demand_kw", "temperature_c", "hour_sin", "hour_cos"],
        "sha256": digest,
        "important_limit": "Synthetic traces reflect programmed seasonality and noise, not observed customer behavior.",
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def load_datasets(data_path: Path) -> Tuple[Dict[str, TensorDataset], Dict[str, float]]:
    loaded = np.load(data_path)
    datasets = {}
    for split in ("source", "train", "val", "test"):
        datasets[split] = TensorDataset(
            torch.from_numpy(loaded[f"{split}_x"]), torch.from_numpy(loaded[f"{split}_y"])
        )
    scaling = {"target_mean": float(loaded["target_mean"]), "target_std": float(loaded["target_std"])}
    return datasets, scaling
