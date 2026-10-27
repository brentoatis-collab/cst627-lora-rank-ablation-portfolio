import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import load_datasets
from .model import ForecastTransformer, freeze_for_lora, trainable_parameter_count
from .reproducibility import seed_everything


def make_model(config: Dict, rank: int) -> ForecastTransformer:
    model = config["model"]
    return ForecastTransformer(
        input_features=int(model["input_features"]), horizon=int(config["data"]["horizon"]),
        d_model=int(model["d_model"]), n_heads=int(model["n_heads"]), n_layers=int(model["n_layers"]),
        d_ff=int(model["d_ff"]), max_sequence_length=int(model["max_sequence_length"]),
        dropout=float(model["dropout"]), rank=rank, lora_dropout=float(model["lora_dropout"]),
    )


def _loader(dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=generator)


def _run_epochs(model, train_data, valid_data, settings, device, seed) -> Dict:
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    loss_fn = nn.MSELoss()
    history = []
    for epoch in range(int(settings["epochs"])):
        model.train()
        total, count = 0.0, 0
        for x, y in _loader(train_data, int(settings["batch_size"]), True, seed + epoch):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            if "gradient_clip_norm" in settings:
                nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), float(settings["gradient_clip_norm"]))
            optimizer.step()
            total += float(loss.detach()) * len(x)
            count += len(x)
        model.eval()
        valid_total, valid_count = 0.0, 0
        with torch.no_grad():
            for x, y in _loader(valid_data, int(settings["batch_size"]), False, seed):
                x, y = x.to(device), y.to(device)
                valid_total += float(loss_fn(model(x), y)) * len(x)
                valid_count += len(x)
        history.append({"epoch": epoch + 1, "train_mse_scaled": total / count, "validation_mse_scaled": valid_total / valid_count})
    return {"history": history}


def _metrics(model, dataset, batch_size, device, scaling, epsilon_kw) -> Dict:
    predictions, targets = [], []
    model.eval()
    with torch.no_grad():
        for x, y in _loader(dataset, batch_size, False, 0):
            predictions.append(model(x.to(device)).cpu().numpy())
            targets.append(y.numpy())
    pred = np.concatenate(predictions) * scaling["target_std"] + scaling["target_mean"]
    truth = np.concatenate(targets) * scaling["target_std"] + scaling["target_mean"]
    error = pred - truth
    return {
        "mae_kw": float(np.mean(np.abs(error))),
        "rmse_kw": float(np.sqrt(np.mean(np.square(error)))),
        "mape_percent": float(np.mean(np.abs(error) / np.maximum(np.abs(truth), epsilon_kw)) * 100),
    }


def pretrain(config: Dict, data_path: Path, checkpoint_path: Path, device: torch.device) -> Dict:
    seed = int(config["reproducibility"]["pretrain_seed"])
    seed_everything(seed, bool(config["reproducibility"]["deterministic_algorithms"]), bool(config["reproducibility"]["warn_only"]))
    datasets, _ = load_datasets(data_path)
    model = make_model(config, rank=4).to(device)
    # Pretraining learns the full shared model; adapter tensors are excluded and stripped from the checkpoint.
    for name, parameter in model.named_parameters():
        parameter.requires_grad = ".lora_" not in name
    started = time.perf_counter()
    outcome = _run_epochs(model, datasets["source"], datasets["val"], config["pretraining"], device, seed)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    base_state = {key: value.cpu() for key, value in model.state_dict().items() if ".lora_" not in key}
    torch.save({"base_state_dict": base_state, "pretrain_seed": seed, "config_project": config["project"]["name"]}, checkpoint_path)
    return {"pretrain_seconds": time.perf_counter() - started, **outcome}


def adapt_and_evaluate(config: Dict, data_path: Path, checkpoint_path: Path, rank: int, seed: int,
                       device: torch.device) -> Dict:
    seed_everything(seed, bool(config["reproducibility"]["deterministic_algorithms"]), bool(config["reproducibility"]["warn_only"]))
    datasets, scaling = load_datasets(data_path)
    model = make_model(config, rank).to(device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(payload["base_state_dict"], strict=False)
    if unexpected or any("lora_" not in key for key in missing):
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    freeze_for_lora(model)
    # Rank changes how many adapter values are initialized. Reset afterward so
    # dropout and other training-time RNG streams remain paired across arms.
    seed_everything(seed, bool(config["reproducibility"]["deterministic_algorithms"]), bool(config["reproducibility"]["warn_only"]))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    run_started = time.perf_counter()
    started = time.perf_counter()
    training = _run_epochs(model, datasets["train"], datasets["val"], config["adaptation"], device, seed)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    evaluation_started = time.perf_counter()
    metrics = _metrics(model, datasets["test"], int(config["adaptation"]["batch_size"]), device, scaling,
                       float(config["evaluation"]["epsilon_kw"]))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    evaluation_elapsed = time.perf_counter() - evaluation_started
    peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else None
    return {
        "rank": rank, "seed": seed, **metrics, "training_seconds": elapsed,
        "evaluation_seconds": evaluation_elapsed,
        "adaptation_and_evaluation_seconds": time.perf_counter() - run_started,
        "peak_gpu_memory_mb": peak_mb, "device": str(device),
        "trainable_parameters": trainable_parameter_count(model), **training,
    }


def capture_environment(destination: Path) -> None:
    gpu = None
    gpu_vram_mib = None
    driver_version = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        gpu_vram_mib = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2))
        try:
            driver_version = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=False,
                capture_output=True,
                text=True,
            ).stdout.strip().splitlines()[0]
        except (OSError, IndexError):
            driver_version = None
    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=False, capture_output=True, text=True).stdout.splitlines()
    except OSError:
        freeze = []
    record = {
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "gpu": gpu, "gpu_vram_mib": gpu_vram_mib,
        "driver_version": driver_version, "pip_freeze": freeze,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
