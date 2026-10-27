# Confirmatory results

The confirmatory study completed 15 runs on October 22, 2026: five seeds for each LoRA rank of 4, 8, and 16. All runs used the same synthetic dataset, shared pretrained backbone, training configuration, Tesla T4 GPU class, and evaluation procedure.

| Rank | Seeds | Mean MAPE +/- sample SD | Min-max MAPE | Mean MAE | Mean RMSE | Trainable parameters | Peak GPU memory | Adapt + eval mean +/- SD |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 5 | 3.2396% +/- 0.0381 | 3.1860-3.2931% | 5.8105 kW | 7.3270 kW | 2,048 | 123.02 MB | 14.19 +/- 0.14 s |
| 8 | 5 | 3.2487% +/- 0.0466 | 3.1854-3.3143% | 5.8289 kW | 7.3522 kW | 4,096 | 123.42 MB | 13.94 +/- 0.23 s |
| 16 | 5 | 3.2669% +/- 0.0578 | 3.1825-3.3333% | 5.8638 kW | 7.3951 kW | 8,192 | 124.21 MB | 13.94 +/- 0.13 s |

Rank 4 produced the lowest mean MAPE, but the mean differences were only 0.0092 percentage points relative to rank 8 and 0.0274 points relative to rank 16. All three seed-level ranges overlap. Under the preregistered decision rule, the study therefore does not establish that any rank is more accurate than another beyond run-to-run variation.

The operational recommendation is to retain rank 4 for this configuration. It uses one-quarter of rank 16's trainable adapter parameters and slightly less peak GPU memory, while higher ranks showed no credible accuracy benefit. This is an efficiency recommendation, not an accuracy-superiority claim.

The shared backbone pretraining time was 53.82 seconds. All confirmatory adaptations used FP32, batch size 128, 48 hourly input steps, a six-hour forecast horizon, and 10 adaptation epochs on an NVIDIA Tesla T4 with 15,360 MiB VRAM. Notebook capture records driver 580.82.07 and host CUDA 13.0; PyTorch 2.9.1 used its CUDA 12.8 runtime.

The finding is limited to this small transformer, six-hour forecast horizon, synthetic target cooperative, fixed training budget, and tested software and hardware environment. It does not establish production performance or generalize to real utility customers.

Evidence: `artifacts/results/seed_level_results.csv`, `artifacts/results/summary_by_rank.csv`, `artifacts/plots/seed_level_mape.png`, `artifacts/environment/environment.json`, and `artifacts/checkpoints/pretraining_log.json`.
