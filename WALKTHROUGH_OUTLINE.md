# 5-8 minute unscripted walkthrough outline

Use these bullets as landmarks, not a script. Keep the repository open and show the files named below.

## 0:00-0:45 - Question and professional context

- "I tested whether LoRA rank 4, 8, or 16 changes six-hour demand-forecasting error when everything else is fixed."
- Target audience: an Applied ML Engineer interviewer supporting forecasting and grid analytics at Great River Energy.
- Decision: choose the smallest rank that does not show a credible accuracy penalty under a one-GPU budget.
- State the boundary immediately: synthetic data, small transformer, one Tesla T4; this is an experiment-quality demonstration, not a production forecast.

## 0:45-1:35 - Compute constraint and controlled design

- Open `configs/ablation.yaml`.
- One shared pretrained backbone; LoRA only on query/value projections.
- Rank is the only factor: 4, 8, 16; alpha equals rank.
- Fixed: 48-hour input, six-hour horizon, batch size 128, 10 adaptation epochs, FP32, optimizer, split, and evaluation.
- Five seeds per rank: 15 confirmatory runs. Explain why repeated measurements matter more than a best run.
- Hardware: Tesla T4, 15,360 MiB; driver 580.82.07; PyTorch CUDA 12.8.

## 1:35-2:35 - Reproducibility path

- Open the README quickstart.
- Show `requirements.txt`, `run_ablation.py`, and `tests/`.
- Show deterministic settings, the YAML seed list, environment capture, and dataset hash.
- Explain restartability: completed rank/seed JSON files are skipped, then aggregation rebuilds CSV summaries and the plot.
- Mention that the notebook is only a launcher, preventing notebook state from becoming a hidden second implementation.

## 2:35-3:50 - Headline result and spread

- Open `artifacts/plots/seed_level_mape.png`, then `artifacts/results/summary_by_rank.csv`.
- Rank 4: 3.2396% mean MAPE, 0.0381 SD, 3.1860-3.2931% range.
- Rank 8: 3.2487% mean, 0.0466 SD, 3.1854-3.3143% range.
- Rank 16: 3.2669% mean, 0.0578 SD, 3.1825-3.3333% range.
- The ranges overlap. Say plainly: "I did not demonstrate an accuracy winner."
- Rank 4 uses 2,048 trainable parameters versus 8,192 at rank 16, with 123.02 MB versus 124.21 MB peak allocated GPU memory.
- Recommendation: rank 4 is the efficiency choice for this exact configuration, not a universal optimum.

## 3:50-4:45 - Measurement lesson and failure diagnosis

- The lowest mean alone is a measurement to distrust; it differs from rank 8 by only 0.0092 percentage points.
- Show the seed-level CSV to make variability visible.
- Explain the audit issue: `artifacts/benchmark.json` was overwritten on a restart and reports a skipped-run wall time. It is excluded from evidence.
- Authoritative time records are the 15 per-run files and pretraining log; adaptation plus evaluation was about 14 seconds per run, and pretraining was 53.82 seconds.

## 4:45-5:50 - Limitations and governance

- Synthetic data does not represent real cooperative customers or extremes.
- Only adaptation seeds varied; the shared pretraining seed and generated dataset did not.
- No rank-specific hyperparameter sweep; fixed settings improve control but may disadvantage a rank.
- One model size, horizon, split, GPU class, and descriptive uncertainty rule.
- Connect intended use, provenance, environment capture, the human decision gate, and residual risk to NIST AI RMF, ISO/IEC 42001, and EU Annex IV evidence practices.
- The package demonstrates documentation discipline; it does not claim certification or production readiness.

## 5:50-6:45 - What more compute and real access would buy

- Repeat pretraining across seeds and regenerate multiple datasets.
- Tune learning rate/regularization by rank with nested validation.
- Validate on approved real cooperative data, seasonal and extreme-weather slices, and held-out service territories.
- Compare stronger baselines and longer horizons.
- Measure end-to-end latency, energy, and operational drift on target infrastructure.
- Use confidence intervals or a preregistered equivalence/non-inferiority design with enough repetitions.

## 6:45-7:15 - Close

- "The skill I am presenting is not choosing rank 4. It is building an auditable path from a constrained experiment to a claim that stops exactly where the evidence stops."
- Point the reviewer to the raw CSV, audit, and README.
- Invite the question: "Which additional evidence would be required before this could influence a real planning workflow?"

## Recording checklist

- Keep camera on and screen share readable.
- Aim for 6:30-7:15; do not read sentences verbatim.
- Show the config, quickstart, raw/summary results, plot, and limitations.
- Upload with view access, test the link in a signed-out/private window, and replace `<VIDEO_LINK_AFTER_RECORDING>` in the README and one-page PDF.
