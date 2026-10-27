# Project status — October 22, 2026

## Completed

- Controlled rank-4/rank-8/rank-16 design and pre-registration.
- Seeded synthetic source/target demand generator with chronological target split and SHA-256 provenance.
- Shared transformer pretraining and frozen-backbone LoRA adaptation on query/value projections.
- YAML-driven CLI, restartable rank/seed matrix, automatic environment capture, CSV aggregation, and seed-level plot.
- Thin Colab launcher, pinned dependencies, README, and built-in validation checks.
- Local CPU smoke run across all three ranks, including a deterministic repeatability check.

The local smoke run used one seed, one epoch, reduced data, and a reduced model. Its MAPE values are plumbing checks and are not evidence about which rank performs best. The most recent complete local smoke benchmark is in `artifacts/smoke/benchmark.json`.

## Completed after Colab execution

- Target-Colab GPU smoke and representative full-size timing benchmark.
- Five-seed decision recorded before confirmatory execution.
- Fifteen confirmatory runs on a Tesla T4.
- Seed-level CSV, summary CSV, plot, environment capture, and shared checkpoint.
- Results interpretation in `RESULTS.md`.

The Annex IV-style documentation and camera-on presentation plan are in `deliverables/`.
