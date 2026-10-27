# Seed-count decision record

- Colab date: October 22, 2026
- GPU model: NVIDIA Tesla T4
- Smoke benchmark artifact SHA-256: 7380a61fc8069dc16720663d4f97153090624bd4b84b4fe1e8681c5043f4c5e8
- Representative benchmark generation seed: 62708
- Representative non-study adaptation seed: 909
- One-epoch shared pretraining time: 8.59 seconds
- Projected eight-epoch shared pretraining time: 68.72 seconds
- One-epoch adaptation time: 1.61 seconds
- Projected ten-epoch adaptation time per run: 16.10 seconds
- Projected total for 9 runs: 3.56 minutes
- Projected total for 15 runs: 5.17 minutes
- 80% budget threshold: 144 minutes
- Selected seed count: 5

Five seeds were selected because the full-size, full-architecture timing
benchmark projected approximately 5.17 minutes for shared pretraining plus
15 adaptation runs. This is comfortably below the preregistered 144-minute
limit. The timing benchmark used a separate dataset-generation seed and a
non-study adaptation seed, so it is not included in confirmatory results.
