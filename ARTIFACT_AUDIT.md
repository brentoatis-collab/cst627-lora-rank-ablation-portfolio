# Option A reproducibility audit

## What the original Week 7 submission did not make easy for a stranger

| Gap | Why it mattered | Portfolio repair |
|---|---|---|
| Week 7 was submitted as a ZIP because no repository was required | A Week 8 reviewer still needs a cloneable, versioned artifact | Refined package published as a new Week 8 portfolio repository and linked in the README and one-page brief |
| Hardware details were split across notebook output and environment JSON | VRAM, driver, host CUDA, and PyTorch CUDA were not visible together | Consolidated as Tesla T4, 15,360 MiB, driver 580.82.07, host CUDA 13.0, PyTorch CUDA 12.8 |
| Main README emphasized course execution rather than an employer decision | Strong mechanics, but professional relevance was implicit | Rewritten for an Applied ML Engineer target at Great River Energy with intended use, out-of-scope use, governance, and skills-to-evidence mapping |
| Data licensing and operational exclusion were not explicit | A reviewer could mistake synthetic evidence for deployable utility evidence | MIT license, provenance, hash, and production exclusions added |
| Result claim needed tighter compute bounds | Lowest mean could be read as superiority | Claim now states that all seed ranges overlap and recommends rank 4 only as the lowest-cost tested choice |
| Benchmark file was inconsistent after restart | `artifacts/benchmark.json` reports 9 scheduled runs and subsecond time because completed results were skipped | Retained for audit history but excluded from evidence; per-run records and pretraining log are authoritative |
| Video link did not yet exist | Submission cannot be fully link-verified | Explicit placeholder added; must be replaced after recording |

## Evidence integrity review

- 15 confirmatory JSON files are present and map to 15 unique rank/seed pairs.
- Ranks are 4, 8, and 16; seeds are 101, 202, 303, 404, and 505 for every rank.
- `seed_level_results.csv` matches the per-run JSON records.
- `summary_by_rank.csv` recomputes from the raw CSV for count, mean, sample standard deviation, minimum, maximum, time, memory, and trainable parameters.
- The seed-level plot is generated from the raw CSV and shows all observations.
- Dataset SHA-256 matches metadata: `201633d0f0c629a39a66ec4d16089c85497ebab0715fa42cca66237a974ba2ef`.
- Captured device is NVIDIA Tesla T4. Notebook `nvidia-smi` output reports 15,360 MiB VRAM, driver 580.82.07, and host CUDA 13.0; environment capture reports PyTorch 2.9.1+cu128 and CUDA runtime 12.8.
- Automated checks pass for deterministic data generation and LoRA-only trainable parameters.

## Reproducibility boundary

The evidence supports procedural and statistical reproduction on a compatible single GPU. It does not promise bitwise equivalence across drivers, CUDA versions, GPU architectures, or future dependency builds. The full confirmatory matrix was not rerun for this Week 8 packaging step; the 15 preserved, internally cross-checked Week 7 measurements are the evidence. No result was fabricated or imputed.

The authoritative Week 7 source was the submitted `cst627-week7-lora-ablation-complete.zip`. The public GitHub repository was created for the Week 8 portfolio requirement after the audit and professional refinement.
