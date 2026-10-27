# Pre-registration: LoRA rank and short-horizon demand forecasting

**Frozen before confirmatory runs:** October 22, 2026  
**Decision owner:** ML Research Engineer, Cobalt Grid (course scenario)

## Question and hypothesis

Does LoRA rank change target-utility forecasting error when adapting one fixed pretrained transformer backbone? The three arms are ranks 4, 8, and 16. The directional hypothesis is that additional capacity may lower error, but no rank is presumed to win.

## Controls

Every arm uses the same generated series, chronological split, shared pretrained checkpoint, q/v target modules, LoRA dropout, effective adapter scale (`alpha / rank = 1`), batch order rule, optimizer, learning rate, batch size, epoch count, evaluation code, and hardware session. The only intentional arm-level change is LoRA rank; the corresponding adapter parameter count is an inherent consequence.

The training-time random-number stream is reset after rank-specific adapter construction. This prevents the different number of initialized adapter values from shifting dropout masks or other training randomness between paired seed arms.

The generator seed is 62707 and the shared pretraining seed is 2026. Confirmatory seeds are 101, 202, and 303. Seeds 404 and 505 may be added only under the budget rule below, for every rank—not selectively.

## Data and metrics

The code generates three source-cooperative series for shared pretraining and one shifted target-cooperative series for adaptation. Each contains programmed daily/weekly/annual demand, weather response, trend, and autocorrelated noise. Target windows are split chronologically 70%/15%/15%. The input is 48 hours and the output is the next 6 hourly demand values.

Primary metric: test MAPE (%). Secondary metrics: MAE (kW), RMSE (kW), adaptation wall-clock seconds, peak allocated GPU memory, and trainable parameters. Test results are examined only after all selected seed/rank runs finish.

## Decision rule

A rank is not declared superior unless it has lower mean test MAPE and its seed-level min-max range does not overlap the comparator's range. With only three or five seeds, this conservative descriptive rule is preferred to underpowered significance testing. Overlap means the evidence is reported as inconclusive. All seed-level values remain visible.

## Compute rule fixed in advance

Run `python run_ablation.py --smoke --mode all` in the target Colab GPU runtime. Project total duration using the measured pretraining and per-run adaptation time. Use five seeds per rank only if shared pretraining plus 15 adaptations is projected at or below 144 minutes (80% of the 180-minute budget). Otherwise use three seeds per rank. The choice applies uniformly to all arms and is recorded before confirmatory execution.

## Exclusions and limitations

No completed confirmatory run is excluded unless the process terminates before writing a valid result artifact; failures are logged and rerun with the same configuration and seed. The study cannot establish production performance, real-utility representativeness, long-horizon behavior, causal effects beyond this configuration, or bitwise equivalence across hardware/software versions.
