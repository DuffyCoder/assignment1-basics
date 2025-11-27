# Learning-rate Sweep Strategy

This document explains how to search for the optimal learning rate for TinyStories while
collecting the deliverables requested in the assignment.

## Search range

1. Start from the base AdamW setup (batch 32, context 256, cosine schedule with warmup).
2. Evaluate a logarithmic grid covering an order of magnitude around the baseline:
   - Coarse sweep: `[1e-4, 2e-4, 3e-4, 5e-4, 7.5e-4, 1e-3]`
   - Fine sweep (after narrowing down): e.g. `[2.5e-4, 2.8e-4, 3e-4, 3.2e-4, 3.5e-4]`
3. Run the sweep via `python -m tests.custom.lr_sweep --config ... --learning-rates ...`
   so that every run is logged under `experiments/runs/` and summarized in
   `experiments/sweeps/`.

## Tokens and schedule alignment

- If training on CPU/MPS, cap total tokens at ~41M by using `total_iters=5000`,
  `batch_size=32`, `context_length=256`.
- Ensure the cosine scheduler `max_iters` matches `total_iters` (handled automatically in
  `lr_sweep.py`) so each run sees a full warmup+decay cycle.
- Keep all other hyperparameters fixed between runs to isolate the effect of learning rate.

## Tracking deliverables

- For each run, plot `train/loss` and `val/loss` versus both `step` and `wall_time` by
  loading `experiments/runs/<run_id>/metrics.jsonl`.
- Record the final validation losses and note any divergence (NaNs, loss spikes) inside
  `experiments/experiment_log.md`.
- To capture the "edge of stability," extend the sweep with one or two higher learning
  rates (e.g., `1.2e-3` or `1.5e-3`) until you observe divergence. Include those curves in
  your report and discuss how the best LR compares to the divergence boundary.

## Detecting divergence

- Monitor `train/loss` for sudden jumps or NaNs/inf in `metrics.jsonl`.
- The sweep helper will continue even if one run fails; mark the run as divergent in the log.
- Divergent runs should still be kept so the plotted family of curves clearly shows the
  instability threshold.

## Recommended workflow

1. Run the coarse sweep and inspect curves to find the best LR candidate.
2. Run a finer sweep around that candidate.
3. Train one longer run at the selected LR to hit the required validation loss target
   (≤1.45 on GPU, ≤2.0 on CPU/MPS as permitted).
4. Document all runs (hyperparameters, losses, divergence behavior) in
   `experiments/experiment_log.md` and include the generated plots in your submission.
