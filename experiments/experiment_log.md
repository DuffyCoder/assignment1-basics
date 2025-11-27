# Experiment Log

This document serves as the single place where assignment experiments are tracked. Each
entry corresponds to exactly one training/evaluation run. Update the table below every
time you finish a run so that you have a chronological record of what was tried and what
worked.

## How to record a run

1. Enable the built-in logger by adding the following snippet to your training config:

   ```json
   {
     "experiment_logging": {
       "base_dir": "experiments/runs",
       "run_name": "baseline",
       "tags": ["assignment1", "tinystories"]
     }
   }
   ```

2. Start training; a new folder will appear under `experiments/runs/<run_id>` containing
   `config.json`, `metrics.jsonl`, and `summary.json`.
3. Plot loss curves by loading `metrics.jsonl` (each record includes `step` and `wall_time`
   so curves can be drawn against gradient steps or seconds).
4. Add a row to the log below summarizing the run, highlighting the important settings and
   observations.

## Learning-rate sweep workflow

Use the helper at `tests/custom/lr_sweep.py` to sweep over multiple learning rates with
consistent logging/config dumps:

```
python -m tests.custom.lr_sweep \
  --config configs/tinystories_base.json \
  --learning-rates 1e-4 2e-4 3e-4 5e-4
```

- Each run inherits the base config and only the learning rate changes.
- Scheduler `max_iters` is automatically aligned with `total_iters`.
- Experiment logger run names/tags are suffixed with the learning rate so plots in
  `metrics.jsonl` are easy to correlate.
- Results are summarized in `experiments/sweeps/` (config snapshots) and
  `experiments/runs/` (metrics/events). Use these to build the learning curves required
  for the report and to document divergence thresholds.

## Runs

| Run ID | Date | Task / Dataset | Config Highlights | Train Loss Trend | Val Loss Trend | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| _add run id_ | _yyyy-mm-dd_ |  |  |  |  |  |

## Example entries (remove once real data is added)

| Run ID | Date | Task / Dataset | Config Highlights | Train Loss Trend | Val Loss Trend | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 20240505_baseline | 2024-05-05 | TinyStories (256 ctx) | d_model=256, AdamW lr=3e-4, warmup 1k | 4.51 → 2.36 over 5k steps | 4.62 → 2.71 | First overfit baseline; observed instabilities before warmup completed. |
| 20240506_cosine_decay | 2024-05-06 | TinyStories (256 ctx) | Added cosine LR schedule + grad clipping 1.0 | 4.5 → 2.1 over 5k steps | 4.6 → 2.4 | Much smoother than baseline; next step is to increase batch size. |
