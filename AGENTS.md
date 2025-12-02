# AGNETS: Assignment 1 Basics Workspace Guide

## Project Snapshot
- **Goal**: Build TinyStories-scale GPT-style language models from scratch for CS336 Assignment 1, including tokenizer/BPE tooling, transformer components, and a reproducible training & experiment logging workflow.
- **Entrypoints**: `tests/adapters.py` connects reference tests to student implementations under `tests/custom/`. `tests/custom/train.py` hosts the production training loop plus CLI config ingestion.
- **Execution environment**: Managed with `uv` (`pyproject.toml`). Use `uv run <module.py>` for scripts and `uv run pytest` for test validation. `make_submission.sh` packages the repo after running pytest.

## Repository Layout Highlights
- `README.md`, `cs336_spring2025_assignment1_basics.pdf`: Assignment overview, setup (data download, pytest invocation).
- `pyproject.toml`, `uv.lock`: Dependency lock (torch, numpy/jaxtyping/einops, wandb, datasets, etc.).
- `cs336_basics/pretokenization_example.py`: Reference helper for chunking raw text before tokenizer training.
- `tests/`: Autograded suites plus fixtures (`tests/fixtures/*.txt`, tokenizer references, `ts_tests/model.pt` for transformer snapshots). `_snapshots/*.npz` feed golden tensors for module-level tests.
- `tests/custom/`: Student implementations:
  - Tokenization stack (`tokenizer.py`, `bpe_counter.py`, `bpe/train_bpe_tinystories.py` via helper imports) and batching utilities (`get_batch.py`)
  - Transformer components (embedding, RMSNorm, RoPE, attention, SwiGLU feed-forward, TransformerBlock, TransformerLM) and math primitives (softmax, cross entropy).
  - Optimization/runtime helpers: custom AdamW, cosine LR schedule, gradient clipping adapter, experiment logger, checkpoint helper, learning-rate sweep runner, and primary `train.py` (config dataclasses, HF/disk dataset ingestion, AMP-aware train loop, wandb + local logging hooks, checkpoint save/load).
- `bpe/`: Notebook + script for training a TinyStories BPE. Serialized tokenizer artifacts already exist in `bpe/bpe_results/tinystories_vocab.json` and `tinystories_merges.pkl`.
- `data/`: Empty placeholder; populate with TinyStories/OpenWebText memmaps per README instructions, or stream directly from Hugging Face via `TrainConfig.hf_*` settings.
- `experiments/`: Process docs (`experiment_log.md`, `lr_sweep_strategy.md`). `experiments/runs` will be populated automatically by `ExperimentLogger`.

## Data & Tokenizer Flow
1. **BPE Training**: Run `python bpe/train_bpe_tinystories.py --input ...` to produce vocab/merges (already present under `bpe/bpe_results/`). Script instruments memory/time and stores pickled merges + JSON vocab.
2. **Tokenizer Runtime** (`tests/custom/tokenizer.py`):
   - GPT-2 style regex pre-tokenization, byte pair merges, special-token preservation, iterable encoding helpers, and decode support.
   - Hugging Face integration via `tests/custom/train._build_tokenizer_from_cfg`, requiring `tokenizer_vocab_path` & `tokenizer_merges_path`.
3. **Dataset Loading**: `TrainConfig` accepts either memmap paths (`train_data_path`, `val_data_path`, `tokenizer_np_dtype`) or Hugging Face dataset metadata (`hf_dataset_name`, splits, EOS token). `GetBatch` plus `MemmapDataset` convert token arrays into `(x, y)` windows aligned by `context_length`.

## Model & Training Stack
- **Modules**: `Embedding`, `Linear`, `RMSNorm`, `MultiheadSelfAttention` (with RoPE variant), `PositionwiseFeedForward` (SwiGLU), combined via `TransformerBlock` and `TransformerLM` for logits projection.
- **Optimization**: `OptimizerConfig` / `SchedulerConfig` feed into `torch.optim.AdamW` and the custom cosine scheduler wrapper. Optional Amp scaling, gradient clipping, and run logging to both wandb and structured JSONL via `ExperimentLogger`.
- **Checkpointing**: `tests/custom/checkpoint.py` saves model/optimizer state dicts + iteration counters, supports resume.
- **Experiment Automation**: `tests/custom/lr_sweep.py` clones configs per-LR, enforces scheduler alignment, and optionally dumps configs under `experiments/sweeps/`.

## Current Status & Known Gaps
- Implementation coverage: Every required module for tokenizer, transformer, optimizer, logging, and training already has a concrete implementation under `tests/custom/`. Snapshot fixtures exist but no recorded pytest output is in the repo, so test status is unknown.
- Tokenizer artifacts: TinyStories vocab/merge files exist in `bpe/bpe_results/`, but `data/` holds no memmap arrays yet.
- Experiment tracking: `experiments/experiment_log.md` only contains template/example rows; `experiments/runs/` hasn’t been created, so no runs have been logged.
- Configs: No JSON configs are checked in (train loop expects user-provided config path).

## Next Suggested Actions
1. **Sanity-check implementations**: Run `uv run pytest tests` to confirm adapters + custom modules satisfy all suites; resolve any failing cases before training.
2. **Prepare training data**: Either download TinyStories/OpenWebText per `README.md` and convert to numpy memmaps, or configure Hugging Face streaming by pointing `TrainConfig` to dataset name (`roneneldan/TinyStories`) and the tokenizer artifacts in `bpe/bpe_results/`.
3. **Author training configs**: Create `configs/*.json` describing your model + optimizer setup, referencing `TrainConfig` fields (`model_module`, `model_class`, checkpoint/wandb/experiment logging options).
4. **Log experiments**: Use `tests/custom/train.py --config ...` for single runs and `tests/custom/lr_sweep.py` for sweeps, ensuring `ExperimentLoggingConfig` base_dir stays inside `experiments/runs/`. Update `experiments/experiment_log.md` after each run (Run ID, loss trends, observations).
5. **Document progress**: Keep `experiments/lr_sweep_strategy.md` aligned with actual choices, note divergence boundaries, and archive plots from `metrics.jsonl` for the final report.
6. **Submission prep**: Once requirements are met, run `make_submission.sh` to execute pytest (non-blocking failures) and zip deliverables, but ensure large raw datasets/checkpoints remain excluded per script filters.

## Guideline
1. Always response in Chinese.
2. You can modify the AGENTS.md file when a big update is done.
3. Always plan first, then start to code.