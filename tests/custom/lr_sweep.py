from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Iterable

from .train import TrainConfig, parse_args, train_loop


def _load_base_config(config_path: str) -> TrainConfig:
    # Reuse train.parse_args to leverage type conversion helpers.
    return parse_args(["--config", config_path])


def _serialize_config(cfg: TrainConfig) -> dict:
    def _convert(obj):
        if dataclasses.is_dataclass(obj):
            return {k: _convert(v) for k, v in dataclasses.asdict(obj).items()}
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        return obj

    return _convert(cfg)


def _prepare_run_config(base_cfg: TrainConfig, lr: float, run_suffix: str) -> TrainConfig:
    cfg = dataclasses.replace(
        base_cfg,
        optimizer=dataclasses.replace(base_cfg.optimizer, lr=lr),
    )
    if cfg.scheduler:
        cfg.scheduler = dataclasses.replace(cfg.scheduler, max_iters=cfg.total_iters)

    suffix = run_suffix or f"lr_{lr:g}"
    if cfg.wandb_run_name:
        cfg.wandb_run_name = f"{cfg.wandb_run_name}_{suffix}"
    if cfg.experiment_logging:
        cfg.experiment_logging = dataclasses.replace(
            cfg.experiment_logging,
            run_name=f"{cfg.experiment_logging.run_name or 'sweep'}_{suffix}",
            tags=list(set((cfg.experiment_logging.tags or []) + ["lr_sweep"])),
        )
    return cfg


def run_lr_sweep(config_path: str, learning_rates: Iterable[float], output_dir: str | None = None) -> None:
    base_cfg = _load_base_config(config_path)
    lr_list = list(learning_rates)
    for idx, lr in enumerate(lr_list, start=1):
        suffix = f"{idx:02d}_lr{lr:g}"
        run_cfg = _prepare_run_config(base_cfg, lr, run_suffix=suffix)
        print(f"[Sweep] Starting run {idx}/{len(lr_list)} with lr={lr}")
        train_loop(run_cfg)
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            cfg_dump_path = Path(output_dir) / f"run_{suffix}_config.json"
            cfg_dump_path.write_text(json.dumps(_serialize_config(run_cfg), indent=2), encoding="utf-8")


def _parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a learning-rate sweep across multiple training jobs.")
    parser.add_argument("--config", required=True, help="Path to the base JSON config.")
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        required=True,
        help="List of learning rates to evaluate (space separated).",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/sweeps",
        help="Where to store serialized configs for reproducibility.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_cli(argv)
    run_lr_sweep(args.config, args.learning_rates, args.output_dir)


if __name__ == "__main__":
    main()
