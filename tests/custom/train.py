
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .checkpoint import Checkpoint
from .experiment_logger import ExperimentLogger, ExperimentLoggingConfig
from .get_batch import GetBatch
import wandb


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.01
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    warmup_iters: int = 1000
    max_iters: int = 100000
    min_lr: float = 1e-5


@dataclass
class CheckpointConfig:
    path: str
    every_n_steps: int = 1000
    keep_latest_only: bool = True


@dataclass
class TrainConfig:
    model_module: str
    model_class: str
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    loss_name: str = "CrossEntropyLoss"
    train_data_path: str = ""
    val_data_path: str | None = None
    tokenizer_np_dtype: str = "uint16"
    context_length: int = 256
    batch_size: int = 32
    micro_batch_size: int | None = None
    total_iters: int = 10000
    validate_every: int = 1000
    log_every: int = 10
    gradient_clip: float | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float32"
    seed: int = 42
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig | None = None
    checkpoint: CheckpointConfig | None = None
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    wandb_config: dict[str, Any] | None = None
    experiment_logging: ExperimentLoggingConfig | None = None


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    parser = argparse.ArgumentParser(description="训练语言模型")
    parser.add_argument("--config", type=str, help="JSON 配置文件路径", required=True)
    args = parser.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        raw_cfg = json.load(f)

    def _convert_optimizer(cfg: dict[str, Any]) -> OptimizerConfig:
        return OptimizerConfig(**cfg)

    def _convert_scheduler(cfg: dict[str, Any] | None) -> SchedulerConfig | None:
        if cfg is None:
            return None
        return SchedulerConfig(**cfg)

    def _convert_checkpoint(cfg: dict[str, Any] | None) -> CheckpointConfig | None:
        if cfg is None:
            return None
        return CheckpointConfig(**cfg)

    optimizer_cfg = _convert_optimizer(raw_cfg.pop("optimizer", {}))
    scheduler_cfg = _convert_scheduler(raw_cfg.pop("scheduler", None))
    checkpoint_cfg = _convert_checkpoint(raw_cfg.pop("checkpoint", None))

    def _convert_experiment_logging(cfg: dict[str, Any] | None) -> ExperimentLoggingConfig | None:
        if cfg is None:
            return None
        return ExperimentLoggingConfig(**cfg)

    experiment_logging_cfg = _convert_experiment_logging(raw_cfg.pop("experiment_logging", None))

    train_cfg = TrainConfig(
        optimizer=optimizer_cfg,
        scheduler=scheduler_cfg,
        checkpoint=checkpoint_cfg,
        experiment_logging=experiment_logging_cfg,
        **raw_cfg,
    )
    return train_cfg


def _load_numpy_memmap(path: str, dtype: str) -> np.memmap:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"找不到数据文件: {path}")
    return np.memmap(path_obj, dtype=getattr(np, dtype), mode="r")


class MemmapDataset(Dataset):
    def __init__(self, memmap_array: np.memmap, context_length: int):
        self.memmap_array = memmap_array
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.memmap_array) - self.context_length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index
        end = start + self.context_length
        x = torch.tensor(self.memmap_array[start:end], dtype=torch.long)
        y = torch.tensor(self.memmap_array[start + 1 : end + 1], dtype=torch.long)
        return x, y


def build_model(cfg: TrainConfig) -> nn.Module:
    module = __import__(cfg.model_module, fromlist=[cfg.model_class])
    model_cls = getattr(module, cfg.model_class)
    model: nn.Module = model_cls(**cfg.model_kwargs)
    return model


def build_optimizer(model: nn.Module, cfg: OptimizerConfig) -> Optimizer:
    name = cfg.name.lower()
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
            eps=cfg.eps,
        )
    raise ValueError(f"不支持的优化器: {cfg.name}")


def build_scheduler(optimizer: Optimizer, cfg: SchedulerConfig) -> torch.optim.lr_scheduler._LRScheduler:
    from .lr_cosine_schedule import LrCosineSchedule

    class _Scheduler(torch.optim.lr_scheduler._LRScheduler):
        def __init__(self, opt: Optimizer, schedule_cfg: SchedulerConfig):
            self.schedule_cfg = schedule_cfg
            super().__init__(opt)

        def get_lr(self) -> list[float]:  # type: ignore[override]
            step = self.last_epoch
            lr = LrCosineSchedule(
                it=step,
                max_learning_rate=self.base_lrs[0],
                min_learning_rate=self.schedule_cfg.min_lr,
                warmup_iters=self.schedule_cfg.warmup_iters,
                cosine_cycle_iters=self.schedule_cfg.max_iters,
            )()
            return [lr for _ in self.optimizer.param_groups]

    return _Scheduler(optimizer, cfg)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if dtype_str not in mapping:
        raise ValueError(f"不支持的数据类型: {dtype_str}")
    return mapping[dtype_str]


def _maybe_cast(model: nn.Module, dtype: torch.dtype, device: str) -> nn.Module:
    model = model.to(device=device, dtype=dtype)
    return model


def _log_to_console(step: int, losses: dict[str, float], metrics: dict[str, float]) -> None:
    log_items = {**losses, **metrics}
    parts = [f"step={step}"] + [f"{k}={v:.4f}" for k, v in log_items.items()]
    print(" | ".join(parts))


def _log_to_wandb(step: int, payload: dict[str, float]) -> None:
    wandb.log(payload, step=step)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: str,
    dtype: torch.dtype,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x.to(dtype=dtype))
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else math.nan


def _prepare_dataloader(memmap_array: np.memmap, cfg: TrainConfig, shuffle: bool) -> DataLoader:
    dataset = MemmapDataset(memmap_array, cfg.context_length)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )


def train_loop(cfg: TrainConfig) -> None:
    _set_seed(cfg.seed)
    dtype = _resolve_dtype(cfg.dtype)

    train_memmap = _load_numpy_memmap(cfg.train_data_path, cfg.tokenizer_np_dtype)
    val_memmap = _load_numpy_memmap(cfg.val_data_path, cfg.tokenizer_np_dtype) if cfg.val_data_path else None

    model = build_model(cfg)
    model = _maybe_cast(model, dtype, cfg.device)
    model.train()

    optimizer = build_optimizer(model, cfg.optimizer)
    scheduler = build_scheduler(optimizer, cfg.scheduler) if cfg.scheduler else None

    loss_fn = getattr(nn, cfg.loss_name)()

    checkpoint_helper = Checkpoint(model, optimizer)
    current_step = 0
    if cfg.checkpoint and Path(cfg.checkpoint.path).exists():
        try:
            current_step = checkpoint_helper.load(cfg.checkpoint.path)
            print(f"恢复训练: step={current_step}")
        except Exception as exc:  # pragma: no cover - 容错
            print(f"加载检查点失败: {exc}")

    if cfg.wandb_project:
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config=cfg.wandb_config or asdict(cfg),
        )

    train_loader = _prepare_dataloader(train_memmap, cfg, shuffle=True)
    val_loader = _prepare_dataloader(val_memmap, cfg, shuffle=False) if val_memmap is not None else None

    scaler = torch.amp.GradScaler(enabled=(dtype == torch.float16 and "cuda" in cfg.device))

    progress = tqdm(range(current_step, cfg.total_iters), initial=current_step, total=cfg.total_iters)
    start_time = time.time()
    run_start = time.perf_counter()

    experiment_logger = ExperimentLogger(cfg.experiment_logging)
    if experiment_logger.enabled:
        experiment_logger.log_config(asdict(cfg))
        experiment_logger.log_event(
            "training_started",
            current_step=current_step,
            total_iters=cfg.total_iters,
            device=cfg.device,
            dtype=cfg.dtype,
        )

    for step in progress:
        x, y = next(iter(train_loader))
        x = x.to(cfg.device)
        y = y.to(cfg.device)

        optimizer.zero_grad(set_to_none=True)

        use_amp = scaler.is_enabled()
        with torch.amp.autocast(enabled=use_amp, dtype=dtype):
            logits = model(x.to(dtype=dtype))
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

        if use_amp:
            scaler.scale(loss).backward()
            if cfg.gradient_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - start_time
        if (step + 1) % cfg.log_every == 0:
            metrics = {
                "tokens_per_sec": cfg.batch_size * cfg.context_length * cfg.log_every / max(elapsed, 1e-6),
                "lr": optimizer.param_groups[0]["lr"],
            }
            payload = {"train/loss": loss.item(), **metrics}
            _log_to_console(step + 1, {"train/loss": loss.item()}, metrics)
            _log_to_wandb(step + 1, payload)
            experiment_logger.log_metrics(step + 1, time.perf_counter() - run_start, payload, split="train")
            start_time = time.time()

        if val_loader is not None and (step + 1) % cfg.validate_every == 0:
            val_loss = evaluate(model, val_loader, loss_fn, cfg.device, dtype)
            payload = {"val/loss": val_loss}
            _log_to_console(step + 1, {"val/loss": val_loss}, {})
            _log_to_wandb(step + 1, payload)
            experiment_logger.log_metrics(step + 1, time.perf_counter() - run_start, payload, split="val")

        if cfg.checkpoint and (step + 1) % cfg.checkpoint.every_n_steps == 0:
            ckpt_path = Path(cfg.checkpoint.path)
            if ckpt_path.is_dir():
                ckpt_file = ckpt_path / f"checkpoint_step_{step+1}.pt"
            else:
                ckpt_file = ckpt_path
            ckpt_file.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_helper.save(step + 1, ckpt_file)
            if cfg.checkpoint.keep_latest_only and ckpt_file.is_file():
                for sibling in ckpt_file.parent.glob("checkpoint_step_*.pt"):
                    if sibling != ckpt_file and sibling.stat().st_mtime < ckpt_file.stat().st_mtime:
                        sibling.unlink(missing_ok=True)

    if cfg.checkpoint:
        final_path = Path(cfg.checkpoint.path)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_helper.save(cfg.total_iters, final_path)

    if wandb is not None and wandb.run is not None:
        wandb.finish()

    if experiment_logger.enabled:
        experiment_logger.log_event("training_finished", final_step=cfg.total_iters)
        experiment_logger.finalize(
            status="completed",
            summary={
                "total_steps": cfg.total_iters,
                "train_data_path": cfg.train_data_path,
                "val_data_path": cfg.val_data_path,
            },
        )


def main(argv: list[str] | None = None) -> None:
    cfg = parse_args(argv)
    train_loop(cfg)


if __name__ == "__main__":
    main()
        
