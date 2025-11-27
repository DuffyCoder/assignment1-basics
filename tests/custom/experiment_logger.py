from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass
class ExperimentLoggingConfig:
    """
    Configuration describing how experiment artifacts should be captured.

    Attributes:
        base_dir: Root directory in which run folders will be created.
        run_name: Optional user-friendly name. If omitted a timestamp-based name is used.
        enabled: Allows turning logging on/off without editing the training loop.
        tags: Optional free-form tags that are saved inside the run metadata.
        notes: Arbitrary notes that get persisted with the project metadata.
    """

    base_dir: str = "experiments/runs"
    run_name: str | None = None
    enabled: bool = True
    tags: list[str] = field(default_factory=list)
    notes: str | None = None


class ExperimentLogger:
    """
    Lightweight run logger that stores metrics to disk so that experiments can
    be tracked without relying on external services such as Weights & Biases.
    """

    def __init__(self, cfg: ExperimentLoggingConfig | None) -> None:
        self.cfg = cfg
        self.enabled = bool(cfg and cfg.enabled)
        self.run_id: str | None = None
        self.run_dir: Path | None = None
        self._metrics_path: Path | None = None
        self._events_path: Path | None = None
        self._summary_path: Path | None = None

        if not self.enabled:
            return

        run_name = cfg.run_name or "run"
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.run_id = f"{timestamp}_{self._slugify(run_name)}"

        self.run_dir = Path(cfg.base_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._metrics_path = self.run_dir / "metrics.jsonl"
        self._events_path = self.run_dir / "events.jsonl"
        self._summary_path = self.run_dir / "summary.json"

        meta = {
            "run_id": self.run_id,
            "created_at": timestamp,
            "tags": cfg.tags,
            "notes": cfg.notes,
        }
        (self.run_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @staticmethod
    def _slugify(value: str) -> str:
        cleaned = "".join(char if char.isalnum() or char in "-_" else "-" for char in value.strip().lower())
        cleaned = cleaned.strip("-_") or "run"
        return cleaned[:64]

    def log_config(self, config_payload: Mapping[str, Any]) -> None:
        if not self.enabled or self.run_dir is None:
            return
        config_path = self.run_dir / "config.json"
        config_path.write_text(json.dumps(config_payload, indent=2, default=str), encoding="utf-8")

    def log_metrics(self, step: int, wall_time: float, metrics: Mapping[str, float], split: str = "train") -> None:
        if not self.enabled or self._metrics_path is None:
            return
        entry = {
            "step": step,
            "wall_time": wall_time,
            "split": split,
            "metrics": dict(metrics),
            "timestamp": time.time(),
        }
        with self._metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    def log_event(self, message: str, **extra: Any) -> None:
        if not self.enabled or self._events_path is None:
            return
        event = {"message": message, "timestamp": time.time(), **extra}
        with self._events_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event) + "\n")

    def finalize(self, status: str, summary: Mapping[str, Any] | None = None) -> None:
        if not self.enabled or self._summary_path is None:
            return
        payload = {"status": status, "timestamp": time.time()}
        if summary:
            payload["summary"] = dict(summary)
        self._summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
