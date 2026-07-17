"""Shared helpers for CodeLlama QLoRA full train and Optuna trials."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

MAIN_DIR = Path(__file__).resolve().parent
REPO_ROOT = MAIN_DIR.parent
DEFAULT_BO_CONFIG = MAIN_DIR / "configs" / "bo_defaults.yaml"

DEFAULT_HYPERPARAMS: dict[str, Any] = {
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "lora_dropout": 0.05,
    "lora_r": 32,
    "lora_alpha": 64,
    "gradient_accumulation_steps": 16,
}


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def load_yaml_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or DEFAULT_BO_CONFIG
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_hyperparams(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    hp = dict(DEFAULT_HYPERPARAMS)
    if overrides:
        hp.update(overrides)
    hp["lora_alpha"] = int(hp["lora_r"]) * 2
    return hp


def load_hyperparams_from_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return merge_hyperparams(data)
