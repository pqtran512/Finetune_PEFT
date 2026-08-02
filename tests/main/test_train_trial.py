from pathlib import Path
import sys
from unittest.mock import patch

import optuna
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "train"))

from train_trial import run_trial  # noqa: E402


def test_run_trial_prunes_on_oom():
    cfg = {
        "model_id": "dummy",
        "data_path": "finetune_lora7b_planB/data/java_completion_train.jsonl",
        "max_length": 1024,
        "proxy_train_fraction": 0.08,
        "proxy_epochs": 1,
        "eval_test_size": 0.02,
        "eval_seed": 42,
        "trial_output_root": "models/java-codellama-lora/bo_trials",
        "neftune_noise_alpha": 5,
    }
    hp = {
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.03,
        "lora_dropout": 0.05,
        "lora_r": 32,
        "lora_alpha": 64,
        "gradient_accumulation_steps": 16,
    }

    with patch(
        "train_trial.load_dataset",
        side_effect=RuntimeError("CUDA out of memory"),
    ):
        with pytest.raises(optuna.TrialPruned):
            run_trial(0, hp, cfg, cache_dir="unused")


def test_run_trial_prunes_on_cuda_unknown_error():
    cfg = {
        "model_id": "dummy",
        "data_path": "finetune_lora7b_planB/data/java_completion_train.jsonl",
        "max_length": 1024,
        "proxy_train_fraction": 0.08,
        "proxy_epochs": 1,
        "eval_test_size": 0.02,
        "eval_seed": 42,
        "trial_output_root": "models/java-codellama-lora/bo_trials",
        "neftune_noise_alpha": 5,
    }
    hp = {
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.03,
        "lora_dropout": 0.05,
        "lora_r": 32,
        "lora_alpha": 64,
        "gradient_accumulation_steps": 16,
    }

    with patch(
        "train_trial.load_dataset",
        side_effect=RuntimeError("CUDA error: unknown error"),
    ):
        with pytest.raises(optuna.TrialPruned):
            run_trial(0, hp, cfg, cache_dir="unused")
