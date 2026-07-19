"""Optuna TPE Bayesian Optimization for CodeLlama QLoRA hyperparams."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import optuna
import torch

from train_trial import run_trial
from training_common import (
    MAIN_DIR,
    REPO_ROOT,
    load_env,
    load_yaml_config,
    merge_hyperparams,
    resolve_repo_path,
)

load_env(MAIN_DIR / ".env")
load_env(REPO_ROOT / ".env")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def suggest_hyperparams(
    trial: optuna.Trial, search_space: dict[str, Any]
) -> dict[str, Any]:
    hp: dict[str, Any] = {}
    for name, spec in search_space.items():
        t = spec["type"]
        if t == "loguniform":
            hp[name] = trial.suggest_float(
                name, float(spec["low"]), float(spec["high"]), log=True
            )
        elif t == "uniform":
            hp[name] = trial.suggest_float(
                name, float(spec["low"]), float(spec["high"])
            )
        elif t == "categorical":
            hp[name] = trial.suggest_categorical(name, list(spec["choices"]))
        else:
            raise ValueError(f"Unknown search type: {t}")
    return merge_hyperparams(hp)


def save_best_params(path: Path, params: dict[str, Any], value: float) -> None:
    hp = merge_hyperparams(params)
    payload = {**hp, "best_eval_loss": float(value)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_best_from_study(study: optuna.Study, best_path: Path) -> bool:
    completed = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed:
        print("No completed trials; best_params not written.")
        return False
    save_best_params(best_path, study.best_params, study.best_value)
    print(
        f"Best trial={study.best_trial.number} "
        f"eval_loss={study.best_value:.6f}"
    )
    print(f"Wrote {best_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Bayesian opt for CodeLlama QLoRA")
    parser.add_argument(
        "--config", type=Path, default=None, help="Path to bo_defaults.yaml"
    )
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument(
        "--export-best",
        action="store_true",
        help="Only load study DB and write best_params.json (no new trials)",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    n_trials = args.n_trials if args.n_trials is not None else int(cfg["n_trials"])

    db_path = resolve_repo_path(cfg["study_db"])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{db_path.as_posix()}"
    best_path = resolve_repo_path(cfg["best_params_path"])

    study = optuna.create_study(
        study_name=cfg["study_name"],
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    if args.export_best:
        export_best_from_study(study, best_path)
        return

    cache_dir = os.environ.get(
        "HF_CACHE_DIR", str(Path.home() / ".cache" / "huggingface")
    )

    def objective(trial: optuna.Trial) -> float:
        hp = suggest_hyperparams(trial, cfg["search_space"])
        trial_hp = {
            k: hp[k]
            for k in (
                "learning_rate",
                "weight_decay",
                "warmup_ratio",
                "lora_dropout",
                "lora_r",
                "gradient_accumulation_steps",
            )
        }
        return run_trial(trial.number, trial_hp, cfg, cache_dir=cache_dir)

    def _save_best_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            export_best_from_study(study, best_path)

    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[_save_best_callback],
    )
    export_best_from_study(study, best_path)


if __name__ == "__main__":
    main()
