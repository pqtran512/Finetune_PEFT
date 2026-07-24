"""Single Optuna trial: proxy QLoRA train → eval_loss."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import optuna
import torch
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq, Trainer

from training_common import (
    build_qlora_model,
    build_tokenizer,
    build_training_args,
    cleanup_cuda,
    get_hf_cache_dir,
    is_oom_error,
    merge_hyperparams,
    prepare_train_eval,
    remove_path_quiet,
    resolve_data_path,
    resolve_repo_path,
)


def run_trial(
    trial_number: int,
    hyperparams: dict[str, Any],
    cfg: dict[str, Any],
    *,
    cache_dir: str | None = None,
) -> float:
    hp = merge_hyperparams(hyperparams)
    cache = cache_dir or get_hf_cache_dir()
    trial_dir = resolve_repo_path(cfg["trial_output_root"]) / f"bo_trial_{trial_number}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    model = None
    trainer = None
    try:
        data_path = resolve_data_path(cfg)
        if not data_path.exists():
            raise FileNotFoundError(f"Missing dataset: {data_path}")

        raw = load_dataset("json", data_files=str(data_path), split="train")
        tokenizer = build_tokenizer(cfg["model_id"], cache)
        train_ds, eval_ds = prepare_train_eval(
            raw,
            tokenizer,
            max_length=int(cfg["max_length"]),
            eval_test_size=float(cfg["eval_test_size"]),
            eval_seed=int(cfg["eval_seed"]),
            proxy_fraction=float(cfg["proxy_train_fraction"]),
            proxy_seed=int(trial_number),
            num_proc=2,
        )

        model = build_qlora_model(
            cfg["model_id"],
            cache,
            lora_r=hp["lora_r"],
            lora_alpha=hp["lora_alpha"],
            lora_dropout=hp["lora_dropout"],
        )

        args = build_training_args(
            output_dir=str(trial_dir),
            hyperparams=hp,
            num_train_epochs=float(cfg["proxy_epochs"]),
            train_len=len(train_ds),
            proxy=True,
            report_to="none",
            run_name=f"bo-trial-{trial_number}",
        )
        trainer = Trainer(
            model=model,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            args=args,
            data_collator=DataCollatorForSeq2Seq(
                tokenizer, padding=True, label_pad_token_id=-100
            ),
        )
        trainer.train()
        metrics = trainer.evaluate()
        return float(metrics["eval_loss"])
    except Exception as exc:
        if is_oom_error(exc):
            cleanup_cuda(trainer, model)
            raise optuna.TrialPruned(f"OOM trial={trial_number}: {exc}") from exc
        raise
    finally:
        cleanup_cuda(trainer, model)
        remove_path_quiet(trial_dir)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
