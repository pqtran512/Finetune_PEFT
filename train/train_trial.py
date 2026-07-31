"""Single Optuna trial: proxy QLoRA train → eval_loss."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import optuna
import torch
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq, Trainer, TrainerCallback

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
    _tokenize_fn,
)


class OptunaPruningCallback(TrainerCallback):
    def __init__(self, trial: optuna.Trial):
        self.trial = trial

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if state.is_local_process_zero and metrics:
            val_loss = metrics.get("eval_loss")
            if val_loss is not None:
                self.trial.report(val_loss, step=state.global_step)
                if self.trial.should_prune():
                    message = f"Trial was pruned at step {state.global_step} with loss {val_loss}"
                    raise optuna.TrialPruned(message)


def run_trial(
    trial: optuna.Trial,
    hyperparams: dict[str, Any],
    cfg: dict[str, Any],
    *,
    cache_dir: str | None = None,
) -> float:
    trial_number = trial.number
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

        # Load train dataset (evol_java_completion_train.jsonl)
        raw_train = load_dataset("json", data_files=str(data_path), split="train")

        # Load validation dataset (java_completion_train.jsonl)
        val_path = data_path.parent / "java_completion_train.jsonl"
        if not val_path.exists():
            val_path = REPO_ROOT / "data" / "jsonl" / "java_completion_train.jsonl"
        if not val_path.exists():
            raise FileNotFoundError(f"Missing validation dataset: {val_path}")

        raw_val = load_dataset("json", data_files=str(val_path), split="train")

        # Select a fixed 1000 samples for validation
        raw_val = raw_val.shuffle(seed=42).select(range(min(1000, len(raw_val))))

        tokenizer = build_tokenizer(cfg["model_id"], cache)

        # Process train proxy slice
        proxy_fraction = float(cfg["proxy_train_fraction"])
        n_train = max(1, int(len(raw_train) * proxy_fraction))
        raw_train_proxy = raw_train.shuffle(seed=42).select(range(n_train))

        # Tokenize datasets
        tokenize_with_mask = _tokenize_fn(tokenizer, int(cfg["max_length"]))

        train_ds = raw_train_proxy.map(
            tokenize_with_mask,
            remove_columns=raw_train_proxy.column_names,
            num_proc=2,
        )
        eval_ds = raw_val.map(
            tokenize_with_mask,
            remove_columns=raw_val.column_names,
            num_proc=2,
        )

        # Filter empty targets
        def has_target(ex):
            return any(l != -100 for l in ex["labels"])

        train_ds = train_ds.filter(has_target)
        eval_ds = eval_ds.filter(has_target)

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
            callbacks=[OptunaPruningCallback(trial)],
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
