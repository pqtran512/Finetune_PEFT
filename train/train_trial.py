"""Single Optuna trial: proxy QLoRA train → eval_loss."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import optuna
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq, Trainer, TrainerCallback

from training_common import (
    REPO_ROOT,
    build_qlora_model,
    build_tokenizer,
    build_training_args,
    cleanup_cuda,
    get_hf_cache_dir,
    is_cuda_error,
    merge_hyperparams,
    remove_path_quiet,
    resolve_data_path,
    resolve_repo_path,
    training_runtime_from_cfg,
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


def resolve_val_path(data_path: Path) -> Path:
    """Legacy CodeLlama runs validate on java_completion while training on Evol."""
    sibling = data_path.parent / "java_completion_train.jsonl"
    if sibling.exists():
        return sibling
    fallback = REPO_ROOT / "data" / "jsonl" / "java_completion_train.jsonl"
    return fallback


def holdout_validation(raw_train, *, val_size: int, seed: int):
    """Fixed validation slice, removed from the pool every trial can train on."""
    shuffled = raw_train.shuffle(seed=seed)
    n_val = min(int(val_size), max(1, len(shuffled) - 1))
    raw_val = shuffled.select(range(n_val))
    pool = shuffled.select(range(n_val, len(shuffled)))
    return pool, raw_val


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

        raw_train = load_dataset("json", data_files=str(data_path), split="train")

        val_size = int(cfg.get("val_size", 1000))
        split_seed = int(cfg.get("eval_seed", 42))
        val_path = resolve_val_path(data_path)
        same_file = val_path.resolve() == data_path.resolve()
        if same_file:
            # Train and val come from one jsonl: hold val out before the proxy slice.
            raw_train, raw_val = holdout_validation(
                raw_train, val_size=val_size, seed=split_seed
            )
            print(
                f"Held out {len(raw_val)} val samples from {data_path.name}; "
                f"train pool={len(raw_train)}"
            )
        else:
            if not val_path.exists():
                raise FileNotFoundError(f"Missing validation dataset: {val_path}")
            raw_val = load_dataset("json", data_files=str(val_path), split="train")
            raw_val = raw_val.shuffle(seed=split_seed).select(
                range(min(val_size, len(raw_val)))
            )

        runtime = training_runtime_from_cfg(cfg)
        tokenizer = build_tokenizer(
            cfg["model_id"],
            cache,
            trust_remote_code=runtime["trust_remote_code"],
        )

        # Proxy slice is taken only from the train pool, never from val.
        proxy_fraction = float(cfg["proxy_train_fraction"])
        n_train = max(1, int(len(raw_train) * proxy_fraction))
        raw_train_proxy = raw_train.shuffle(seed=split_seed).select(range(n_train))

        # Tokenize datasets
        tokenize_with_mask = _tokenize_fn(tokenizer, int(cfg["max_length"]))
        map_num_proc = int(cfg.get("map_num_proc", 2))

        train_ds = raw_train_proxy.map(
            tokenize_with_mask,
            remove_columns=raw_train_proxy.column_names,
            num_proc=map_num_proc,
        )
        eval_ds = raw_val.map(
            tokenize_with_mask,
            remove_columns=raw_val.column_names,
            num_proc=map_num_proc,
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
            trust_remote_code=runtime["trust_remote_code"],
            attn_implementation=runtime["attn_implementation"],
        )

        args = build_training_args(
            output_dir=str(trial_dir),
            hyperparams=hp,
            num_train_epochs=float(cfg["proxy_epochs"]),
            train_len=len(train_ds),
            proxy=True,
            report_to="none",
            run_name=f"bo-trial-{trial_number}",
            per_device_train_batch_size=runtime["per_device_train_batch_size"],
            per_device_eval_batch_size=runtime["per_device_eval_batch_size"],
            dataloader_num_workers=runtime["dataloader_num_workers"],
            neftune_noise_alpha=runtime["neftune_noise_alpha"],
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
        if is_cuda_error(exc):
            cleanup_cuda(trainer, model)
            raise optuna.TrialPruned(f"CUDA trial={trial_number}: {exc}") from exc
        raise
    finally:
        cleanup_cuda(trainer, model)
        remove_path_quiet(trial_dir)
