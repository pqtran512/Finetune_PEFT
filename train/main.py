"""Fine-tune CodeLlama-7B QLoRA trên Java method-body completion (prefix/target).

Default data: data/jsonl/evol_java_completion_train.jsonl
  (build: python data/build_evol_completion_dataset.py)
Target hardware: RTX 3060 12GB — QLoRA 4-bit + bf16 compute.
Optional: load best hyperparams from Optuna (`train/studies/best_params.json`).
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq, Trainer
from transformers.trainer_utils import get_last_checkpoint

from training_common import (
    MAIN_DIR,
    REPO_ROOT,
    build_qlora_model,
    build_tokenizer,
    build_training_args,
    get_hf_cache_dir,
    load_env,
    load_yaml_config,
    prepare_train_eval,
    resolve_data_path,
    resolve_repo_path,
    resolve_train_hyperparams,
    training_runtime_from_cfg,
)

load_env(MAIN_DIR / ".env")
load_env(REPO_ROOT / ".env")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
if WANDB_API_KEY:
    import wandb

    wandb.login(key=WANDB_API_KEY)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON hyperparams (default: studies/best_params.json if present)",
    )
    parser.add_argument(
        "--bo-config",
        type=Path,
        default=None,
        help="Optional path to bo_defaults.yaml for paths/model",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.bo_config)
    best_path = resolve_repo_path(cfg["best_params_path"])
    if args.config is None and not best_path.exists():
        print(
            f"Best-params file not found ({best_path}). "
            "Using built-in defaults. Run bayes_opt.py or pass --config."
        )
    hp = resolve_train_hyperparams(
        args.config,
        best_path,
        allow_repo_default=args.bo_config is None and args.config is None,
    )
    runtime = training_runtime_from_cfg(cfg)

    MODEL_ID = cfg["model_id"]
    OUTPUT_DIR = str(resolve_repo_path(cfg["output_dir"]))
    MAX_LENGTH = int(cfg["max_length"])
    NUM_EPOCHS = int(cfg.get("num_train_epochs", 3))
    CACHE_DIR = get_hf_cache_dir()
    print(f"HF_HOME / cache: {CACHE_DIR}")
    print(
        f"Model: {MODEL_ID} | max_length={MAX_LENGTH} | epochs={NUM_EPOCHS} | "
        f"micro_batch={runtime['per_device_train_batch_size']}"
    )

    if WANDB_API_KEY:
        import wandb

        wandb.init(
            project=str(cfg.get("wandb_project", "codellama-java")),
            name=str(cfg.get("wandb_run_name", "completion-qlora-3060-12gb")),
        )

    data_path = resolve_data_path(cfg)
    print("--- Loading dataset ---")
    if not data_path.exists():
        raise FileNotFoundError(
            f"Không thấy {data_path}. "
            "Chạy `python data/build_evol_completion_dataset.py` trước."
        )
    start = time.time()
    dataset = load_dataset("json", data_files=str(data_path), split="train")
    print(f"Load: {time.time() - start:.2f}s, n={len(dataset)}")

    print("--- Loading tokenizer ---")
    tokenizer = build_tokenizer(
        MODEL_ID,
        CACHE_DIR,
        trust_remote_code=runtime["trust_remote_code"],
    )

    print("--- Tokenizing ---")
    train_ds, eval_ds = prepare_train_eval(
        dataset,
        tokenizer,
        max_length=MAX_LENGTH,
        eval_test_size=float(cfg["eval_test_size"]),
        eval_seed=int(cfg["eval_seed"]),
        proxy_fraction=None,
        num_proc=int(cfg.get("map_num_proc", 4)),
    )
    print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")
    print(f"Hyperparams: {hp}")

    print("--- Loading model (QLoRA 4-bit nf4, bf16 compute) ---")
    model = build_qlora_model(
        MODEL_ID,
        CACHE_DIR,
        lora_r=hp["lora_r"],
        lora_alpha=hp["lora_alpha"],
        lora_dropout=hp["lora_dropout"],
        trust_remote_code=runtime["trust_remote_code"],
        attn_implementation=runtime["attn_implementation"],
    )
    model.print_trainable_parameters()

    report_to = "wandb" if WANDB_API_KEY else "none"
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=build_training_args(
            output_dir=OUTPUT_DIR,
            hyperparams=hp,
            num_train_epochs=NUM_EPOCHS,
            train_len=len(train_ds),
            proxy=False,
            report_to=report_to,
            run_name=str(cfg.get("run_name", "codellama-completion-qlora-3060")),
            per_device_train_batch_size=runtime["per_device_train_batch_size"],
            per_device_eval_batch_size=runtime["per_device_eval_batch_size"],
            dataloader_num_workers=runtime["dataloader_num_workers"],
            neftune_noise_alpha=runtime["neftune_noise_alpha"],
        ),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True, label_pad_token_id=-100
        ),
    )

    print("--- Starting training ---")
    checkpoint = get_last_checkpoint(OUTPUT_DIR) if os.path.isdir(OUTPUT_DIR) else None
    print(f"Checkpoint: {checkpoint}")

    start = time.time()
    trainer.train(resume_from_checkpoint=checkpoint)
    print(f"Training: {(time.time() - start) / 60:.2f} min")

    trainer.save_model(OUTPUT_DIR)
    print(f"Done. Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
