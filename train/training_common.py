"""Shared helpers for CodeLlama QLoRA full train and Optuna trials."""
from __future__ import annotations

import gc
import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

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

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


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


def get_hf_cache_dir() -> str:
    """Chỉ dùng HF_HOME (chuẩn Hugging Face). Ví dụ trong .env: HF_HOME=D:/cache/huggingface"""
    return os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))


def resolve_train_hyperparams(
    config_arg: Path | None = None,
    default_path: Path | None = None,
    *,
    allow_repo_default: bool = True,
) -> dict[str, Any]:
    """Priority: --config path > default_path > studies/best_params.json > defaults.

    allow_repo_default=False skips the CodeLlama best_params.json fallback so a
    model-specific BO config cannot silently train with another model's params.
    """
    if config_arg is not None:
        return load_hyperparams_from_json(config_arg)
    if default_path is not None and default_path.exists():
        return load_hyperparams_from_json(default_path)
    if allow_repo_default:
        default_best = MAIN_DIR / "studies" / "best_params.json"
        if default_best.exists():
            return load_hyperparams_from_json(default_best)
    return merge_hyperparams()


def training_runtime_from_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Hardware knobs. Missing keys keep the CodeLlama / RTX 3060 recipe."""
    return {
        "per_device_train_batch_size": int(cfg.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(cfg.get("per_device_eval_batch_size", 1)),
        "dataloader_num_workers": int(cfg.get("dataloader_num_workers", 0)),
        "neftune_noise_alpha": float(cfg.get("neftune_noise_alpha", 5)),
        "trust_remote_code": bool(cfg.get("trust_remote_code", False)),
        "attn_implementation": cfg.get("attn_implementation") or None,
    }


def resolve_data_path(cfg: dict[str, Any]) -> Path:
    p = Path(cfg["data_path"])
    return p if p.is_absolute() else (REPO_ROOT / p)


def resolve_repo_path(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (REPO_ROOT / p)


def is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda oom" in msg


def is_cuda_error(exc: BaseException) -> bool:
    """Recoverable GPU failures (OOM, driver reset, async kernel errors)."""
    if is_oom_error(exc):
        return True
    msg = str(exc).lower()
    return "cuda error" in msg or "cublas" in msg or "cudnn" in msg


def cleanup_cuda(*objects: Any) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except RuntimeError:
        pass
    try:
        torch.cuda.empty_cache()
    except RuntimeError:
        pass


def build_tokenizer(model_id: str, cache_dir: str, *, trust_remote_code: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _tokenize_fn(tokenizer, max_length: int):
    def tokenize_with_mask(ex):
        prefix_ids = tokenizer(ex["prefix"], add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(
            ex["target"] + tokenizer.eos_token, add_special_tokens=False
        )["input_ids"]
        if len(prefix_ids) + len(target_ids) > max_length:
            return {"input_ids": [], "attention_mask": [], "labels": []}
        input_ids = prefix_ids + target_ids
        labels = [-100] * len(prefix_ids) + target_ids
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    return tokenize_with_mask


def prepare_train_eval(
    raw_dataset,
    tokenizer,
    *,
    max_length: int,
    eval_test_size: float = 0.02,
    eval_seed: int = 42,
    proxy_fraction: float | None = None,
    proxy_seed: int | None = None,
    num_proc: int = 4,
):
    split = raw_dataset.train_test_split(test_size=eval_test_size, seed=eval_seed)
    tokenize_with_mask = _tokenize_fn(tokenizer, max_length)
    train_ds = split["train"].map(
        tokenize_with_mask,
        remove_columns=split["train"].column_names,
        num_proc=num_proc,
    )
    eval_ds = split["test"].map(
        tokenize_with_mask,
        remove_columns=split["test"].column_names,
        num_proc=num_proc,
    )

    def has_target(ex):
        return any(l != -100 for l in ex["labels"])

    train_ds = train_ds.filter(has_target)
    eval_ds = eval_ds.filter(has_target)

    if proxy_fraction is not None:
        n = max(1, int(len(train_ds) * proxy_fraction))
        train_ds = train_ds.shuffle(seed=proxy_seed or 0).select(range(n))

    return train_ds, eval_ds


def build_qlora_model(
    model_id: str,
    cache_dir: str,
    *,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    trust_remote_code: bool = False,
    attn_implementation: str | None = None,
):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    load_kwargs: dict[str, Any] = {}
    if trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    if attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=cache_dir,
        **load_kwargs,
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=int(lora_r),
        lora_alpha=int(lora_alpha),
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=float(lora_dropout),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    return model


def build_training_args(
    *,
    output_dir: str,
    hyperparams: dict[str, Any],
    num_train_epochs: float,
    train_len: int,
    proxy: bool,
    report_to: str = "none",
    run_name: str = "codellama-qlora",
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 1,
    dataloader_num_workers: int = 0,
    neftune_noise_alpha: float = 5,
) -> TrainingArguments:
    grad_accum = int(hyperparams["gradient_accumulation_steps"])
    micro_batch = max(1, int(per_device_train_batch_size))
    effective_batch = max(1, micro_batch * grad_accum)
    steps_per_epoch = max(1, train_len // effective_batch)
    eval_steps = max(50, int(0.1 * steps_per_epoch))

    kwargs: dict[str, Any] = dict(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=micro_batch,
        per_device_eval_batch_size=max(1, int(per_device_eval_batch_size)),
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=float(hyperparams["learning_rate"]),
        lr_scheduler_type="cosine",
        warmup_ratio=float(hyperparams["warmup_ratio"]),
        weight_decay=float(hyperparams["weight_decay"]),
        optim="paged_adamw_8bit",
        bf16=True,
        tf32=True,
        neftune_noise_alpha=float(neftune_noise_alpha),
        logging_steps=10,
        dataloader_num_workers=int(dataloader_num_workers),
        report_to=report_to,
        run_name=run_name,
    )
    if proxy:
        kwargs.update(
            eval_strategy="epoch",
            save_strategy="no",
            load_best_model_at_end=False,
        )
    else:
        kwargs.update(
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=eval_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        )
    return TrainingArguments(**kwargs)


def remove_path_quiet(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.is_file():
        path.unlink(missing_ok=True)
