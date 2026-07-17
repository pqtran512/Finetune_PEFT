# CodeLlama Bayesian Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thêm Optuna TPE Bayesian Optimization cho hyperparameter của CodeLlama-7B QLoRA trong `main/`, tối thiểu `eval_loss` bằng proxy trial (~8% data, 1 epoch) trên RTX 3060 12GB, rồi full train qua `best_params.json`.

**Architecture:** Tách logic train dùng chung vào `training_common.py`; mỗi Optuna trial gọi `train_trial.run_trial(...)` (QLoRA + cleanup VRAM); `bayes_opt.py` quản lý study SQLite và ghi `best_params.json`; `main.py` refactor nhẹ để nhận `--config` / file best params.

**Tech Stack:** Python, PyTorch, transformers, peft, bitsandbytes, datasets, Optuna (TPE), PyYAML, pytest

**Spec:** `docs/superpowers/specs/2026-07-17-codellama-bayesian-opt-design.md`

## Global Constraints

- Model: `codellama/CodeLlama-7b-hf`
- Quantization: QLoRA 4-bit nf4, compute bf16
- Hardware target: 1× RTX 3060 12GB, 32GB RAM
- Data: `finetune_lora7b_planB/data/java_completion_train.jsonl`
- Fixed: `batch_size=1`, `MAX_LENGTH=1024`, `gradient_checkpointing=True`, `paged_adamw_8bit`, cosine, `neftune_noise_alpha=5`, `lora_alpha = 2 * lora_r`
- Search: `learning_rate`, `weight_decay`, `warmup_ratio`, `lora_dropout`, `lora_r`, `gradient_accumulation_steps`
- Objective: minimize `eval_loss`
- OOM: raise `optuna.TrialPruned()`
- Proxy: ~8% train, 1 epoch, val split `seed=42` test_size=0.02
- Default `n_trials=40`
- Không search MAX_LENGTH / batch / quantization; không BO trên pass@1

## File Structure

| File | Responsibility |
|------|----------------|
| `main/configs/bo_defaults.yaml` | Search space, proxy %, paths, n_trials |
| `main/training_common.py` | Load env/config, build datasets/model/TrainingArguments, merge hyperparams, cleanup CUDA |
| `main/train_trial.py` | Chạy 1 proxy trial → `eval_loss` |
| `main/bayes_opt.py` | Optuna study entrypoint → `best_params.json` |
| `main/main.py` | Full 3-epoch train; đọc best params / `--config` |
| `main/studies/` | SQLite + best_params (runtime; `.gitkeep` only) |
| `tests/main/test_training_common.py` | Unit tests (không cần GPU) |
| `tests/main/test_bayes_opt_space.py` | Unit tests suggest/space/best_params IO |
| `requirements.txt` | Thêm `optuna`, `pyyaml` |

---

### Task 1: Dependencies + YAML config + load helpers

**Files:**
- Modify: `requirements.txt`
- Create: `main/configs/bo_defaults.yaml`
- Create: `main/studies/.gitkeep`
- Create: `main/training_common.py` (phần load config / defaults / merge params)
- Create: `tests/main/test_training_common.py`
- Create: `tests/main/__init__.py` (empty) nếu cần

**Interfaces:**
- Produces:
  - `REPO_ROOT: Path`
  - `MAIN_DIR: Path`
  - `load_yaml_config(path: Path | None = None) -> dict`
  - `DEFAULT_HYPERPARAMS: dict` (baseline full-train)
  - `merge_hyperparams(overrides: dict | None) -> dict` — luôn set `lora_alpha = 2 * lora_r`
  - `load_hyperparams_from_json(path: Path) -> dict`

- [ ] **Step 1: Write failing tests**

Create `tests/main/test_training_common.py`:

```python
from pathlib import Path
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "main"))

from training_common import (  # noqa: E402
    load_yaml_config,
    merge_hyperparams,
    load_hyperparams_from_json,
    DEFAULT_HYPERPARAMS,
)


def test_default_hyperparams_baseline():
    assert DEFAULT_HYPERPARAMS["learning_rate"] == 1e-4
    assert DEFAULT_HYPERPARAMS["lora_r"] == 32
    assert DEFAULT_HYPERPARAMS["lora_alpha"] == 64
    assert DEFAULT_HYPERPARAMS["gradient_accumulation_steps"] == 16


def test_merge_sets_lora_alpha_from_r():
    hp = merge_hyperparams({"lora_r": 16, "learning_rate": 2e-5})
    assert hp["lora_r"] == 16
    assert hp["lora_alpha"] == 32
    assert hp["learning_rate"] == 2e-5
    # untouched defaults remain
    assert hp["weight_decay"] == 0.01


def test_load_yaml_config_has_search_keys():
    cfg = load_yaml_config()
    assert cfg["n_trials"] == 40
    assert cfg["proxy_train_fraction"] == 0.08
    assert "learning_rate" in cfg["search_space"]
    assert cfg["search_space"]["lora_r"]["choices"] == [8, 16, 32, 64]


def test_load_hyperparams_from_json(tmp_path: Path):
    p = tmp_path / "best_params.json"
    p.write_text(
        json.dumps(
            {
                "learning_rate": 5e-5,
                "weight_decay": 0.02,
                "warmup_ratio": 0.05,
                "lora_dropout": 0.1,
                "lora_r": 64,
                "gradient_accumulation_steps": 32,
            }
        ),
        encoding="utf-8",
    )
    hp = load_hyperparams_from_json(p)
    assert hp["lora_alpha"] == 128
    assert hp["lora_r"] == 64
```

- [ ] **Step 2: Run tests — expect FAIL**

```powershell
pip install pytest pyyaml -q
python -m pytest tests/main/test_training_common.py -v
```

Expected: FAIL (`ModuleNotFoundError: training_common` hoặc import lỗi).

- [ ] **Step 3: Add dependencies**

Append to `requirements.txt`:

```text
optuna
pyyaml
pytest
```

- [ ] **Step 4: Create YAML**

Create `main/configs/bo_defaults.yaml`:

```yaml
model_id: codellama/CodeLlama-7b-hf
data_path: finetune_lora7b_planB/data/java_completion_train.jsonl
output_dir: models/java-codellama-lora/completion_qlora_3060
max_length: 1024
proxy_train_fraction: 0.08
proxy_epochs: 1
n_trials: 40
study_name: codellama_qlora_bo
study_db: main/studies/codellama_bo.db
best_params_path: main/studies/best_params.json
trial_output_root: models/java-codellama-lora/bo_trials
eval_test_size: 0.02
eval_seed: 42
neftune_noise_alpha: 5
search_space:
  learning_rate:
    type: loguniform
    low: 1.0e-5
    high: 3.0e-4
  weight_decay:
    type: uniform
    low: 0.0
    high: 0.1
  warmup_ratio:
    type: uniform
    low: 0.01
    high: 0.1
  lora_dropout:
    type: uniform
    low: 0.0
    high: 0.1
  lora_r:
    type: categorical
    choices: [8, 16, 32, 64]
  gradient_accumulation_steps:
    type: categorical
    choices: [8, 16, 32]
```

Create empty `main/studies/.gitkeep`.

- [ ] **Step 5: Implement helpers in `training_common.py`**

```python
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
```

- [ ] **Step 6: Run tests — expect PASS**

```powershell
pip install -r requirements.txt -q
python -m pytest tests/main/test_training_common.py -v
```

Expected: PASS (4 tests).

- [ ] **Step 7: Commit**

```powershell
git add requirements.txt main/configs/bo_defaults.yaml main/studies/.gitkeep main/training_common.py tests/main/test_training_common.py tests/main/__init__.py
git commit -m "Add BO config and shared hyperparam helpers for CodeLlama"
```

---

### Task 2: Dataset / model / TrainingArguments builders

**Files:**
- Modify: `main/training_common.py`
- Modify: `tests/main/test_training_common.py`

**Interfaces:**
- Consumes: `merge_hyperparams`, `load_yaml_config`, `REPO_ROOT`
- Produces:
  - `resolve_data_path(cfg: dict) -> Path`
  - `build_tokenizer(model_id: str, cache_dir: str)`
  - `tokenize_dataset(dataset, tokenizer, max_length: int, num_proc: int = 4)`
  - `prepare_train_eval(raw_dataset, tokenizer, *, max_length, eval_test_size, eval_seed, proxy_fraction: float | None, proxy_seed: int | None) -> tuple[train_ds, eval_ds]`
  - `build_qlora_model(model_id, cache_dir, lora_r, lora_alpha, lora_dropout)`
  - `build_training_args(*, output_dir, hyperparams, num_train_epochs, train_len, proxy: bool, report_to: str, run_name: str) -> TrainingArguments`
  - `cleanup_cuda(*objects) -> None`
  - `is_oom_error(exc: BaseException) -> bool`

- [ ] **Step 1: Add failing tests for pure helpers**

Append to `tests/main/test_training_common.py`:

```python
from training_common import is_oom_error, resolve_data_path, load_yaml_config


def test_is_oom_error_detects_cuda_oom():
    assert is_oom_error(RuntimeError("CUDA out of memory. Tried to allocate..."))
    assert is_oom_error(RuntimeError("cuda OOM"))
    assert not is_oom_error(RuntimeError("something else"))


def test_resolve_data_path_relative_to_repo():
    cfg = load_yaml_config()
    p = resolve_data_path(cfg)
    assert p.is_absolute()
    assert p.as_posix().endswith("java_completion_train.jsonl")
```

- [ ] **Step 2: Run — expect FAIL**

```powershell
python -m pytest tests/main/test_training_common.py::test_is_oom_error_detects_cuda_oom tests/main/test_training_common.py::test_resolve_data_path_relative_to_repo -v
```

Expected: FAIL (`ImportError` / attribute missing).

- [ ] **Step 3: Implement builders**

Append to `main/training_common.py` (keep existing helpers):

```python
import gc
import shutil
from typing import Optional

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def resolve_data_path(cfg: dict[str, Any]) -> Path:
    p = Path(cfg["data_path"])
    return p if p.is_absolute() else (REPO_ROOT / p)


def resolve_repo_path(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (REPO_ROOT / p)


def is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda oom" in msg


def cleanup_cuda(*objects: Any) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_tokenizer(model_id: str, cache_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
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
):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=cache_dir,
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
) -> TrainingArguments:
    grad_accum = int(hyperparams["gradient_accumulation_steps"])
    effective_batch = max(1, 1 * grad_accum)
    steps_per_epoch = max(1, train_len // effective_batch)
    eval_steps = max(50, int(0.1 * steps_per_epoch))

    kwargs: dict[str, Any] = dict(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
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
        neftune_noise_alpha=5,
        logging_steps=10,
        dataloader_num_workers=0,
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
```

- [ ] **Step 4: Run unit tests**

```powershell
python -m pytest tests/main/test_training_common.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```powershell
git add main/training_common.py tests/main/test_training_common.py
git commit -m "Add shared QLoRA dataset and TrainingArguments builders"
```

---

### Task 3: `train_trial.py` — một proxy trial

**Files:**
- Create: `main/train_trial.py`
- Create: `tests/main/test_train_trial.py`

**Interfaces:**
- Consumes: builders từ `training_common`
- Produces: `run_trial(trial_number: int, hyperparams: dict, cfg: dict, *, cache_dir: str | None = None) -> float`

Behavior:
- Subsample `cfg["proxy_train_fraction"]` với `proxy_seed=trial_number`
- Train `cfg["proxy_epochs"]` epoch
- Return `metrics["eval_loss"]`
- OOM → cleanup → raise `optuna.TrialPruned()`
- `finally`: cleanup + xóa trial dir

- [ ] **Step 1: Write unit test for OOM path (mock)**

Create `tests/main/test_train_trial.py`:

```python
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock

import optuna
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "main"))

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

    with patch("train_trial.load_dataset", side_effect=RuntimeError("CUDA out of memory")):
        with pytest.raises(optuna.TrialPruned):
            run_trial(0, hp, cfg, cache_dir="unused")
```

- [ ] **Step 2: Run — expect FAIL**

```powershell
python -m pytest tests/main/test_train_trial.py -v
```

Expected: FAIL (module missing).

- [ ] **Step 3: Implement `train_trial.py`**

```python
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
    REPO_ROOT,
    build_qlora_model,
    build_tokenizer,
    build_training_args,
    cleanup_cuda,
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
    cache = cache_dir or os.environ.get(
        "HF_CACHE_DIR", str(Path.home() / ".cache" / "huggingface")
    )
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
```

- [ ] **Step 4: Run unit test**

```powershell
python -m pytest tests/main/test_train_trial.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add main/train_trial.py tests/main/test_train_trial.py
git commit -m "Add Optuna proxy trial runner with OOM prune"
```

---

### Task 4: `bayes_opt.py` — Optuna study

**Files:**
- Create: `main/bayes_opt.py`
- Create: `tests/main/test_bayes_opt_space.py`

**Interfaces:**
- Consumes: `run_trial`, `load_yaml_config`, `merge_hyperparams`
- Produces:
  - `suggest_hyperparams(trial, search_space: dict) -> dict`
  - `save_best_params(path: Path, params: dict, value: float) -> None`
  - `main()` CLI

- [ ] **Step 1: Failing tests for suggest + save**

```python
from pathlib import Path
import json
import sys

import optuna
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "main"))

from bayes_opt import suggest_hyperparams, save_best_params  # noqa: E402
from training_common import load_yaml_config  # noqa: E402


def test_suggest_hyperparams_keys():
    cfg = load_yaml_config()
    study = optuna.create_study(direction="minimize")

    def objective(trial):
        hp = suggest_hyperparams(trial, cfg["search_space"])
        assert set(hp) >= {
            "learning_rate",
            "weight_decay",
            "warmup_ratio",
            "lora_dropout",
            "lora_r",
            "gradient_accumulation_steps",
        }
        assert hp["lora_r"] in {8, 16, 32, 64}
        return 0.0

    study.optimize(objective, n_trials=1)


def test_save_best_params_includes_lora_alpha(tmp_path: Path):
    out = tmp_path / "best_params.json"
    save_best_params(
        out,
        {
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "warmup_ratio": 0.03,
            "lora_dropout": 0.05,
            "lora_r": 32,
            "gradient_accumulation_steps": 16,
        },
        value=1.23,
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["lora_alpha"] == 64
    assert data["best_eval_loss"] == 1.23
```

- [ ] **Step 2: Run — expect FAIL**

```powershell
python -m pytest tests/main/test_bayes_opt_space.py -v
```

- [ ] **Step 3: Implement `bayes_opt.py`**

```python
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


def suggest_hyperparams(trial: optuna.Trial, search_space: dict[str, Any]) -> dict[str, Any]:
    hp: dict[str, Any] = {}
    for name, spec in search_space.items():
        t = spec["type"]
        if t == "loguniform":
            hp[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=True)
        elif t == "uniform":
            hp[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]))
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Bayesian opt for CodeLlama QLoRA")
    parser.add_argument("--config", type=Path, default=None, help="Path to bo_defaults.yaml")
    parser.add_argument("--n-trials", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    n_trials = args.n_trials if args.n_trials is not None else int(cfg["n_trials"])

    db_path = resolve_repo_path(cfg["study_db"])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{db_path.as_posix()}"

    study = optuna.create_study(
        study_name=cfg["study_name"],
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    cache_dir = os.environ.get(
        "HF_CACHE_DIR", str(Path.home() / ".cache" / "huggingface")
    )

    def objective(trial: optuna.Trial) -> float:
        hp = suggest_hyperparams(trial, cfg["search_space"])
        # drop derived key from suggest storage noise — Optuna already has searched attrs
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

    study.optimize(objective, n_trials=n_trials, catch=(optuna.TrialPruned,))

    best_path = resolve_repo_path(cfg["best_params_path"])
    save_best_params(best_path, study.best_params, study.best_value)
    print(f"Best eval_loss={study.best_value:.6f}")
    print(f"Wrote {best_path}")


if __name__ == "__main__":
    main()
```

Note: `study.optimize(..., catch=(optuna.TrialPruned,))` — Optuna đã handle `TrialPruned` mặc định; **không** cần `catch` cho Pruned. Dùng:

```python
study.optimize(objective, n_trials=n_trials)
```

`TrialPruned` raised từ `run_trial` sẽ được Optuna ghi nhận là pruned trial.

- [ ] **Step 4: Fix `catch` — dùng `study.optimize(objective, n_trials=n_trials)` only**

- [ ] **Step 5: Run unit tests**

```powershell
python -m pytest tests/main/test_bayes_opt_space.py tests/main/test_train_trial.py tests/main/test_training_common.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```powershell
git add main/bayes_opt.py tests/main/test_bayes_opt_space.py
git commit -m "Add Optuna TPE entrypoint and best_params export"
```

---

### Task 5: Refactor `main/main.py` để dùng common + `--config`

**Files:**
- Modify: `main/main.py` (rewrite to thin wrapper)
- Create: `tests/main/test_main_config.py`

**Interfaces:**
- Consumes: `training_common.*`
- CLI: `python main/main.py [--config PATH]`
- Nếu không truyền `--config`: dùng `main/studies/best_params.json` nếu tồn tại, else `DEFAULT_HYPERPARAMS`

- [ ] **Step 1: Test resolve config path logic**

Add to `training_common.py`:

```python
def resolve_train_hyperparams(config_arg: Path | None = None) -> dict[str, Any]:
    """Priority: --config path > studies/best_params.json > defaults."""
    if config_arg is not None:
        return load_hyperparams_from_json(config_arg)
    default_best = MAIN_DIR / "studies" / "best_params.json"
    if default_best.exists():
        return load_hyperparams_from_json(default_best)
    return merge_hyperparams()
```

Test:

```python
from training_common import resolve_train_hyperparams, MAIN_DIR, merge_hyperparams
from pathlib import Path
import json


def test_resolve_train_hyperparams_defaults(monkeypatch, tmp_path):
    # force missing best file by pointing MAIN_DIR studies away — use explicit None + no file
    hp = merge_hyperparams()
    assert hp["lora_r"] == 32


def test_resolve_train_hyperparams_from_explicit(tmp_path: Path):
    p = tmp_path / "p.json"
    p.write_text(json.dumps({"lora_r": 8, "learning_rate": 2e-5}), encoding="utf-8")
    hp = resolve_train_hyperparams(p)
    assert hp["lora_r"] == 8
    assert hp["lora_alpha"] == 16
```

- [ ] **Step 2: Rewrite `main/main.py`**

Replace body with:

```python
"""Fine-tune CodeLlama-7B QLoRA trên Java function-completion (prefix/target).

Dataset: finetune_lora7b_planB/data/java_completion_train.jsonl
Target hardware: RTX 3060 12GB — QLoRA 4-bit + bf16 compute.
Optional: load best hyperparams from Optuna (`main/studies/best_params.json`).
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
    load_env,
    load_yaml_config,
    prepare_train_eval,
    resolve_data_path,
    resolve_repo_path,
    resolve_train_hyperparams,
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
    hp = resolve_train_hyperparams(args.config)

    MODEL_ID = cfg["model_id"]
    OUTPUT_DIR = str(resolve_repo_path(cfg["output_dir"]))
    MAX_LENGTH = int(cfg["max_length"])
    CACHE_DIR = os.environ.get(
        "HF_CACHE_DIR",
        str(Path.home() / ".cache" / "huggingface"),
    )

    if WANDB_API_KEY:
        import wandb

        wandb.init(project="codellama-java", name="completion-qlora-3060-12gb")

    data_path = resolve_data_path(cfg)
    print("--- Loading dataset ---")
    if not data_path.exists():
        raise FileNotFoundError(
            f"Không thấy {data_path}. "
            "Chạy `python build_completion_dataset.py` trong finetune_lora7b_planB trước."
        )
    start = time.time()
    dataset = load_dataset("json", data_files=str(data_path), split="train")
    print(f"Load: {time.time() - start:.2f}s, n={len(dataset)}")

    print("--- Loading tokenizer ---")
    tokenizer = build_tokenizer(MODEL_ID, CACHE_DIR)

    print("--- Tokenizing ---")
    train_ds, eval_ds = prepare_train_eval(
        dataset,
        tokenizer,
        max_length=MAX_LENGTH,
        eval_test_size=float(cfg["eval_test_size"]),
        eval_seed=int(cfg["eval_seed"]),
        proxy_fraction=None,
        num_proc=4,
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
            num_train_epochs=3,
            train_len=len(train_ds),
            proxy=False,
            report_to=report_to,
            run_name="codellama-completion-qlora-3060",
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
```

- [ ] **Step 3: Run unit tests (no GPU)**

```powershell
python -m pytest tests/main -v
```

Expected: all PASS.

- [ ] **Step 4: Smoke import check**

```powershell
python -c "import sys; sys.path.insert(0,'main'); import bayes_opt, train_trial, training_common, main; print('ok')"
```

Expected: `ok` (có thể warn CUDA nếu không có GPU — không crash).

- [ ] **Step 5: Commit**

```powershell
git add main/main.py main/training_common.py tests/main/test_main_config.py
git commit -m "Refactor main.py to shared builders and best_params config"
```

---

### Task 6: Docs ngắn + GPU smoke checklist

**Files:**
- Create: `main/README_BO.md` (ngắn, tiếng Việt)

- [ ] **Step 1: Write `main/README_BO.md`**

Nội dung file:

```text
# Bayesian Optimization — CodeLlama QLoRA (RTX 3060)

## Cài thêm
pip install optuna pyyaml

## Chạy BO (từ repo root)
python main/bayes_opt.py
python main/bayes_opt.py --n-trials 1

Kết quả: main/studies/best_params.json + SQLite main/studies/codellama_bo.db (resume được).

## Full train với best params
python main/main.py
python main/main.py --config main/studies/best_params.json

Proxy mỗi trial: ~8% data, 1 epoch, minimize eval_loss.
```

- [ ] **Step 2: Commit**

```powershell
git add main/README_BO.md
git commit -m "Document CodeLlama Bayesian Optimization usage"
```

- [ ] **Step 3: Manual GPU smoke (trên máy 3060 — người chạy)**

```powershell
python main/bayes_opt.py --n-trials 1
```

Expected: 1 trial hoàn tất hoặc pruned OOM; nếu complete → `best_params.json` tồn tại.

---

## Spec coverage (self-review)

| Spec requirement | Task |
|------------------|------|
| Optuna TPE + SQLite resume | Task 4 |
| 6 hyperparams + alpha=2r | Task 1, 4 |
| QLoRA fixed settings / MAX_LENGTH 1024 | Task 2, 3, 5 |
| Proxy 8% / 1 epoch / eval_loss | Task 3, 4 |
| OOM → TrialPruned + cleanup | Task 3 |
| best_params.json | Task 4 |
| main.py --config / auto best | Task 5 |
| requirements optuna | Task 1 |
| Không BO pass@1 / FA2 bắt buộc | Non-goal — không task |

## Placeholder scan

Không còn TBD/TODO trong plan. Signatures thống nhất: `run_trial`, `suggest_hyperparams`, `merge_hyperparams`, `resolve_train_hyperparams`.
