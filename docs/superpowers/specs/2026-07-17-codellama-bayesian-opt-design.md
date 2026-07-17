# Design: Bayesian Optimization for CodeLlama-7B QLoRA (RTX 3060 12GB)

**Date:** 2026-07-17  
**Status:** Ready for user review  
 
**Scope:** Hyperparameter search for `main/main.py` (CodeLlama-7B QLoRA), using Plan B completion dataset.

## Goal

Find a better training hyperparameter set for CodeLlama-7B QLoRA on Java function-completion by minimizing validation `eval_loss` via Bayesian Optimization (Optuna TPE), runnable on a single RTX 3060 12GB / 32GB RAM machine.

Final model quality is judged later on multiple benchmarks (not only HumanEval-Java). BO uses `eval_loss` as a fast, stable proxy.

## Non-goals

- Optimizing pass@1 / HumanEval-Java inside each BO trial
- Changing Plan B dataset construction
- Running bf16 full LoRA Plan B (`finetune_lora7b_planB/main_qwen.py`) on 12GB
- Parallel multi-GPU trials
- FlashAttention-2 as a hard requirement on Windows/3060

## Constraints

| Item | Value |
|------|--------|
| Model | `codellama/CodeLlama-7b-hf` |
| Quantization | QLoRA 4-bit nf4, bf16 compute |
| GPU | 1× RTX 3060 12GB |
| RAM | 32GB |
| Data | `finetune_lora7b_planB/data/java_completion_train.jsonl` |
| Baseline script | `main/main.py` |

Fixed every trial (not searched):

- `per_device_train_batch_size=1`, `per_device_eval_batch_size=1`
- `MAX_LENGTH=1024`
- `gradient_checkpointing=True`
- `optim=paged_adamw_8bit`, `lr_scheduler_type=cosine`
- `neftune_noise_alpha=5`
- LoRA `target_modules`: q/k/v/o + gate/up/down
- `lora_alpha = 2 × lora_r`

## Approach

**Optuna TPE** study with SQLite storage for resume.

Alternative frameworks (Ray Tune, Hyperopt) were considered and rejected as heavier or weaker for single-GPU resume/pruning workflows.

## Architecture

```
main/
  main.py                 # full train; accepts optional best_params.json
  bayes_opt.py            # Optuna study entrypoint
  train_trial.py          # one trial: proxy train → eval_loss
  training_common.py      # shared builders (datasets, model, TrainingArguments)
  configs/bo_defaults.yaml
  studies/
    codellama_bo.db       # Optuna SQLite
    best_params.json      # written after study
```

```
Optuna study (SQLite)
  → suggest 6 hyperparams
  → train_trial (QLoRA, ~8% train, 1 epoch)
  → report eval_loss (minimize)
  → OOM → prune / inf (study continues)
  → after N trials → best_params.json
  → main.py full train (3 epochs) with best params
```

Shared helpers live in `training_common.py` so trial and full train stay consistent. `train_trial.py` stays separate so each trial can fully tear down the model and free VRAM.

## Search space

| Param | Type | Range | Baseline in `main.py` |
|-------|------|-------|------------------------|
| `learning_rate` | log-uniform | `1e-5` … `3e-4` | `1e-4` |
| `weight_decay` | uniform | `0.0` … `0.1` | `0.01` |
| `warmup_ratio` | uniform | `0.01` … `0.1` | `0.03` |
| `lora_dropout` | uniform | `0.0` … `0.1` | `0.05` |
| `lora_r` | categorical | `{8, 16, 32, 64}` | `32` |
| `gradient_accumulation_steps` | categorical | `{8, 16, 32}` | `16` |

Objective: **minimize** final `eval_loss` from `trainer.evaluate()` after the proxy train.

## Proxy protocol (per trial)

1. Load full JSONL once (or reuse cached tokenized split where practical).
2. Train/val split: `train_test_split(test_size=0.02, seed=42)` — same as `main.py`.
3. Subsample **~8%** of the train split for the trial (`seed` derived from `trial.number` for reproducibility).
4. Keep the full validation split (or a fixed small eval subsample if wall-clock is too high; default = full val).
5. Train **1 epoch**, `save_strategy="no"`.
6. Evaluate → return `eval_loss`.
7. Delete temporary trial output dir under `./models/java-codellama-lora/bo_trial_{n}/`.

Default `n_trials`: **40** (overridable in yaml).

Sampler: Optuna TPE. Optional MedianPruner if mid-epoch eval is enabled later; initial implementation may report only end-of-trial metrics.

## Error handling

| Case | Behavior |
|------|----------|
| CUDA OOM | Catch, free CUDA memory, raise `optuna.TrialPruned()`; do not kill the study |
| Other exceptions | Log traceback; mark trial failed; study remains resumable via SQLite |
| Process interrupt | Resume with same study name / DB path |

Every trial ends in `finally`: delete model/trainer refs, `gc.collect()`, `torch.cuda.empty_cache()`, remove trial temp dir.

## Integration with `main.py`

1. Refactor `main.py` to call shared builders from `training_common.py`.
2. If `main/studies/best_params.json` exists (or `--config` is passed), use those hyperparams for full training.
3. Otherwise keep current defaults (`r=32`, `lr=1e-4`, `grad_accum=16`, …).
4. Full train remains **3 epochs**, full dataset, checkpoints + `load_best_model_at_end` as today.

Usage:

```powershell
python main/bayes_opt.py
python main/main.py
# or
python main/main.py --config main/studies/best_params.json
```

## Config file (`configs/bo_defaults.yaml`)

Holds at least:

- model id, data path, cache dir, max length
- `proxy_train_fraction` (default `0.08`)
- `proxy_epochs` (default `1`)
- `n_trials` (default `40`)
- search space bounds / categoricals
- study DB path and study name
- output paths for best params and trial dirs

## Dependencies

Add `optuna` (and keep existing `bitsandbytes`, `peft`, `transformers`, etc.). Document in root or `main` requirements as appropriate for this repo layout.

## Success criteria

1. `python main/bayes_opt.py` completes/resumes a study on RTX 3060 without unrecovered OOM crashes.
2. `best_params.json` is written with the six searched keys + derived `lora_alpha`.
3. `python main/main.py` can train with those params on the full Plan B dataset.
4. Trial isolation: VRAM is reclaimable between trials (no progressive OOM after a few successful trials).

## Out of scope for first implementation

- WandB logging per trial (optional later; full train keeps existing WandB behavior)
- Automatic multi-benchmark evaluation after BO
- Searching `MAX_LENGTH`, batch size, or quantization settings
