# Design: Bayesian Optimization cho CodeLlama-7B QLoRA (RTX 3060 12GB)

**Ngày:** 2026-07-17  
**Trạng thái:** Chờ người dùng review  
**Phạm vi:** Tối ưu hyperparameter cho `main/main.py` (CodeLlama-7B QLoRA), dùng dataset completion Plan B.

## Mục tiêu

Tìm bộ hyperparameter training tốt hơn cho CodeLlama-7B QLoRA trên bài Java function-completion bằng Bayesian Optimization (Optuna TPE), tối thiểu hóa `eval_loss` trên validation, chạy được trên máy 1× RTX 3060 12GB / 32GB RAM.

Chất lượng model cuối sẽ đánh giá sau trên nhiều benchmark (không chỉ HumanEval-Java). BO dùng `eval_loss` làm proxy nhanh và ổn định.

## Ngoài phạm vi

- Tối ưu pass@1 / HumanEval-Java bên trong mỗi trial BO
- Thay đổi cách build dataset Plan B
- Chạy bf16 full LoRA Plan B (`finetune_lora7b_planB/main_qwen.py`) trên 12GB
- Chạy song song nhiều GPU
- Bắt buộc FlashAttention-2 trên Windows/3060

## Ràng buộc

| Mục | Giá trị |
|-----|---------|
| Model | `codellama/CodeLlama-7b-hf` |
| Quantization | QLoRA 4-bit nf4, compute bf16 |
| GPU | 1× RTX 3060 12GB |
| RAM | 32GB |
| Data | `finetune_lora7b_planB/data/java_completion_train.jsonl` |
| Script baseline | `main/main.py` |

Cố định mọi trial (không search):

- `per_device_train_batch_size=1`, `per_device_eval_batch_size=1`
- `MAX_LENGTH=1024`
- `gradient_checkpointing=True`
- `optim=paged_adamw_8bit`, `lr_scheduler_type=cosine`
- `neftune_noise_alpha=5`
- LoRA `target_modules`: q/k/v/o + gate/up/down
- `lora_alpha = 2 × lora_r`

## Hướng tiếp cận

**Optuna TPE** với SQLite storage để resume khi bị gián đoạn.

Đã cân nhắc Ray Tune / Hyperopt và loại bỏ vì nặng hơn hoặc kém hơn về resume/pruning trên single-GPU.

## Kiến trúc

```
main/
  main.py                 # full train; nhận optional best_params.json
  bayes_opt.py            # entrypoint Optuna study
  train_trial.py          # một trial: proxy train → eval_loss
  training_common.py      # builder dùng chung (datasets, model, TrainingArguments)
  configs/bo_defaults.yaml
  studies/
    codellama_bo.db       # Optuna SQLite
    best_params.json      # ghi sau khi study xong
```

```
Optuna study (SQLite)
  → suggest 6 hyperparams
  → train_trial (QLoRA, ~8% train, 1 epoch)
  → report eval_loss (minimize)
  → OOM → TrialPruned (study tiếp tục)
  → sau N trials → best_params.json
  → main.py full train (3 epochs) với best params
```

Logic dùng chung nằm trong `training_common.py` để trial và full train nhất quán. `train_trial.py` tách riêng để mỗi trial teardown model và giải phóng VRAM sạch.

## Search space

| Param | Kiểu | Range | Baseline trong `main.py` |
|-------|------|-------|--------------------------|
| `learning_rate` | log-uniform | `1e-5` … `3e-4` | `1e-4` |
| `weight_decay` | uniform | `0.0` … `0.1` | `0.01` |
| `warmup_ratio` | uniform | `0.01` … `0.1` | `0.03` |
| `lora_dropout` | uniform | `0.0` … `0.1` | `0.05` |
| `lora_r` | categorical | `{8, 16, 32, 64}` | `32` |
| `gradient_accumulation_steps` | categorical | `{8, 16, 32}` | `16` |

Objective: **minimize** `eval_loss` từ `trainer.evaluate()` sau proxy train.

## Proxy protocol (mỗi trial)

1. Load JSONL (hoặc tái dùng tokenized split đã cache nếu khả thi).
2. Train/val split: `train_test_split(test_size=0.02, seed=42)` — giống `main.py`.
3. Subsample **~8%** train cho trial (`seed` suy từ `trial.number` để tái lập).
4. Giữ nguyên validation split đầy đủ (mặc định); chỉ subsample eval nếu wall-clock quá cao.
5. Train **1 epoch**, `save_strategy="no"`.
6. Evaluate → trả về `eval_loss`.
7. Xóa thư mục tạm `./models/java-codellama-lora/bo_trial_{n}/`.

`n_trials` mặc định: **40** (chỉnh được trong yaml).

Sampler: Optuna TPE. MedianPruner tùy chọn nếu sau này bật mid-epoch eval; bản đầu chỉ report metric cuối trial.

## Xử lý lỗi

| Trường hợp | Hành vi |
|------------|---------|
| CUDA OOM | Bắt lỗi, giải phóng CUDA, raise `optuna.TrialPruned()`; không kill study |
| Exception khác | Log traceback; mark trial failed; study vẫn resume được qua SQLite |
| Process bị ngắt | Resume cùng study name / DB path |

Mọi trial kết thúc trong `finally`: xóa ref model/trainer, `gc.collect()`, `torch.cuda.empty_cache()`, xóa thư mục tạm trial.

## Tích hợp `main.py`

1. Refactor `main.py` gọi builder từ `training_common.py`.
2. Nếu có `main/studies/best_params.json` (hoặc truyền `--config`) → dùng bộ hyperparams đó cho full train.
3. Không có best params → giữ defaults hiện tại (`r=32`, `lr=1e-4`, `grad_accum=16`, …).
4. Full train vẫn **3 epochs**, full dataset, checkpoint + `load_best_model_at_end` như hiện tại.

Cách chạy:

```powershell
python main/bayes_opt.py
python main/main.py
# hoặc
python main/main.py --config main/studies/best_params.json
```

## File cấu hình (`configs/bo_defaults.yaml`)

Ít nhất gồm:

- model id, data path, cache dir, max length
- `proxy_train_fraction` (mặc định `0.08`)
- `proxy_epochs` (mặc định `1`)
- `n_trials` (mặc định `40`)
- biên / categorical của search space
- đường dẫn study DB và study name
- đường dẫn best params và thư mục trial

## Dependencies

Thêm `optuna` (giữ `bitsandbytes`, `peft`, `transformers`, …). Ghi vào requirements phù hợp layout repo.

## Tiêu chí thành công

1. `python main/bayes_opt.py` chạy/resume study trên RTX 3060 mà không crash không phục hồi vì OOM.
2. Ghi được `best_params.json` với 6 key đã search + `lora_alpha` suy ra.
3. `python main/main.py` train được full dataset Plan B với bộ params đó.
4. Cách ly trial: VRAM thu hồi được giữa các trial (không OOM dần sau vài trial thành công).

## Ngoài phạm vi bản đầu

- WandB logging từng trial (có thể thêm sau; full train giữ WandB hiện có)
- Tự động evaluate multi-benchmark sau BO
- Search `MAX_LENGTH`, batch size, hoặc cấu hình quantization
