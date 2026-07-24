# Bayesian Optimization / Evol completion train — CodeLlama QLoRA (RTX 3060)

## Cài thêm

```powershell
pip install optuna pyyaml
```

Trong `.env` (repo root):

```env
HF_HOME=D:/cache/huggingface
```

## 1) Build dataset (Evol + Magicoder → method-body completion)

Nguồn:
- `nickrosh/Evol-Instruct-Code-80k-v1`
- `ise-uiuc/Magicoder-Evol-Instruct-110K`
- `ise-uiuc/Magicoder-OSS-Instruct-75K`

Cả ba map cùng format: `prefix` = docs + signature + `{`, `target` = body + `}`.
Chỉ lấy method đầu mỗi block (an toàn ngữ cảnh). Không giữ chat/`[INST]`.

```powershell
python data/build_evol_completion_dataset.py
# output: data/jsonl/evol_java_completion_train.jsonl
```

## 2) Full train với best params (cùng recipe BO hiện tại)

```powershell
python train/main.py --config train/studies/best_params.json
```

Output mặc định: `models/java-codellama-lora/evol_bo_best`

## 3) BO (nếu chạy lại search trên data Evol)

```powershell
python train/bayes_opt.py
python train/bayes_opt.py --n-trials 1
python train/bayes_opt.py --export-best
```

Proxy: ~8% data, 1 epoch, minimize `eval_loss`.
