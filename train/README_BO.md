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

## Qwen2.5-Coder-32B (H100 80GB)

Cùng pipeline (TPE, proxy 8%, 1 epoch, minimize `eval_loss`). Base model là `Qwen/Qwen2.5-Coder-32B` (completion, không dùng bản Instruct). Data là `data/jsonl/java_completion_train.jsonl`. 1.000 mẫu validation được giữ riêng trước khi cắt 8% proxy, nên eval loss không tính trên mẫu trial đã học. QLoRA 4-bit NF4, micro-batch 2, `max_length` 2048. Effective batch vẫn thuộc {8, 16, 32}.

Study riêng, không đụng DB / best params của CodeLlama.

```powershell
python train/bayes_opt.py --config train/configs/bo_qwen32b_h100.yaml
python train/bayes_opt.py --config train/configs/bo_qwen32b_h100.yaml --n-trials 1
python train/bayes_opt.py --config train/configs/bo_qwen32b_h100.yaml --export-best

python train/main.py --bo-config train/configs/bo_qwen32b_h100.yaml
```

Best params: `train/studies/best_params_qwen32b.json`. Adapter: `models/java-qwen32b-lora/java_completion_bo`.

Nếu trial bị prune vì OOM: trong yaml đặt `per_device_train_batch_size: 1` và `gradient_accumulation_steps.choices: [8, 16, 32]`. Muốn FlashAttention 2 thì đặt `attn_implementation: flash_attention_2` (cần cài `flash-attn`).
