# Finetune PEFT — Java Code Completion

Fine-tune **CodeLlama** / **CodeQwen** với PEFT (LoRA / QLoRA) cho bài toán **Java function completion** (prefix → method body), đánh giá theo kiểu HumanEval-Java / MultiPL-E.

Repo cũng có prototype web **AJAC** (Automatic Java Code Generator) để demo sinh code Java từ mô tả.

---

## Yêu cầu

| Mục | Gợi ý |
|-----|--------|
| OS | Windows 10+ (PowerShell / CMD) |
| Python | 3.10+ |
| GPU | NVIDIA CUDA (RTX 3060 12GB trở lên cho QLoRA; H100 cho bf16 LoRA Plan B) |
| Disk | ~30–50 GB (base model + cache + checkpoint) |
| JDK | Có `javac` trên PATH (để evaluate) |

---

## Cài đặt

```powershell
python -m venv env
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\env\Scripts\Activate.ps1

pip install -U -r requirements.txt

# Torch khớp CUDA của máy (ví dụ CUDA 11.8 / 12.1):
pip install torch --index-url https://download.pytorch.org/whl/cu118
# hoặc: pip install torch --index-url https://download.pytorch.org/whl/cu121

# Prototype web (không có trong requirements.txt):
pip install flask
```

Tạo file `.env` ở root (không commit):

```env
HF_HOME=D:/cache/huggingface
HF_TOKEN=hf_xxx
WANDB_API_KEY=xxx
```

Một số dataset (ví dụ [bigcode/the-stack-smol](https://huggingface.co/datasets/bigcode/the-stack-smol)) cần accept license trên Hugging Face trước khi `HF_TOKEN` dùng được.

---

## Cấu trúc thư mục

```
.
├── data/                 # Build dataset (completion / evol)
├── train/                # Train + Bayesian Optimization
├── merge/                # Merge LoRA vào base
├── inference/            # Sinh completion trên benchmark
├── evaluate/             # Compile + chạy test Java
├── prototype/            # Web UI AJAC (Flask)
├── finetune_lora7b_planB/# Recipe Plan B (bf16 LoRA, H100)
├── models/               # Checkpoint / adapter đã train
├── run_pipeline.py       # Chạy full pipeline tự động
├── run_prototype*.bat    # Khởi động demo web
└── requirements.txt
```

---

## Pipeline chính (root)

Luồng: **build data → (optional BO) → train → merge → inference → evaluate**.

### Cách nhanh — `run_pipeline.py`

```powershell
# Full pipeline (root layout)
python run_pipeline.py --dir .

# Smoke test train (~0.01 epoch)
python run_pipeline.py --dir . --smoke-test

# Bỏ qua bước đã xong
python run_pipeline.py --dir . --skip-build --skip-bo

# Plan B (thư mục riêng)
python run_pipeline.py --dir finetune_lora7b_planB
```

Log ghi vào `pipeline_run.log` trong thư mục đang chạy.

### Chạy từng bước (CodeLlama + Evol / BO)

Chi tiết hơn: [`train/README_BO.md`](train/README_BO.md).

```powershell
# 1) Dataset Evol → method-body completion
python data/build_evol_completion_dataset.py
# → data/jsonl/evol_java_completion_train.jsonl

# 2) Train với best params BO
python train/main.py --config train/studies/best_params.json
# → models/java-codellama-lora/...

# 3) Merge / inference / evaluate
python merge/merge_model.py
python inference/inference.py
python evaluate/evaluate.py <file_inference.jsonl>
```

### CodeQwen (stage / Plan B)

Dataset completion đa nguồn:

```powershell
python data/build_completion_dataset.py
# hoặc trong plan: python finetune_lora7b_planB/build_completion_dataset.py
```

Train / merge / infer (xem cấu hình `MODEL_ID`, `OUTPUT_DIR`, `DATA_PATH` trong script):

```powershell
python train/main_qwen.py
python merge/merge_qwen.py
python inference/inference_qwen.py
python evaluate/evaluate.py <file_inference.jsonl>
```

Plan B (bf16 LoRA, FlashAttention, mục tiêu pass@1 cao hơn trên H100): xem [`finetune_lora7b_planB/README.md`](finetune_lora7b_planB/README.md).

---

## Bayesian Optimization

```powershell
python train/bayes_opt.py
python train/bayes_opt.py --n-trials 1
python train/bayes_opt.py --export-best
```

Proxy mặc định: ~8% data, 1 epoch, minimize `eval_loss`. Kết quả / best params nằm dưới `train/studies/`.

---

## Prototype web (AJAC)

Demo sinh Java từ prompt; mặc định load **CodeLlama-7B + LoRA** (8-bit trên GPU).

```powershell
# Mock — không load model, vào UI ngay
.\run_prototype_mock.bat

# Thật — cache HF mặc định trong bat (chỉnh path nếu cần)
.\run_prototype.bat

# Dùng cache đã có sẵn (ví dụ D:/cache/huggingface)
.\run_prototype_cached.bat
```

Mở [http://127.0.0.1:5000](http://127.0.0.1:5000).

Hoặc:

```powershell
python prototype/app.py --mock
python prototype/app.py
```

Model path cấu hình trong `prototype/app.py` (`BASE_MODEL`, `LORA_PATH`).

---

## Đánh giá

`evaluate/evaluate.py` compile và chạy unit test Java trên file JSONL sinh từ inference. Cần JDK (`javac` / `java`). JAR `javatuples` được tải tự động nếu thiếu.

Output dạng `*.report.txt`: pass@1 và breakdown `PASSED` / `COMPILE_ERROR` / `FAILED` / `TIMEOUT`.

---

## Tests

```powershell
pytest tests/
```

---

## Ghi chú

- Checkpoint lớn và `wandb/` nằm trong `.gitignore` — không push trọng số / log lên remote.
- Cache Hugging Face nên trỏ ra ổ đủ chỗ (`HF_HOME` trong `.env` hoặc trong các file `.bat`).
- Resume train: nhiều script dùng `get_last_checkpoint(OUTPUT_DIR)` — chạy lại cùng lệnh là tiếp tục từ checkpoint mới nhất.
- Không commit token (`HF_TOKEN`, `WANDB_API_KEY`); giữ trong `.env` local.
