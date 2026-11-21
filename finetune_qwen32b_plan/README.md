# Qwen2.5-Coder-32B plan — QLoRA 4-bit

Mục tiêu: vượt pass@1 **73.69%** trên HumanEval-Java.

## Phân tích tài nguyên — vì sao chỉ QLoRA khả thi

Qwen2.5-Coder-32B là model 32B parameter. Trên H100 80GB đơn:

| Option | VRAM cần | Có fit không? |
|--------|----------|---------------|
| Full bf16 FT | 64 + 64 + 128 (Adam fp32) = ~256GB | ❌ Không thể |
| Full bf16 FT + 8-bit Adam | 64 + 64 + 32 = ~160GB | ❌ Không thể |
| **LoRA trên bf16 base** | 64 + 5 (LoRA) + 5 (activations) = **~74GB** | ⚠️ Tight, dễ OOM khi spike |
| **QLoRA 4-bit** | **16 + 5 + 10 = ~31GB** | ✅ An toàn |
| LoRA bf16 multi-GPU (FSDP) | distribute 64GB across N GPUs | ✅ Nhưng cần ≥2 GPU |

**→ QLoRA 4-bit là phương án DUY NHẤT khả thi 1 GPU 80GB.** Plan này dùng QLoRA + FA2 + LoRA r=64 α=128, NEFTune α=5, 2 epochs, LR 5e-5.

## Tài nguyên dự kiến (H100 80GB)

| Mục | Giá trị |
|-----|---------|
| VRAM peak (training) | 30-40 GB |
| VRAM (merge) | ~70 GB (load bf16 base + LoRA) |
| VRAM (inference bf16) | ~64 GB |
| VRAM (inference 4-bit) | ~16 GB |
| RAM | 60-100 GB |
| CPU | 12-16 cores |
| Disk | **~150 GB** (base cache 64GB + merged 64GB + LoRA ckpt + data) |
| Wall-clock 2 epochs SFT | **22-40h** (chỉ vừa đủ trong 3-day budget cho 1 run) |
| Wall-clock merge | 10-20 phút |
| Wall-clock inference (158 problems) | 60-90 phút bf16 |

⚠️ **Quan trọng:** chỉ chạy được 1-2 training run trong 3 ngày. Không có buffer để fail nhiều lần. Phải chạy smoke test ngắn (100 steps) trước khi commit full run.

## Pass@1 — kỳ vọng so với target 73.69%

Qwen2.5-Coder-32B base trên HumanEval-Java (greedy, MultiPL-E) đã rất cao (~70-75%). Fine-tune trên narrow corpus có **rủi ro làm GIẢM** pass@1 vì catastrophic forgetting — model lớn + data nhỏ là công thức memorize quirks dataset thay vì học general pattern.

| Scenario | Kỳ vọng pass@1 |
|----------|----------------|
| Base Qwen2.5-Coder-32B (no fine-tune) | ~70-75% |
| QLoRA SFT với data hiện tại (boilerplate-heavy) | **65-72%** (có thể giảm vs base) |
| QLoRA SFT với data Plan B-style (algorithmic, quality-filtered) | **72-78%** |
| QLoRA SFT + few-shot hybrid (E.1 trong PLAN.md) | **74-80%** |
| Few-shot trên base, không fine-tune | **73-77%** |

→ **Cách an toàn nhất để vượt 73.69%:** chạy inference base TRƯỚC, đo baseline. Nếu base đã ≥73.69%, đừng fine-tune mù — risk giảm điểm cao hơn. Xem section "Workflow đề xuất" bên dưới.

## Setup

```powershell
python -m venv env
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
env\Scripts\Activate.ps1

pip install -U -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn --no-build-isolation
copy .env.example .env
# Điền HF_TOKEN trong .env
```

Accept license trên HuggingFace Hub:
- [bigcode/the-stack-smol](https://huggingface.co/datasets/bigcode/the-stack-smol)
- [bigcode/the-stack-dedup](https://huggingface.co/datasets/bigcode/the-stack-dedup)
- [bigcode/self-oss-instruct-sc2-exec-filter-50k](https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k)
- [Qwen/Qwen2.5-Coder-32B](https://huggingface.co/Qwen/Qwen2.5-Coder-32B) (non-gated nhưng nên check)

## Workflow đề xuất (đo baseline trước, fine-tune chỉ khi cần)

### Bước 0 — Đo baseline pass@1 của base 32B (KHÔNG fine-tune)

Quan trọng nhất: biết model base đạt bao nhiêu để quyết định có fine-tune không.

Tạo script `inference_base.py` tạm (copy `inference_qwen.py` rồi đổi `MODEL_PATH = "Qwen/Qwen2.5-Coder-32B"` và `OUTPUT_FILE = "java_qwen32b_base.jsonl"`), chạy:

```powershell
python inference_base.py  # ~60-90 phút
python evaluate.py java_qwen32b_base.jsonl
```

**Quyết định:**
- Nếu baseline > 73.69% → **dừng**, đã đạt target. Khoá kết quả và làm few-shot hybrid để stretch cao hơn.
- Nếu baseline 70-73% → fine-tune CÓ THỂ giúp nhưng rủi ro. Chạy build + train + eval theo các bước dưới, eval mỗi save_step để bắt early-stop.
- Nếu baseline < 70% → có cơ hội rõ ràng để fine-tune cải thiện.

### Bước 1-4 — Pipeline đầy đủ (chỉ làm nếu Bước 0 quyết định fine-tune)

```powershell
# 1. Build training dataset (~1-2 giờ)
python build_completion_dataset.py

# 2. Train (QLoRA 4-bit, FA2, 2 epochs, ~22-40h)
python main_qwen.py

# 3. Merge LoRA vào bf16 base (10-20 phút, cần ~70GB VRAM)
python merge_qwen.py

# 4. Generate + evaluate
python inference_qwen.py
python evaluate.py java_qwen32b_inference.jsonl
```

## Khác biệt vs Plan B (CodeQwen1.5-7B)

| | Plan B (7B bf16 LoRA) | Qwen32B plan (32B QLoRA 4bit) |
|--|----------------------|-------------------------------|
| Base model | Qwen/CodeQwen1.5-7B | Qwen/Qwen2.5-Coder-32B |
| Quantization | bf16 (full precision base) | 4-bit nf4 (QLoRA — VRAM constraint) |
| LR | 8e-5 | **5e-5** (32B cần update chậm hơn) |
| Epochs | 3 | **2** (32B đã capable, ít epoch để tránh memorize) |
| grad_accum | 32 | **16** (step chậm hơn, cần feedback nhanh) |
| save_total_limit | 3 | **2** (32B LoRA ckpt ~2GB mỗi cái) |
| VRAM training | 32-42GB | 30-40GB (4-bit base nhẹ hơn) |
| Wall-clock 3 epochs | 10-14h | **22-40h cho 2 epochs** |
| Inference VRAM | bf16 ~14GB | **bf16 ~64GB hoặc 4-bit ~16GB** |
| Pass@1 baseline ước tính | 40-45% | **70-75%** |

## Failure mode đặc biệt cho 32B

1. **Catastrophic forgetting nghiêm trọng hơn 7B** vì base đã rất capable. Triệu chứng: pass@1 sau train giảm vs base. Cứu: giảm epoch (2 → 1), giảm LR (5e-5 → 2e-5), tăng `eval_steps` và load best checkpoint sớm.

2. **Memorization của dataset quirks** — 32B sẽ học cả các pattern lỗi trong data (indent loạn, external library refs). Cứu: tăng strictness của quality filter trong `build_completion_dataset.py`, kiểm tra failure-mode breakdown sau eval.

3. **Wall-clock blow-up** — nếu data quá lớn (>60k samples), 2 epochs có thể vượt 40h. Cứu: hạ cap `code_search_net` từ 50k xuống 25k (vẫn cao hơn baseline cũ).

4. **Disk full khi save merged** — base 64GB + merged 64GB = 128GB. Plus checkpoints. Cần ≥150GB free.

## Optional stretch: few-shot hybrid (Section E.1 của PLAN.md)

Nếu cần ép thêm pass@1: implement few-shot hybrid. Yêu cầu `MAX_LENGTH` 1536 → 2048 trong main_qwen.py (VRAM thêm ~3-5GB, vẫn fit). Lift kỳ vọng +2-4 pts. Chi tiết trong `../PLAN.md` section E.

## Multi-GPU stretch (không bắt buộc)

Nếu sau này có thêm GPU (≥2× H100 80GB hoặc 1× H200 141GB), có thể switch sang **LoRA bf16 full precision** thay vì QLoRA — quality cao hơn ~1-2 pts. Dùng `accelerate launch --multi_gpu main_qwen.py` với FSDP. Plan hiện tại không cover vì user có 1 GPU.
