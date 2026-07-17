# Plan B — LoRA bf16 + data rebuild đầy đủ (Khuyến nghị)

Mục tiêu: pass@1 HumanEval-Java **52-57%** trong **2-3 ngày training** trên H100 80GB. Đây là phương án cao nhất / rủi ro thấp nhất trong [PLAN.md](../PLAN.md).

Plan B làm 3 thay đổi lớn so với baseline:
1. **Data rebuild** với loosened `METHOD_RE` (unlock 5-10× yield Magicoder/Evol), quality filter (drop `@Nullable`, external libs, indent broken), thêm Magicoder-Evol-110K.
2. **bf16 LoRA** thay QLoRA 4-bit — quality tăng, vẫn còn rất nhiều headroom VRAM trên H100 80GB.
3. **Training mods**: FlashAttention 2, NEFTune α=5, LoRA r=64 α=128, 3 epochs, LR 8e-5.

## Tài nguyên (H100 80GB)

| Mục | Giá trị |
|-----|---------|
| VRAM peak | 32-42 GB (40-52% của 80GB — cân bằng) |
| RAM | 50-80 GB |
| CPU | 8-12 cores |
| Disk | ~50 GB (base bf16 ~14GB + 3 LoRA ckpt + packed data) |
| Wall-clock 3 epochs | ~10-14h |

## Setup

```powershell
python -m venv env
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
env\Scripts\Activate.ps1

pip install -U -r requirements.txt
# Torch wheel khớp CUDA, ví dụ CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121
# Flash-Attention 2 (cần GPU Ampere/Hopper, H100 OK):
pip install flash-attn --no-build-isolation

copy .env.example .env
# Mở .env và điền HF_TOKEN + WANDB_API_KEY
```

**Quan trọng:** trước khi chạy build_completion_dataset.py, vào HuggingFace Hub và accept license cho:
- [bigcode/the-stack-smol](https://huggingface.co/datasets/bigcode/the-stack-smol)

Account của bạn phải accept license trước, sau đó HF_TOKEN mới truy cập được.

## Chạy 4 bước

```powershell
# 1. Build training dataset (~1-2 giờ, có internet tốt)
python build_completion_dataset.py
# Kỳ vọng: ~55-60k samples (code_search_net chiếm phần lớn).
# Audit output: bao nhiêu mẫu có helper appended, sample prefix khớp MultiPL-E.

# 2. Train (bf16 LoRA, FA2, 3 epochs, ~10-14h trên H100)
python main_qwen.py

# 3. Merge LoRA vào base (10-15 phút, ~16GB VRAM)
python merge_qwen.py

# 4. Generate + evaluate
python inference_qwen.py
python evaluate.py java_qwen_inference_planB.jsonl
```

Output cuối: `java_qwen_inference_planB.jsonl.report.txt` — pass@1 % + breakdown PASSED / COMPILE_ERROR / FAILED / TIMEOUT.

## Khác biệt vs Plan A

| | Plan A | Plan B |
|--|--------|--------|
| Quantization | QLoRA 4-bit nf4 | **bf16 LoRA** |
| LR | 1e-4 | **8e-5** |
| METHOD_RE | strict (`public static`) | **loosened (≥1 modifier)** |
| Quality filter | không | **annotations + external classes + indent jumps** |
| Nguồn dataset | 4 (cũ) | **5** (+ Magicoder-Evol) |
| `code_search_net` cap | 25k | **50k** (sau quality filter) |
| Dataset yield kỳ vọng | ~26-30k | **~55-60k** |
| Algorithmic share | ~8% | **~25-40%** |
| VRAM | 14-18GB | 32-42GB |
| Wall-clock | 12-18h | 10-14h (FA2 + bf16 nhanh hơn) |
| Pass@1 kỳ vọng | 48-53% | **52-57%** |

## Resume

Training bị gián đoạn → chạy lại `python main_qwen.py` — `get_last_checkpoint(OUTPUT_DIR)` tự tìm ckpt mới nhất trong `./models/codeqwen/planB/`.

## Failure analysis

Sau khi có report.txt, đọc phần "Failures" để phân loại:
- **COMPILE_ERROR đa số** → cần `constrained decoding` (xem section D của [PLAN.md](../PLAN.md)) hoặc lọc helper xuất ra ngoài whitelist.
- **FAILED (test logic sai) đa số** → cần thêm data algorithmic chất lượng cao hoặc DPO (section C của PLAN.md).
- **TIMEOUT đa số** → model emit infinite loop → check pattern, có thể cần tăng penalty cho lặp.

## Optional: Few-shot hybrid (Section E.1 của PLAN.md)

Nếu Plan B baseline ra 52-55% và bạn muốn ép thêm +2-4 pts: implement few-shot hybrid (train 25-30% mẫu có prepended 2-shot, inference dùng fixed 2-3 shot). Đòi hỏi `MAX_LENGTH` 1536 → 2048 trong main_qwen.py và sửa build script + inference. Chi tiết trong PLAN.md section E.1.
