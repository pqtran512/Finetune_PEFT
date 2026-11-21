# Qwen2.5-Coder-32B plan — v2 (prompt + repair)

Mục tiêu: nâng pass@1 **68.35% → 73.69%+** trên HumanEval-Java **mà không retrain**.

## Bối cảnh

v1 (`../result/java_qwen32b_inference.jsonl.report.txt`) đạt **68.35% (108/158)**.
Base Qwen2.5-Coder-32B (chưa fine-tune) đo được **~66%** → fine-tune có cải thiện
~2.35pp, nhưng còn cách target 5.34pp.

Phân tích 50 fail của v1:
| Loại | Số | Pattern |
|------|----|---------|
| COMPILE_ERROR | 22 | Model gọi helper (`isPrime`, `factorial`, `isPalindrome`, `sumOfDigits`, `isVowel`, ...) nhưng **không** định nghĩa trong cùng file. 2 task lỗi `Pair.getFirst()/getSecond()` (API thật là `getValue0()/getValue1()`). |
| FAILED        | 28 | Lỗi logic / assertion / runtime — không thuộc scope v2. |

v2 nhắm chính xác vào 22 COMPILE_ERROR bằng 2 biện pháp song hành.

## Pipeline

```
inference_qwen.py
    └─> java_qwen32b_inference_v2.jsonl
            │
            ▼
evaluate.py  (pass 1)
    ├─> java_qwen32b_inference_v2.jsonl.report.txt
    └─> java_qwen32b_inference_v2.jsonl.failures.jsonl
            │
            ▼
repair_qwen.py   (chỉ chạm COMPILE_ERROR)
    └─> java_qwen32b_inference_v2_repaired.jsonl
            │
            ▼
evaluate.py  (pass 2 - kết quả cuối cùng)
    └─> java_qwen32b_inference_v2_repaired.jsonl.report.txt
```

## Hai biện pháp

### 1. Prompt directive chống helper (inference_qwen.py)

Trước mỗi prompt HumanEval-Java, prepend block comment hướng dẫn:
- Implementation phải self-contained, không gọi method chưa định nghĩa.
- Nếu cần helper → định nghĩa luôn trong cùng class, sau closing brace của method chính.
- Chỉ dùng `java.util.*`, `java.util.stream.*`, `java.util.regex.*`, `java.lang.Math`, và `org.javatuples.Pair`.
- Đặc biệt: `Pair` của javatuples dùng `getValue0()/getValue1()`, **không phải** `getFirst()/getSecond()` — fix luôn 2 task lỗi Pair của v1.

Đây là biện pháp rẻ nhất (chỉ tốn 1 lần gen lại 158 task).

### 2. Repair pass cho COMPILE_ERROR (repair_qwen.py)

Khi prompt directive không đủ (model vẫn gọi helper không define), evaluate.py
sẽ parse compiler stderr để extract `(method_name, arg_types)` của các symbol
bị thiếu, ghi vào `.failures.jsonl`. repair_qwen.py:

1. Đọc chỉ entry có `status == "COMPILE_ERROR"` và `missing_methods != []`.
2. Build repair prompt: liệt kê các helper bị thiếu + compile stderr + toàn bộ
   `Problem.java` hiện tại, kết thúc bằng `    private static ` để hint model
   bắt đầu emit signature.
3. Generate helper definitions, dán vào cuối `completion` cũ.
4. Merge với 158 task gốc → `..._repaired.jsonl`.

Chỉ COMPILE_ERROR được sửa. FAILED / TIMEOUT là lỗi logic — out-of-scope của v2.

## Cách chạy

```bash
cd finetune_qwen32b_plan/v2

# Bước 1: generate với directive (~60-80 phút H100 bf16)
python inference_qwen.py

# Bước 2: eval lần 1, xem prompt directive đã đỡ được bao nhiêu
python evaluate.py

# Bước 3: repair COMPILE_ERROR còn lại (~5-15 phút tuỳ số task)
python repair_qwen.py

# Bước 4: eval lần cuối
python evaluate.py java_qwen32b_inference_v2_repaired.jsonl
```

## Tham số

Đọc từ env (cùng cơ chế với v1):

| Biến | Mặc định | Ghi chú |
|------|----------|---------|
| `MODEL_PATH` | `../final_merged_qwen32b_model` | Đường dẫn merged model |
| `INFERENCE_DTYPE` | `bf16` | hoặc `4bit` để chạy ~16GB VRAM |
| `HF_HOME` | `D:/cache/huggingface` | Hugging Face model/dataset cache |

## Kỳ vọng kết quả

Phép thử tốt nhất:
- Prompt directive một mình có thể recover **8-15 / 22** COMPILE_ERROR → 113-115 PASSED → ~71-73%.
- Repair pass cover phần còn lại → 117-122 PASSED → ~74-77%.

Phép thử dở nhất:
- Directive ít hiệu lực với model đã SFT mạnh theo style cũ → recover 3-5 → ~70%.
- Repair pass cover thêm 8-12 → ~75%.

Cả hai trường hợp đều có khả năng vượt 73.69%. Nếu không vượt: tín hiệu rằng
phần lỗi cốt lõi nằm ở 28 FAILED (logic), cần tác động vào training data, không
phải prompt.

## Không thuộc v2

- Không retrain.
- Không sửa 28 FAILED (lỗi logic) — nếu sau v2 vẫn không đạt target, mới cân
  nhắc retrain hoặc sửa data.
- Không ghi đè bất kỳ file nào ở `..` — toàn bộ artifact (jsonl, report, failures)
  nằm trong thư mục `v2/`.
