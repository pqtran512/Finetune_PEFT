"""
Variant của inference_qwen.py:
- Inject "// Algorithm plan:\n// 1." vào cuối prompt để buộc model
  liệt kê các bước thuật toán trước khi code (chain-of-thought style
  trong completion format).
- Chỉ chạy 10 task được liệt kê trong TARGET_TASKS (các bài FAILED đại diện
  Pattern B - hiểu sai requirement).
- Tăng max_new_tokens lên 768 để có chỗ cho cả plan + code.
"""

import torch
import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

_SCRIPT_DIR = Path(__file__).parent.resolve()
# Model nằm ở cùng thư mục với script inference
# Resolve absolute để tránh HFValidationError với "./" prefix trên transformers mới.
MODEL_PATH = str((_SCRIPT_DIR / "final_merged_qwen_model_v4").resolve())
OUTPUT_FILE = "java_qwen_inference_reason.jsonl"
CACHE_DIR = str(Path(__file__).parent / "cache" / "hugging_face")

TARGET_TASKS = {
    "HumanEval_9_rolling_max",
    "HumanEval_26_remove_duplicates",
    "HumanEval_37_sort_even",
    "HumanEval_41_car_race_collision",
    "HumanEval_69_search",
    "HumanEval_72_will_it_fly",
    "HumanEval_75_is_multiply_prime",
    "HumanEval_89_encrypt",
    "HumanEval_106_f",
    "HumanEval_115_max_fill",
}

REASONING_HINT = "\n        // Algorithm plan (read docstring carefully):\n        // 1."

STOP_STRINGS = [
    "\n    }\n",
    "\n}\n",
    "\npublic static void main",
    "\n```",
]


def clean_output(gen_text: str) -> str:
    if "```java" in gen_text:
        gen_text = gen_text.split("```java", 1)[1].split("```", 1)[0]
    elif "```" in gen_text:
        gen_text = gen_text.split("```", 1)[1].split("```", 1)[0]
    cut_markers = [
        "\npublic class ", "\nclass ", "\n// Test",
        "\npublic static void main", "\nimport ", "\npackage ",
    ]
    for m in cut_markers:
        i = gen_text.find(m)
        if i != -1:
            gen_text = gen_text[:i]
    return gen_text.rstrip()


def main():
    print("Loading model...")
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Local model path not found: {MODEL_PATH}. "
            "Please verify the directory and update MODEL_PATH if needed."
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    print("Loading dataset...")
    dataset = load_dataset(
        "nuprl/MultiPL-E", "humaneval-java", split="test", cache_dir=CACHE_DIR
    )

    eos_ids = [tokenizer.eos_token_id]
    extra_eos = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if isinstance(extra_eos, int) and extra_eos != tokenizer.unk_token_id and extra_eos not in eos_ids:
        eos_ids.append(extra_eos)

    results = []
    for item in tqdm(dataset, desc="Generating (reasoning)"):
        if item["name"] not in TARGET_TASKS:
            continue

        original_prompt = item["prompt"]
        # Inject reasoning hint: prompt giữ nguyên, append hint sau dấu { mở method.
        # Vì prompt kết thúc bằng "(...)" + "\n" (sau "{"), append vào cuối.
        augmented_prompt = original_prompt + REASONING_HINT

        inputs = tokenizer(augmented_prompt, return_tensors="pt").to(model.device)
        input_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=768,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=eos_ids,
                stop_strings=STOP_STRINGS,
                tokenizer=tokenizer,
            )

        gen_ids = outputs[0][input_length:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # CRITICAL: completion sẽ được ghép vào prompt GỐC (không augmented)
        # khi evaluate. Vì REASONING_HINT là comment hợp lệ Java, ta prepend
        # nó vào completion để evaluate vẫn nhận được plan trong source.
        full_completion = REASONING_HINT + clean_output(gen_text)

        results.append({
            "task_id": item["name"],
            "prompt": original_prompt,
            "completion": full_completion,
            "tests": item["tests"],
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")

    print(f"Done. Wrote {len(results)} samples to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
