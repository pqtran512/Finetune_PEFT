import torch
import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# 1. Cấu hình
MODEL_PATH = "./final_merged_qwen_model_v4"
OUTPUT_FILE = "java_qwen_inference_v4.jsonl"
CACHE_DIR = str(Path(__file__).parent / "cache" / "hugging_face")

STOP_STRINGS = [
    "\n    }\n",          # canonical MultiPL-E Java stop (closes method)
    "\n}\n",              # fallback if model emits less indentation
    "\npublic static void main",
    "\n```",
]


def clean_output(gen_text: str) -> str:
    # Tách block markdown nếu có
    if "```java" in gen_text:
        gen_text = gen_text.split("```java", 1)[1].split("```", 1)[0]
    elif "```" in gen_text:
        gen_text = gen_text.split("```", 1)[1].split("```", 1)[0]

    # Cắt sớm nếu model bắt đầu sinh class/method top-level mới
    cut_markers = [
        "\npublic class ",
        "\nclass ",
        "\n// Test",
        "\npublic static void main",
        "\nimport ",
        "\npackage ",
    ]
    for m in cut_markers:
        i = gen_text.find(m)
        if i != -1:
            gen_text = gen_text[:i]
    return gen_text.rstrip()


def main():
    print("🚀 Loading Merged Model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    print("📥 Loading Dataset...")
    dataset = load_dataset(
        "nuprl/MultiPL-E", "humaneval-java", split="test", cache_dir=CACHE_DIR
    )

    eos_ids = [tokenizer.eos_token_id]
    extra_eos = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if isinstance(extra_eos, int) and extra_eos != tokenizer.unk_token_id and extra_eos not in eos_ids:
        eos_ids.append(extra_eos)

    results = []
    for item in tqdm(dataset, desc="Generating Java Code"):
        prompt = item["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,    # greedy for stable single-sample pass@1
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=eos_ids,
                stop_strings=STOP_STRINGS,
                tokenizer=tokenizer,
            )

        gen_ids = outputs[0][input_length:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        results.append(
            {
                "task_id": item["name"],
                "prompt": prompt,
                "completion": clean_output(gen_text),
                "tests": item["tests"],
            }
        )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")

    print(f"✅ Đã xong! Kết quả lưu tại {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
