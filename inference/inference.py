import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_REPO_ROOT = Path(__file__).resolve().parent.parent
_load_env(Path(__file__).parent / ".env")
_load_env(_REPO_ROOT / ".env")

# 1. Cấu hình
MODEL_PATH = "./final_merged_model_ckpt352"
OUTPUT_FILE = "java_inference_ckpt352.jsonl"
CACHE_DIR = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
print("HF_HOME / cache:", CACHE_DIR)

# 2. Load Model & Tokenizer
print("Loading Merged Model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
)

# 3. Load Dataset MultiPL-E (Java version)
print("Loading Dataset...")
dataset = load_dataset(
    "nuprl/MultiPL-E", "humaneval-java", split="test", cache_dir=CACHE_DIR
)


def clean_output(gen_text):
    """Lọc lấy code Java, dừng lại nếu thấy định nghĩa class mới hoặc EOF"""
    if "```java" in gen_text:
        gen_text = gen_text.split("```java")[-1].split("```")[0]
    return gen_text.strip()


# 4. Chạy Inference
results = []
for item in tqdm(dataset, desc="Generating Java Code"):
    prompt = item["prompt"]
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_code = full_output[len(prompt) :]

    results.append(
        {
            "task_id": item["name"],
            "prompt": prompt,
            "completion": clean_output(generated_code),
            "tests": item["tests"],
        }
    )

# 5. Lưu ra file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for entry in results:
        f.write(json.dumps(entry) + "\n")

print(f"Done. Saved to {OUTPUT_FILE}")
