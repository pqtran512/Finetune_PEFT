"""Qwen2.5-Coder-32B plan v2 — Generate Java completions on HumanEval-Java.

Khác biệt so với v1:
- Prepend một block hướng dẫn vào mỗi prompt để CẤM model gọi helper method
  mà nó không tự định nghĩa trong cùng file. Đây là biện pháp rẻ nhất để
  loại bỏ failure-mode chiếm 22/50 fail ở v1 (model gọi `isPrime`, `factorial`,
  `isPalindrome`, ... mà không emit definition).
- Output file: java_qwen32b_inference_v2.jsonl (không ghi đè v1).

Pipeline v2:
  1. inference_qwen.py        -> java_qwen32b_inference_v2.jsonl
  2. evaluate.py              -> .report.txt + .failures.jsonl
  3. repair_qwen.py           -> java_qwen32b_inference_v2_repaired.jsonl
  4. evaluate.py (lần 2)      -> .report.txt cuối cùng

Cách chạy: python inference_qwen.py
"""
import os
import sys
import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset


def _load_env(path: Path):
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


_load_env(Path(__file__).parent / ".env")
_load_env(Path(__file__).parent.parent / ".env")

MODEL_PATH = os.environ.get("MODEL_PATH", "../final/final_merged_qwen32b_model")
OUTPUT_FILE = "java_qwen32b_inference_v2.jsonl"
CACHE_DIR = str(Path(__file__).parent.parent / "cache" / "hugging_face")
INFERENCE_DTYPE = os.environ.get("INFERENCE_DTYPE", "bf16").lower()

# Ensure repo root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.directives import PROMPT_DIRECTIVE, STOP_STRINGS, apply_directive
from shared.cleaner import clean_output


def main():
    print(f"Loading merged model from {MODEL_PATH} (dtype={INFERENCE_DTYPE})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    if INFERENCE_DTYPE == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()

    print("Loading HumanEval-Java (MultiPL-E)...")
    dataset = load_dataset("nuprl/MultiPL-E", "humaneval-java", split="test", cache_dir=CACHE_DIR)

    eos_ids = [tokenizer.eos_token_id]
    extra_eos = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if isinstance(extra_eos, int) and extra_eos != tokenizer.unk_token_id and extra_eos not in eos_ids:
        eos_ids.append(extra_eos)
    extra_im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(extra_im_end, int) and extra_im_end != tokenizer.unk_token_id and extra_im_end not in eos_ids:
        eos_ids.append(extra_im_end)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    results = []
    for item in tqdm(dataset, desc="Generating Java v2"):
        original_prompt = item["prompt"]
        directed_prompt = apply_directive(original_prompt)
        inputs = tokenizer(directed_prompt, return_tensors="pt").to(model.device)
        input_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=768,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=eos_ids,
                stop_strings=STOP_STRINGS,
                tokenizer=tokenizer,
            )

        gen_ids = outputs[0][input_length:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        results.append({
            "task_id": item["name"],
            "prompt": original_prompt,
            "completion": clean_output(gen_text),
            "tests": item["tests"],
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")

    print(f"Done. Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
