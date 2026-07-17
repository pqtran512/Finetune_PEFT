"""Plan B — Generate Java completions on HumanEval-Java (MultiPL-E)."""
import os
import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
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

MODEL_PATH = "./final_merged_qwen_model"
OUTPUT_FILE = "java_qwen_inference_planB.jsonl"
CACHE_DIR = os.environ.get(
    "HF_CACHE_DIR",
    str(Path.home() / ".cache" / "huggingface"),
)

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
    print(f"Loading merged model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    print("Loading HumanEval-Java (MultiPL-E)...")
    dataset = load_dataset("nuprl/MultiPL-E", "humaneval-java", split="test", cache_dir=CACHE_DIR)

    eos_ids = [tokenizer.eos_token_id]
    extra_eos = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if isinstance(extra_eos, int) and extra_eos != tokenizer.unk_token_id and extra_eos not in eos_ids:
        eos_ids.append(extra_eos)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    results = []
    for item in tqdm(dataset, desc="Generating Java"):
        prompt = item["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
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
            "prompt": prompt,
            "completion": clean_output(gen_text),
            "tests": item["tests"],
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")

    print(f"Done. Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
