"""Inference HumanEval-Java với CodeQwen base (không finetune) để đo baseline thật."""
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

BASE_MODEL = "Qwen/CodeQwen1.5-7B"
OUTPUT_FILE = "java_qwen_inference_base.jsonl"
CACHE_DIR = "D:/cache/hugging_face"

STOP_STRINGS = [
    "\npublic class ",
    "\nclass ",
    "\n// Test",
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
    print(f"🚀 Loading BASE model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=CACHE_DIR,
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
    for item in tqdm(dataset, desc="Base CodeQwen Java"):
        prompt = item["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.2,
                top_p=0.95,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.eos_token_id,
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

    print(f"✅ Done. Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
