import torch
import json
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from datasets import load_dataset

# 1. Cấu hình
MODEL_PATH = "./final_merged_qwen_model"
OUTPUT_FILE = "java_qwen_inference.jsonl"
CACHE_DIR = "D:/cache/hugging_face"

STOP_STRINGS = [
    "\npublic class ",
    "\nclass ",
    "\n// Test",
    "\npublic static void main",
    "\n```",
]


class StopOnSubstring(StoppingCriteria):
    def __init__(self, stops, tokenizer, prompt_len):
        self.stops = stops
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs):
        gen = self.tokenizer.decode(
            input_ids[0][self.prompt_len:], skip_special_tokens=True
        )
        return any(s in gen for s in self.stops)


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

        stopping = StoppingCriteriaList(
            [StopOnSubstring(STOP_STRINGS, tokenizer, input_length)]
        )

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
                stopping_criteria=stopping,
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
