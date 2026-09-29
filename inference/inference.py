import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.directives import apply_directive, STOP_STRINGS


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


def clean_output(gen_text: str) -> str:
    """Lọc lấy code Java, dừng lại nếu thấy định nghĩa class mới hoặc EOF"""
    if "```java" in gen_text:
        gen_text = gen_text.split("```java")[-1].split("```")[0]
    elif "```" in gen_text:
        gen_text = gen_text.split("```")[-1].split("```")[0]
    return gen_text.strip()


def main():
    parser = argparse.ArgumentParser(description="Inference CodeLlama-7B trên HumanEval-Java benchmark.")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(_REPO_ROOT / "final_models" / "evol_completion_bo_v2"),
        help="Đường dẫn mô hình CodeLlama đã merge",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Đường dẫn file jsonl đầu ra (mặc định tự chọn theo directive)",
    )
    parser.add_argument(
        "--use-directive",
        action="store_true",
        help="Chèn Prompt Directive vào đầu prompt trước khi sinh mã",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Số lượng token tối đa sinh ra (mặc định: 512)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Giới hạn số lượng tasks cần sinh (dùng để smoke test nhanh)",
    )
    args = parser.parse_args()

    _load_env(Path(__file__).parent / ".env")
    _load_env(_REPO_ROOT / ".env")

    model_path = Path(args.model_path)
    if args.output:
        output_file = Path(args.output)
    elif args.use_directive:
        output_file = _REPO_ROOT / "inference" / "jsonl" / "inference_codellama_directive.jsonl"
    else:
        output_file = _REPO_ROOT / "inference" / "jsonl" / "inference_evol_completion_bo_v2.jsonl"

    cache_dir = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))

    print("=" * 60)
    print("HF_HOME / cache:", cache_dir)
    print("Model path     :", model_path)
    print("Output file    :", output_file)
    print("Prompt Directive:", "BẬT" if args.use_directive else "TẮT")
    print("=" * 60)

    # 1. Load Model & Tokenizer
    print("Loading Merged Model...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    # 2. Load Dataset MultiPL-E (Java version)
    print("Loading Dataset...")
    dataset = load_dataset(
        "nuprl/MultiPL-E", "humaneval-java", split="test", cache_dir=cache_dir
    )

    if args.limit:
        dataset = list(dataset)[: args.limit]
        print(f"Giới hạn chạy thử nghiệm: {len(dataset)} tasks.")

    # 3. Chạy Inference
    results = []
    desc = f"Generating Java Code ({'Directive' if args.use_directive else 'Standard'})"
    for item in tqdm(dataset, desc=desc):
        prompt = item["prompt"]
        input_prompt = apply_directive(prompt) if args.use_directive else prompt
        inputs = tokenizer(input_prompt, return_tensors="pt").to(model.device)
        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    stop_strings=STOP_STRINGS,
                    tokenizer=tokenizer,
                )
            except TypeError:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

        gen_ids = outputs[0][input_len:]
        generated_code = tokenizer.decode(gen_ids, skip_special_tokens=True)

        results.append(
            {
                "task_id": item["name"],
                "prompt": prompt,
                "completion": clean_output(generated_code),
                "tests": item["tests"],
            }
        )

    # 4. Lưu ra file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")

    print(f"\n✅ Đã hoàn tất! Kết quả đã lưu vào: {output_file}")


if __name__ == "__main__":
    main()

