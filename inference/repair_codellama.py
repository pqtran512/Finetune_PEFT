"""Repair pass cho mô hình CodeLlama-7B trên HumanEval-Java benchmark.

Đọc file .failures.jsonl từ evaluate.py, với mỗi entry status=COMPILE_ERROR có missing_methods,
gọi lại mô hình CodeLlama yêu cầu sinh các helper methods đó.
Output là file jsonl đã được vá lỗi để chạy evaluate.py lần 2.

Cách chạy:
    python inference/repair_codellama.py [input_inference.jsonl] [failures.jsonl] [output_repaired.jsonl]
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.directives import STOP_STRINGS
from shared.repair import repair_completions


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


def main():
    parser = argparse.ArgumentParser(description="Chạy Repair Pass cho mô hình CodeLlama-7B")
    parser.add_argument(
        "input_file",
        nargs="?",
        default=str(_REPO_ROOT / "inference" / "jsonl" / "inference_evol_completion_bo_v2.jsonl"),
        help="Đường dẫn file jsonl kết quả inference gốc",
    )
    parser.add_argument(
        "failures_file",
        nargs="?",
        default="",
        help="Đường dẫn file .failures.jsonl (mặc định: {input_file}.failures.jsonl)",
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        default="",
        help="Đường dẫn file jsonl sau repair (mặc định: {input_stem}_repaired.jsonl)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(_REPO_ROOT / "final_models" / "evol_completion_bo_v2"),
        help="Đường dẫn thư mục mô hình CodeLlama đã merge",
    )
    args = parser.parse_args()

    _load_env(Path(__file__).parent / ".env")
    _load_env(_REPO_ROOT / ".env")

    input_path = Path(args.input_file)
    if not input_path.exists():
        fallback = _REPO_ROOT / "inference" / "jsonl" / "inference_evol_completion_bo_v2.jsonl"
        if fallback.exists():
            input_path = fallback
        else:
            print(f"Error: Không tìm thấy input file: {input_path}")
            sys.exit(1)

    failures_path = Path(args.failures_file) if args.failures_file else Path(f"{input_path}.failures.jsonl")
    if not failures_path.exists():
        print(f"Error: Không tìm thấy failures file: {failures_path}")
        print("Gợi ý: Hãy chạy evaluate trước: python evaluate/evaluate.py " + str(input_path))
        sys.exit(1)

    if args.output_file:
        output_path = Path(args.output_file)
    else:
        suffix = input_path.suffix
        output_path = input_path.parent / f"{input_path.stem}_repaired{suffix}"

    model_path = Path(args.model_path)
    print("=" * 60)
    print("Input file   :", input_path)
    print("Failures file:", failures_path)
    print("Output file  :", output_path)
    print("Model path   :", model_path)
    print("=" * 60)

    print("Loading CodeLlama Model...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    def generate_fn(prompt_text: str) -> str:
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        input_len = inputs.input_ids.shape[1]
        with torch.no_grad():
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    stop_strings=STOP_STRINGS,
                    tokenizer=tokenizer,
                )
            except TypeError:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
        gen_ids = outputs[0][input_len:]
        return tokenizer.decode(gen_ids, skip_special_tokens=True)

    print("Starting Repair Pass...")
    repaired_count = repair_completions(
        input_path=str(input_path),
        failures_path=str(failures_path),
        output_path=str(output_path),
        generate_fn=generate_fn,
    )

    print(f"\n✅ Đã hoàn tất Repair Pass ({repaired_count} tasks được sửa)!")
    print(f"File repaired: {output_path}")
    print(f"\nĐể đánh giá kết quả cuối cùng, chạy:")
    print(f"  python evaluate/evaluate.py {output_path}")


if __name__ == "__main__":
    main()
