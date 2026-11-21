"""Qwen2.5-Coder-32B plan v2 — Repair pass cho COMPILE_ERROR.

Đọc .failures.jsonl từ evaluate.py, với mỗi entry status=COMPILE_ERROR có
missing_methods (parse từ "cannot find symbol method X(Y)"), gọi lại model
yêu cầu emit ĐÚNG những helper đó. Output là một jsonl đầy đủ 158 task
(merge các task PASSED nguyên gốc với các task được repair) để đem chạy
evaluate.py lần 2.

Lưu ý:
- Chỉ repair COMPILE_ERROR (FAILED/TIMEOUT là lỗi logic, không phải lỗi
  thiếu helper — không trong scope của pass này).
- Repair prompt yêu cầu model TRẢ VỀ chỉ phần helper code, dán tiếp vào
  cuối completion sẵn có.

Cách chạy:
    python repair_qwen.py [input_inference.jsonl] [failures.jsonl]

Mặc định:
    input_inference  = java_qwen32b_inference_v2.jsonl
    failures         = java_qwen32b_inference_v2.jsonl.failures.jsonl
    output           = java_qwen32b_inference_v2_repaired.jsonl
"""
import os
import json
import sys
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


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
CACHE_DIR = str(Path(__file__).parent.parent / "cache" / "hugging_face")
INFERENCE_DTYPE = os.environ.get("INFERENCE_DTYPE", "bf16").lower()

DEFAULT_INPUT = "java_qwen32b_inference_v2.jsonl"
DEFAULT_FAILURES = "java_qwen32b_inference_v2.jsonl.failures.jsonl"
OUTPUT_FILE = "java_qwen32b_inference_v2_repaired.jsonl"

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.directives import REPAIR_INSTRUCTION
from shared.cleaner import clean_helper_output
from shared.repair import build_repair_prompt


def load_model():
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
    return tokenizer, model


def repair(input_path: str, failures_path: str, output_path: str):
    if not os.path.exists(input_path):
        print(f"Không thấy {input_path}")
        sys.exit(1)
    if not os.path.exists(failures_path):
        print(f"Không thấy {failures_path}")
        sys.exit(1)

    failures_by_id = {}
    with open(failures_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("status") == "COMPILE_ERROR":
                failures_by_id[entry["task_id"]] = entry

    if not failures_by_id:
        print("Không có COMPILE_ERROR nào để repair. Copy nguyên input sang output.")
        with open(input_path, "r", encoding="utf-8") as src, open(output_path, "w", encoding="utf-8") as dst:
            dst.write(src.read())
        return

    print(f"Sẽ repair {len(failures_by_id)} task COMPILE_ERROR.")
    tokenizer, model = load_model()

    eos_ids = [tokenizer.eos_token_id]
    for tok in ("<|endoftext|>", "<|im_end|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid != tokenizer.unk_token_id and tid not in eos_ids:
            eos_ids.append(tid)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    stop_strings = ["\n}\n", "\n// ===", "\n```"]

    repaired_completions = {}
    for task_id, entry in tqdm(failures_by_id.items(), desc="Repairing"):
        prompt_text = build_repair_prompt(entry)
        # build_repair_prompt ends mid-token "    private static " — keep that
        # as a soft hint to the model to start emitting a method signature.
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        input_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=eos_ids,
                stop_strings=stop_strings,
                tokenizer=tokenizer,
            )
        gen_ids = outputs[0][input_length:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        helpers = clean_helper_output(gen_text)

        # Prepend the "    private static " hint we seeded so the helper is
        # a valid signature even when the model continued mid-token.
        if helpers and not helpers.lstrip().startswith(("private", "public", "protected", "static")):
            helpers = "    private static " + helpers.lstrip()

        new_completion = entry["completion"].rstrip() + "\n\n" + helpers + "\n"
        repaired_completions[task_id] = new_completion

    with open(input_path, "r", encoding="utf-8") as src, open(output_path, "w", encoding="utf-8") as dst:
        for line in src:
            data = json.loads(line)
            if data["task_id"] in repaired_completions:
                data["completion"] = repaired_completions[data["task_id"]]
            dst.write(json.dumps(data) + "\n")

    print(f"Saved to {output_path}. Repaired {len(repaired_completions)} tasks.")
    print(f"Next: python evaluate.py {output_path}")


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    failures_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_FAILURES
    repair(input_path, failures_path, OUTPUT_FILE)


if __name__ == "__main__":
    main()
