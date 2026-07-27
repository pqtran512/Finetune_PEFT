import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


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

# CONFIG
MODEL_PATH = "./models/java-codellama-lora/evol_bo_best"
MERGE_MODEL = "./final_merged_model_evol"
BASE_MODEL = "codellama/CodeLlama-7b-hf"
# Chỉ dùng HF_HOME trong .env, ví dụ: HF_HOME=D:/cache/huggingface
CACHE_DIR = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE use", DEVICE)
print("HF_HOME / cache:", CACHE_DIR)

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.2
TOP_P = 0.95

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=CACHE_DIR)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    device_map={"": 0},
    cache_dir=CACHE_DIR,
)

print("Loading LoRA...")
model = PeftModel.from_pretrained(base_model, MODEL_PATH)

print("Merging LoRA...")
model = model.merge_and_unload()

print("Saving merged model...")
model.save_pretrained(MERGE_MODEL)
tokenizer.save_pretrained(MERGE_MODEL)
print(f"Done. Merged model saved to: {MERGE_MODEL}")
