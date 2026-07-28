"""Plan B — Merge LoRA adapter vào base model."""
import os
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


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

LORA_PATH = "./models/codeqwen/planB"
MERGE_OUT = "./final_merged_qwen_model"
BASE_MODEL = "Qwen/CodeQwen1.5-7B"
CACHE_DIR = os.environ.get(
    "HF_HOME",
    str(Path.home() / ".cache" / "huggingface"),
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=CACHE_DIR)

print("Loading base model (bf16)...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    cache_dir=CACHE_DIR,
)

print(f"Loading LoRA adapter from {LORA_PATH}...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)

print("Merging LoRA into base...")
model = model.merge_and_unload()

print(f"Saving merged model to {MERGE_OUT}...")
gc = getattr(model, "generation_config", None)
if gc is not None and not getattr(gc, "do_sample", False):
    gc.top_p = None
    gc.top_k = None
    gc.temperature = None
model.save_pretrained(MERGE_OUT, safe_serialization=True)
tokenizer.save_pretrained(MERGE_OUT)
print("Done.")
