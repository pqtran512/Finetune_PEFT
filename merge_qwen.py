import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# CONFIG
MODEL_PATH = "./models/java-qwen/stage_2_v4"
MERGE_MODEL = "./final_merged_qwen_model_v4"
BASE_MODEL = "Qwen/CodeQwen1.5-7B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE use",DEVICE)

MAX_NEW_TOKENS = 512
# TEMPERATURE = 0.2
# TOP_P = 0.95

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# LOAD MODEL
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, dtype=torch.bfloat16,   # match training compute dtype
    device_map={"": 0},   # ép toàn bộ model vào GPU 0
)

print("Loading LoRA...")
model = PeftModel.from_pretrained(base_model, MODEL_PATH)

print("Merging LoRA...")
model = model.merge_and_unload()

print("Saving merged model...")
gc = getattr(model, "generation_config", None)
if gc is not None and not getattr(gc, "do_sample", False):
    gc.top_p = None
    gc.top_k = None
    gc.temperature = None
model.save_pretrained(MERGE_MODEL, safe_serialization=True)
tokenizer.save_pretrained(MERGE_MODEL)