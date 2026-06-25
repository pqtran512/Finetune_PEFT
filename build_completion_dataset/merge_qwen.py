import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# CONFIG
MODEL_PATH = "./models/java-qwen/stage_3_completion"
MERGE_MODEL = "./final_merged_qwen_model"
BASE_MODEL = "Qwen/CodeQwen1.5-7B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE use",DEVICE)
PROMPT_FILE = "datasets/humaneval-java.json"
OUTPUT_FILE = "java_predictions.jsonl"

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
    BASE_MODEL, dtype=torch.float16,
    #  device_map="auto"
    device_map={"": 0},   # ép toàn bộ model vào GPU 0
)

print("Loading LoRA...")
model = PeftModel.from_pretrained(base_model, MODEL_PATH)

print("Merging LoRA...")
model = model.merge_and_unload()

model.generation_config.do_sample = True
model.generation_config.top_p = 0.95
model.generation_config.temperature = 0.2

print("Saving merged model...")
model.save_pretrained(MERGE_MODEL, safe_serialization=True)
tokenizer.save_pretrained(MERGE_MODEL)
