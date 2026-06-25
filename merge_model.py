import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# CONFIG
MODEL_PATH = "./models/java-codellama-lora/stage_2_v1"
MERGE_MODEL = "./final_merged_model"
BASE_MODEL = "codellama/CodeLlama-7b-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE use",DEVICE)
PROMPT_FILE = "datasets/humaneval-java.json"
OUTPUT_FILE = "java_predictions.jsonl"

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.2
TOP_P = 0.95

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

print("Saving merged model...")
model.save_pretrained(MERGE_MODEL)
tokenizer.save_pretrained(MERGE_MODEL)
# model.eval()

# LOAD DATASET
# with open(PROMPT_FILE, "r") as f:
#     problems = json.load(f)

# # GENERATE
# def generate_completion(prompt):
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=MAX_NEW_TOKENS,
#             temperature=TEMPERATURE,
#             top_p=TOP_P,
#             do_sample=True,
#             pad_token_id=tokenizer.eos_token_id
#         )

#     decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     # Remove prompt from output
#     completion = decoded[len(prompt):]

#     return completion.strip()


# print("Generating...")
# with open(OUTPUT_FILE, "w") as out:
#     for problem in tqdm(problems):
#         task_id = problem["task_id"]
#         prompt = problem["prompt"]

#         completion = generate_completion(prompt)

#         result = {
#             "task_id": task_id,
#             "completion": completion
#         }

#         out.write(json.dumps(result) + "\n")

# print("Done! Saved to", OUTPUT_FILE)
