import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json
import re
from tqdm import tqdm

# 1. Cấu hình
MODEL_PATH = "./final_merged_model"
OUTPUT_FILE = "java_inference.jsonl"

# 2. Load Model & Tokenizer
print("🚀 Loading Merged Model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 3. Load Dataset MultiPL-E (Java version)
print("📥 Loading Dataset...")
dataset = load_dataset("nuprl/MultiPL-E", "humaneval-java", split="test")

def clean_output(gen_text):
    """Lọc lấy code Java, dừng lại nếu thấy định nghĩa class mới hoặc EOF"""
    # Nếu model sinh ra ```java ... ```
    if "```java" in gen_text:
        gen_text = gen_text.split("```java")[-1].split("```")[0]
    return gen_text.strip()

# 4. Chạy Inference
results = []
for item in tqdm(dataset, desc="Generating Java Code"):
    prompt = item["prompt"]
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0,      # Pass@1 cần ổn định tuyệt đối
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Chỉ lấy phần model sinh thêm sau prompt
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_code = full_output[len(prompt):] 
    
    results.append({
        "task_id": item["name"],
        "prompt": prompt,
        "completion": clean_output(generated_code),
        "tests": item["tests"]
    })

# 5. Lưu ra file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for entry in results:
        f.write(json.dumps(entry) + "\n")

print(f"✅ Đã xong! Kết quả lưu tại {OUTPUT_FILE}")