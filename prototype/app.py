import os
import sys
import time
import argparse
from pathlib import Path
from flask import Flask, request, jsonify, render_template

# Phân tích tham số dòng lệnh trước để tránh import thư viện nặng nếu chạy Mock
parser = argparse.ArgumentParser(description="Java Code Generator Prototype Server")
parser.add_argument("--mock", action="store_true", help="Chạy server ở chế độ giả lập không load model")
args, unknown = parser.parse_known_args()

MOCK_MODE = args.mock

# Thiết lập đường dẫn dự án
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_REPO_ROOT))

# Hàm load file .env thủ công nếu có
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

_load_env(_REPO_ROOT / ".env")

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

# Cấu hình đường dẫn model
# BASE_MODEL = "Qwen/CodeQwen1.5-7B"
# LORA_PATH = str((_REPO_ROOT / "models" / "java-qwen" / "stage_2_v4").resolve())
BASE_MODEL = "codellama/CodeLlama-7b-hf"
LORA_PATH = str((_REPO_ROOT / "models" / "java-codellama-lora" / "evol_completion_bo_v2").resolve())
CACHE_DIR = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))

# Biến lưu trữ model & tokenizer
model = None
tokenizer = None
device = "cpu"
model_status = "unloaded"

if MOCK_MODE:
    device = "CPU (Mock Mode)"
    model_status = "Sẵn sàng (Mock Mode)"
    print("==================================================")
    print("   RUNNING UI PROTOTYPE IN MOCK MODE              ")
    print("   (No model weights loaded, instant startup)      ")
    print("==================================================")
else:
    # Chỉ import các thư viện nặng (torch, transformers) khi chạy thật
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    device = "cuda" if torch.cuda.is_available() else "cpu"

def init_model():
    global model, tokenizer, model_status
    if MOCK_MODE:
        print("Mock model mode initialized successfully.")
        return
        
    try:
        model_status = "Dang tai mo hinh..."
        print(f"[{model_status}] Target device: {device}")
        
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=CACHE_DIR)
        
        if device == "cuda":
            print("Configuring 8-bit quantization for GPU 12GB VRAM...")
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            print(f"Loading base model ({BASE_MODEL}) in 8-bit mode...")
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                quantization_config=bnb_config,
                device_map="auto",
                cache_dir=CACHE_DIR,
                torch_dtype=torch.float16
            )
        else:
            print(f"Loading base model ({BASE_MODEL}) on CPU (slow)...")
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                device_map="cpu",
                cache_dir=CACHE_DIR,
                torch_dtype=torch.float32
            )
            
        if Path(LORA_PATH).exists():
            print(f"Loading LoRA adapter from {LORA_PATH}...")
            model = PeftModel.from_pretrained(base_model, LORA_PATH)
            print("LoRA adapter merged successfully.")
        else:
            print(f"LoRA path not found: {LORA_PATH}. Using base model.")
            model = base_model
            
        model.eval()
        model_status = "Sẵn sàng"
        print("Model loaded successfully. Ready for inference.")
    except Exception as e:
        model_status = f"Lỗi khi tải mô hình: {str(e)}"
        print(f"Error: {e}")

# Các chuỗi dừng mặc định
STOP_STRINGS = [
    "\n    }\n",
    "\n}\n",
    "\npublic static void main",
    "\n```",
]

def clean_output(gen_text: str) -> str:
    """Làm sạch đầu ra và định dạng code Java"""
    # Nếu output trả về dạng markdown code block
    if "```java" in gen_text:
        gen_text = gen_text.split("```java", 1)[1].split("```", 1)[0]
    elif "```" in gen_text:
        gen_text = gen_text.split("```", 1)[1].split("```", 1)[0]

    # Cắt code thừa khi model cố sinh thêm class hoặc hàm khác
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

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status", methods=["GET"])
def get_status():
    return jsonify({
        "status": model_status,
        "device": device,
        "base_model": BASE_MODEL if not MOCK_MODE else "Qwen/CodeQwen1.5-7B (Simulated)",
        "lora_path": LORA_PATH if not MOCK_MODE else "models/java-qwen/stage_2_v4 (Simulated)"
    })

@app.route("/api/generate", methods=["POST"])
def generate():
    global model, tokenizer
    
    data = request.get_json() or {}
    prompt = data.get("prompt", "")
    temperature = float(data.get("temperature", 0.2))
    max_new_tokens = int(data.get("max_new_tokens", 512))
    
    if not prompt:
        return jsonify({"error": "Prompt khong duoc de trong."}), 400
        
    start_time = time.time()
    
    # XỬ LÝ MOCK MODE (GIẢ LẬP SUY LUẬN)
    if MOCK_MODE:
        # Giả lập thời gian suy luận (1.5 giây)
        time.sleep(1.5)
        
        # Sinh code mẫu tùy theo bài toán mẫu hoặc prompt của user
        cleaned_code = ""
        if "rollingMax" in prompt:
            cleaned_code = """
        List<Integer> result = new ArrayList<>();
        if (numbers.isEmpty()) return result;
        int max = numbers.get(0);
        for (int n : numbers) {
            max = Math.max(max, n);
            result.add(max);
        }
        return result;
    }
}"""
        elif "removeDuplicates" in prompt:
            cleaned_code = """
        List<Integer> result = new ArrayList<>();
        for (int n : numbers) {
            int count = 0;
            for (int x : numbers) {
                if (x == n) count++;
            }
            if (count == 1) {
                result.add(n);
            }
        }
        return result;
    }
}"""
        elif "sortEven" in prompt:
            cleaned_code = """
        List<Integer> evens = new ArrayList<>();
        for (int i = 0; i < l.size(); i += 2) {
            evens.add(l.get(i));
        }
        Collections.sort(evens);
        List<Integer> result = new ArrayList<>(l);
        int evenIdx = 0;
        for (int i = 0; i < result.size(); i += 2) {
            result.set(i, evens.get(evenIdx++));
        }
        return result;
    }
}"""
        elif "willItFly" in prompt:
            cleaned_code = """
        // Kiểm tra palindromic
        int n = q.size();
        for (int i = 0; i < n / 2; i++) {
            if (!q.get(i).equals(q.get(n - 1 - i))) {
                return false;
            }
        }
        // Kiểm tra tổng khối lượng
        int sum = 0;
        for (int val : q) {
            sum += val;
        }
        return sum <= w;
    }
}"""
        else:
            cleaned_code = f"""
        // TIEN TRINH GIA LAP (MOCK MODE)
        // System is currently running in user-interface evaluation mode.
        // User parameters:
        // - Temperature: {temperature}
        // - Max New Tokens: {max_new_tokens}
        
        System.out.println("Hello from Antigravity Mock Mode!");
        return null;
    }}
}}"""
        elapsed = time.time() - start_time
        return jsonify({
            "generated_code": cleaned_code,
            "raw_output": cleaned_code,
            "time_taken": f"{elapsed:.2f}s (Simulated)"
        })

    # XỬ LÝ INFERENCE THỰC TẾ
    if model is None or tokenizer is None:
        return jsonify({"error": "Mo hinh chua duoc tai len he thong."}), 500
        
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_length = inputs.input_ids.shape[1]
        
        # Thiết lập EOS tokens
        eos_ids = [tokenizer.eos_token_id]
        extra_eos = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        if isinstance(extra_eos, int) and extra_eos != tokenizer.unk_token_id and extra_eos not in eos_ids:
            eos_ids.append(extra_eos)
            
        do_sample = temperature > 0
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=0.95 if do_sample else None,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=eos_ids,
                stop_strings=STOP_STRINGS,
                tokenizer=tokenizer,
            )
            
        gen_ids = outputs[0][input_length:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        cleaned_code = clean_output(gen_text)
        
        elapsed = time.time() - start_time
        
        return jsonify({
            "generated_code": cleaned_code,
            "raw_output": gen_text,
            "time_taken": f"{elapsed:.2f}s"
        })
        
    except Exception as e:
        return jsonify({"error": f"Loi trong qua trinh sinh code: {str(e)}"}), 500

if __name__ == "__main__":
    print("Initializing model status...")
    init_model()
    # Chạy trên port 5000 mặc định
    app.run(host="0.0.0.0", port=5000, debug=False)
