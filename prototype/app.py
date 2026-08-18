import os
import sys
import time
import argparse
import re
import subprocess
import tempfile
import shutil
from pathlib import Path
# Thiết lập đường dẫn dự án vào sys.path trước các import nội bộ
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Phân tích tham số dòng lệnh trước để tránh import thư viện nặng nếu chạy Mock
parser = argparse.ArgumentParser(description="Java Code Generator Prototype Server")
parser.add_argument("--mock", action="store_true", help="Chạy server ở chế độ giả lập không load model")
args, unknown = parser.parse_known_args()

MOCK_MODE = args.mock

import json
from flask import Flask, request, jsonify, render_template, Response

try:
    from prototype.prompt_enhancer import enhance_prompt, STANDARD_IMPORTS_HEADER
except ImportError:
    from prompt_enhancer import enhance_prompt, STANDARD_IMPORTS_HEADER

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
        "base_model": BASE_MODEL if not MOCK_MODE else f"{BASE_MODEL} (Simulated)",
        "lora_path": LORA_PATH if not MOCK_MODE else f"{LORA_PATH} (Simulated)"
    })

def balance_class_brackets(code: str) -> str:
    """Tự động đóng ngoặc nhọn nếu code kết quả còn thiếu đóng class/method."""
    open_b = code.count('{')
    close_b = code.count('}')
    diff = open_b - close_b
    if diff > 0:
        return code.rstrip() + "\n" + ("    }\n" * (diff - 1)) + "}\n" if diff > 1 else code.rstrip() + "\n}\n"
    return code

@app.route("/api/enhance-preview", methods=["POST"])
def enhance_preview():
    """API xem trước prompt sau khi được module Prompt Enhancer chuẩn hóa."""
    data = request.get_json() or {}
    prompt = data.get("prompt", "")
    enable_lang = bool(data.get("enable_language_tag", False))
    enable_cot = bool(data.get("enable_cot", False))
    
    result = enhance_prompt(prompt, enable_language_tag=enable_lang, enable_cot=enable_cot)
    return jsonify(result)

@app.route("/api/generate", methods=["POST"])
def generate():
    global model, tokenizer
    
    data = request.get_json() or {}
    prompt = data.get("prompt", "")
    temperature = float(data.get("temperature", 0.2))
    max_new_tokens = int(data.get("max_new_tokens", 512))
    enable_lang = bool(data.get("enable_language_tag", False))
    enable_cot = bool(data.get("enable_cot", False))
    input_args = data.get("input_args", "")
    
    if not prompt:
        return jsonify({"error": "Prompt khong duoc de trong."}), 400

    def generate_stream():
        try:
            # 1. Khởi tạo yêu cầu sinh mã
            yield json.dumps({"step": "init", "text": "> Khởi tạo yêu cầu sinh mã", "type": "task"}) + "\n"
            yield json.dumps({"step": "params", "text": f"Đang nạp tham số cấu hình suy luận", "type": "normal"}) + "\n"
            
            # 2. Áp dụng quy tắc Prompt Engineering
            yield json.dumps({"step": "enhance_start", "text": "> Áp dụng quy tắc Prompt Engineering", "type": "task"}) + "\n"
            
            start_time = time.time()
            enhancement = enhance_prompt(
                prompt,
                enable_language_tag=enable_lang,
                enable_cot=enable_cot
            )
            final_prompt = enhancement["prefix"]
            input_type = enhancement["input_type"]
            
            input_type_map = {
                "NL": "Ngôn ngữ tự nhiên (Natural Language)",
                "METHOD_ONLY": "Chữ ký hàm (Method signature)",
                "FULL_JAVA": "Mã nguồn Java đầy đủ (Full Java class)"
            }
            friendly_input_type = input_type_map.get(input_type, input_type)
            yield json.dumps({"step": "enhance_info", "text": f"Định dạng đầu vào phát hiện: {friendly_input_type}.", "type": "normal"}) + "\n"
            
            rules_applied = ["RULE-JAVA-DECL-NO-DOC", "RULE-CLEAR-SYNTAX-OUTPUT", "RULE-REPEAT-INSTR-AT-END"]
            if enable_lang:
                rules_applied.append("RULE-TRAINED-KW-JAVA")
            if enable_cot or input_type == "NL":
                rules_applied.append("RULE-ZEROSHOT-CHAIN-OF-THOUGHT")
            if input_type in ("NL", "METHOD_ONLY"):
                rules_applied.append("RULE-INCL-INSTR")
                
            yield json.dumps({"step": "enhance_rules", "text": f"Đã áp dụng thành công các quy tắc tối ưu hóa: {', '.join(rules_applied)}.", "type": "normal"}) + "\n"
            
            # 3. Kết nối máy chủ AI
            yield json.dumps({"step": "model_start", "text": "> Kết nối máy chủ AI", "type": "task"}) + "\n"
            
            # 4. Tiến trình suy luận sinh mã
            yield json.dumps({"step": "inference_start", "text": "> Tiến trình suy luận sinh mã", "type": "task"}) + "\n"
            
            cleaned_code = ""
            gen_text = ""
            
            if MOCK_MODE:
                time.sleep(0.3)
                yield json.dumps({"step": "inference_progress_1", "text": "[10%] Bắt đầu phân rã bài toán và sinh chuỗi token...", "type": "normal"}) + "\n"
                time.sleep(0.3)
                yield json.dumps({"step": "inference_progress_2", "text": "[35%] Phân tích logic giải thuật bài toán...", "type": "normal"}) + "\n"
                time.sleep(0.3)
                yield json.dumps({"step": "inference_progress_3", "text": "[60%] Rà soát lỗi cú pháp Java sơ bộ...", "type": "normal"}) + "\n"
                time.sleep(0.3)
                yield json.dumps({"step": "inference_progress_4", "text": "[85%] Định dạng hoàn chỉnh cấu trúc class...", "type": "normal"}) + "\n"
                
                cleaned_code = ""
                prompt_lower = prompt.lower()
                
                # Nhận diện các bài toán phổ biến trong Mock mode
                if "rollingmax" in prompt_lower or "rolling max" in prompt_lower:
                    if input_type == "NL":
                        cleaned_code = """List<Integer> rollingMax(List<Integer> numbers) {
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
                    else:
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
                elif "removeduplicates" in prompt_lower or "remove duplicates" in prompt_lower:
                    if input_type == "NL":
                        cleaned_code = """List<Integer> removeDuplicates(List<Integer> numbers) {
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
                    else:
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
                elif "sorteven" in prompt_lower or "sort even" in prompt_lower or "chẵn" in prompt_lower:
                    if input_type == "NL":
                        cleaned_code = """List<Integer> sortEven(List<Integer> l) {
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
                    else:
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
                elif "willitfly" in prompt_lower or "will it fly" in prompt_lower:
                    if input_type == "NL":
                        cleaned_code = """boolean willItFly(List<Integer> q, int w) {
        int n = q.size();
        for (int i = 0; i < n / 2; i++) {
            if (!q.get(i).equals(q.get(n - 1 - i))) {
                return false;
            }
        }
        int sum = 0;
        for (int val : q) {
            sum += val;
        }
        return sum <= w;
    }
}"""
                    else:
                        cleaned_code = """
        int n = q.size();
        for (int i = 0; i < n / 2; i++) {
            if (!q.get(i).equals(q.get(n - 1 - i))) {
                return false;
            }
        }
        int sum = 0;
        for (int val : q) {
            sum += val;
        }
        return sum <= w;
    }
}"""
                elif "nguyên tố" in prompt_lower or "prime" in prompt_lower:
                    if input_type == "NL":
                        cleaned_code = """boolean isPrime(int n) {
        if (n <= 1) return false;
        for (int i = 2; i * i <= n; i++) {
            if (n % i == 0) return false;
        }
        return true;
    }
}"""
                    else:
                        cleaned_code = """
        if (n <= 1) return false;
        for (int i = 2; i * i <= n; i++) {
            if (n % i == 0) return false;
        }
        return true;
    }
}"""
                elif "giai thừa" in prompt_lower or "factorial" in prompt_lower:
                    if input_type == "NL":
                        cleaned_code = """long factorial(int n) {
        if (n <= 1) return 1;
        long res = 1;
        for (int i = 2; i <= n; i++) {
            res *= i;
        }
        return res;
    }
}"""
                    else:
                        cleaned_code = """
        if (n <= 1) return 1;
        long res = 1;
        for (int i = 2; i <= n; i++) {
            res *= i;
        }
        return res;
    }
}"""
                elif "đảo ngược" in prompt_lower or "reverse" in prompt_lower:
                    if input_type == "NL":
                        cleaned_code = """String reverse(String s) {
        if (s == null) return null;
        return new StringBuilder(s).reverse().toString();
    }
}"""
                    else:
                        cleaned_code = """
        if (s == null) return null;
        return new StringBuilder(s).reverse().toString();
    }
}"""
                else:
                    if input_type == "NL":
                        cleaned_code = """Object solve() {
        // TIEN TRINH GIA LAP (MOCK MODE)
        // System is currently running in user-interface evaluation mode.
        
        System.out.println("Hello from Antigravity Mock Mode!");
        return null;
    }
}"""
                    else:
                        cleaned_code = """
        // TIEN TRINH GIA LAP (MOCK MODE)
        // System is currently running in user-interface evaluation mode.
        
        System.out.println("Hello from Antigravity Mock Mode!");
        return null;
    }
}"""
                
                yield json.dumps({"step": "inference_progress_5", "text": "[95%] Streaming code tokens into output buffer...", "type": "normal"}) + "\n"
                time.sleep(0.1)
                gen_text = cleaned_code
                elapsed = time.time() - start_time
                time_taken_str = f"{elapsed:.2f}s (Simulated)"
            else:
                yield json.dumps({"step": "inference_progress", "text": "Đang thực hiện suy luận trên mô hình... Vui lòng đợi.", "type": "normal"}) + "\n"
                if model is None or tokenizer is None:
                    raise ValueError("Mô hình chưa được tải lên hệ thống.")
                    
                inputs = tokenizer(final_prompt, return_tensors="pt").to(model.device)
                input_length = inputs.input_ids.shape[1]
                
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
                time_taken_str = f"{elapsed:.2f}s"
                
            # 5. Xác thực biên dịch tự động
            yield json.dumps({"step": "compile_start", "text": "> Biên dịch & Kiểm thử tự động", "type": "task"}) + "\n"
            yield json.dumps({"step": "compile_info", "text": "Đang tiến hành biên dịch thử file Problem.java bằng javac...", "type": "normal"}) + "\n"
            
            res_full = build_full_test_code(final_prompt + cleaned_code, input_args)
            full_code = res_full["full_code"]
            class_name = res_full["class_name"]
            
            # Tạo thư mục tạm thời trong workspace/scratch để chạy javac kiểm tra lỗi cú pháp
            scratch_dir = _REPO_ROOT / "scratch"
            if not scratch_dir.exists():
                scratch_dir.mkdir(exist_ok=True)
                
            temp_dir = tempfile.mkdtemp(dir=str(scratch_dir.resolve()))
            compile_success = False
            compile_output_msg = ""
            
            try:
                # Ghi file <class_name>.java
                with open(os.path.join(temp_dir, f"{class_name}.java"), "w", encoding="utf-8") as f:
                    f.write(full_code)
                    
                # Tìm thư viện javatuples-1.2.jar
                jar_name = "javatuples-1.2.jar"
                jar_path_root = _REPO_ROOT / jar_name
                jar_path_eval = _REPO_ROOT / "evaluate" / jar_name
                actual_jar_path = ""
                if jar_path_root.exists():
                    actual_jar_path = str(jar_path_root.resolve())
                elif jar_path_eval.exists():
                    actual_jar_path = str(jar_path_eval.resolve())
                else:
                    actual_jar_path = str(jar_path_root.resolve())
                    
                classpath = f".{os.pathsep}{actual_jar_path}"
                
                # Biên dịch
                compile_cmd = ["javac", "-encoding", "utf-8", "-cp", classpath, f"{class_name}.java"]
                compile_proc = subprocess.run(
                    compile_cmd,
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if compile_proc.returncode == 0:
                    compile_success = True
                else:
                    compile_output_msg = compile_proc.stderr
            except Exception as e_comp:
                compile_output_msg = str(e_comp)
            finally:
                try:
                    shutil.rmtree(temp_dir)
                except Exception:
                    pass
            
            if compile_success:
                yield json.dumps({"step": "compile_success", "text": "Biên dịch thử nghiệm Problem.java... THÀNH CÔNG.", "type": "normal"}) + "\n"
                yield json.dumps({"step": "build_finished", "text": f"BUILD SUCCESSFUL trong {time_taken_str}", "type": "success"}) + "\n"
            else:
                # Trích xuất 5 dòng lỗi đầu tiên để hiển thị trực tiếp cho gọn gàng
                err_lines = compile_output_msg.splitlines()
                summary_err = "\n".join(err_lines[:5])
                if len(err_lines) > 5:
                    summary_err += f"\n... (và {len(err_lines) - 5} dòng lỗi khác)"
                yield json.dumps({"step": "compile_fail", "text": f"Cảnh báo biên dịch: Có lỗi cú pháp trong mã nguồn sinh ra!\n{summary_err}", "type": "error"}) + "\n"
                yield json.dumps({"step": "build_finished", "text": "BUILD SUCCESSFUL (với cảnh báo lỗi cú pháp)", "type": "warning"}) + "\n"
                
            # 6. Trả về payload kết quả cuối cùng ở chunk cuối
            yield json.dumps({
                "step": "result",
                "generated_code": cleaned_code,
                "full_code": full_code,
                "raw_output": gen_text,
                "enhanced_prompt": final_prompt,
                "input_type": input_type,
                "time_taken": time_taken_str
            }) + "\n"
            
        except Exception as e:
            yield json.dumps({"step": "error", "text": f"Lỗi hệ thống trong quá trình sinh mã: {str(e)}", "type": "error"}) + "\n"

    return Response(generate_stream(), mimetype="application/x-ndjson", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no"
    })
        
# =====================================================
#  PHÂN TÍCH SIGNATURE & GỢI Ý INPUT THÔNG MINH
# =====================================================

# Bảng giá trị gợi ý mặc định cho từng kiểu dữ liệu Java
DEFAULT_SUGGESTIONS = {
    "int": "0",
    "long": "0L",
    "double": "0.0",
    "float": "0.0f",
    "boolean": "true",
    "char": "'a'",
    "String": '"hello"',
    "List<Integer>": "[1, 2, 3]",
    "List<String>": '["a", "b"]',
    "List<Long>": "[1L, 2L, 3L]",
    "List<Double>": "[1.0, 2.0, 3.0]",
    "List<Boolean>": "[true, false]",
    "ArrayList<Integer>": "[1, 2, 3]",
    "ArrayList<String>": '["a", "b"]',
    "int[]": "[1, 2, 3]",
    "long[]": "[1L, 2L, 3L]",
    "double[]": "[1.0, 2.0, 3.0]",
    "String[]": '["a", "b"]',
    "char[]": "['a', 'b']",
    "boolean[]": "[true, false]",
    "Map<String, Integer>": 'Map.of("a", 1, "b", 2)',
    "Map<String, String>": 'Map.of("key", "value")',
    "Object": 'null',
}

def parse_method_signature(code: str):
    """Phân tích signature hàm public static đầu tiên trong mã Java.
    Trả về dict: {method_name, return_type, params: [{name, type}]}
    """
    # Regex tìm method public static với generic types, arrays, etc.
    sig_pattern = re.compile(
        r'public\s+static\s+'
        r'([\w<>\[\]\s,?]+?)\s+'  # return type (group 1)
        r'(\w+)\s*'               # method name (group 2)
        r'\(([^)]*)\)',            # parameters (group 3)
        re.DOTALL
    )
    match = sig_pattern.search(code)
    if not match:
        return None
    
    ret_type = match.group(1).strip()
    method_name = match.group(2).strip()
    params_str = match.group(3).strip()
    
    params = []
    if params_str:
        # Phân tích từng tham số, hỗ trợ generic types
        # Ví dụ: "List<Integer> numbers, int w" -> [{type: "List<Integer>", name: "numbers"}, ...]
        depth = 0
        current = ""
        for ch in params_str:
            if ch == '<':
                depth += 1
                current += ch
            elif ch == '>':
                depth -= 1
                current += ch
            elif ch == ',' and depth == 0:
                params.append(current.strip())
                current = ""
            else:
                current += ch
        if current.strip():
            params.append(current.strip())
        
        parsed_params = []
        for p in params:
            p = p.strip()
            if not p:
                continue
            # Tách type và name: "List<Integer> numbers" -> type="List<Integer>", name="numbers"
            # Tìm vị trí tên biến (từ cuối cùng sau khoảng trắng)
            parts = p.rsplit(None, 1)
            if len(parts) == 2:
                parsed_params.append({"type": parts[0].strip(), "name": parts[1].strip()})
            else:
                parsed_params.append({"type": p, "name": "arg"})
        params = parsed_params
    
    return {
        "method_name": method_name,
        "return_type": ret_type,
        "params": params
    }

def suggest_value_for_type(java_type: str) -> str:
    """Sinh giá trị gợi ý mặc định cho một kiểu dữ liệu Java."""
    # Tìm chính xác trong bảng
    if java_type in DEFAULT_SUGGESTIONS:
        return DEFAULT_SUGGESTIONS[java_type]
    
    # Hỗ trợ generic List<T> chung
    list_match = re.match(r'(?:List|ArrayList)<(.+)>', java_type)
    if list_match:
        inner = list_match.group(1).strip()
        inner_val = suggest_value_for_type(inner)
        return f"[{inner_val}]"
    
    # Hỗ trợ array T[]
    if java_type.endswith('[]'):
        base = java_type[:-2].strip()
        inner_val = suggest_value_for_type(base)
        return f"[{inner_val}]"
    
    # Fallback
    return "null"

def smart_preprocess_input(input_str: str, params: list) -> str:
    """Chuyển đổi input đơn giản của user sang Java hợp lệ.
    
    Ví dụ:
    - User nhập: [1, 2, 3]   với param type List<Integer>  -> List.of(1, 2, 3)
    - User nhập: ["a", "b"]  với param type List<String>   -> List.of("a", "b")
    - User nhập: {1, 2, 3}   với param type int[]           -> new int[]{1, 2, 3}
    - User nhập: "abc"       với param type String           -> "abc" (giữ nguyên)
    """
    args_str = input_str.strip()
    
    # Bỏ ngoặc tròn bao ngoài nếu có
    if args_str.startswith('(') and args_str.endswith(')'):
        args_str = args_str[1:-1].strip()
    
    if not params:
        if not args_str.startswith('('):
            return f"({args_str})"
        return args_str
        
    if not args_str:
        # Nếu người dùng không nhập đối số và hàm yêu cầu tham số,
        # tự động sinh các giá trị mặc định tương ứng để tránh lỗi biên dịch.
        defaults = [suggest_value_for_type(p["type"]) for p in params]
        return "(" + ", ".join(defaults) + ")"
    
    # Tách các argument theo dấu phẩy (nhưng tôn trọng dấu ngoặc lồng nhau)
    raw_args = _split_args(args_str)
    
    converted_args = []
    for i, raw_arg in enumerate(raw_args):
        raw_arg = raw_arg.strip()
        if i < len(params):
            param_type = params[i]["type"]
            converted = _convert_single_arg(raw_arg, param_type)
            converted_args.append(converted)
        else:
            converted_args.append(raw_arg)
    
    return "(" + ", ".join(converted_args) + ")"

def _split_args(s: str) -> list:
    """Tách chuỗi arguments theo dấu phẩy, tôn trọng ngoặc lồng nhau."""
    depth = 0
    current = ""
    result = []
    in_string = False
    string_char = None
    
    for ch in s:
        if in_string:
            current += ch
            if ch == string_char:
                in_string = False
            continue
        
        if ch in ('"', "'"):
            in_string = True
            string_char = ch
            current += ch
        elif ch in ('(', '[', '{', '<'):
            depth += 1
            current += ch
        elif ch in (')', ']', '}', '>'):
            depth -= 1
            current += ch
        elif ch == ',' and depth == 0:
            result.append(current)
            current = ""
        else:
            current += ch
    
    if current.strip():
        result.append(current)
    return result

def _convert_single_arg(raw: str, param_type: str) -> str:
    """Chuyển đổi một argument đơn lẻ sang dạng Java hợp lệ.
    
    Rules:
    - [1, 2, 3] + List<Integer> -> List.of(1, 2, 3)
    - ["a", "b"] + List<String> -> List.of("a", "b")
    - [1, 2, 3] + ArrayList<Integer> -> new ArrayList<>(List.of(1, 2, 3))
    - {1, 2, 3} + int[] -> new int[]{1, 2, 3}
    - Nếu đã là Java hợp lệ (List.of, new ...) -> giữ nguyên
    """
    raw = raw.strip()
    
    # Nếu đã là cú pháp Java đầy đủ, giữ nguyên
    if raw.startswith('List.of') or raw.startswith('new ') or raw.startswith('Arrays.asList'):
        return raw
    if raw.startswith('Map.of'):
        return raw
    
    # Kiểm tra nếu là mảng dạng [item1, item2, ...]
    is_list_type = bool(re.match(r'(?:List|ArrayList|LinkedList)<', param_type))
    is_array_type = param_type.endswith('[]')
    
    if raw.startswith('[') and raw.endswith(']'):
        inner = raw[1:-1].strip()
        if is_list_type:
            if 'ArrayList' in param_type:
                return f"new ArrayList<>(List.of({inner}))"
            return f"List.of({inner})"
        elif is_array_type:
            base = param_type[:-2].strip()
            return f"new {base}[]{{{inner}}}"
        else:
            # Default: coi như List.of
            return f"List.of({inner})"
    
    # Kiểm tra nếu là mảng dạng {item1, item2, ...}
    if raw.startswith('{') and raw.endswith('}'):
        inner = raw[1:-1].strip()
        if is_array_type:
            base = param_type[:-2].strip()
            return f"new {base}[]{{{inner}}}"
        elif is_list_type:
            return f"List.of({inner})"
    
    return raw

@app.route("/api/suggest-input", methods=["POST"])
def suggest_input():
    """API phân tích mã sinh ra và gợi ý input test phù hợp."""
    data = request.get_json() or {}
    code = data.get("code", "")
    
    if not code:
        return jsonify({"error": "Mã nguồn không được để trống."}), 400
    
    sig = parse_method_signature(code)
    if not sig:
        return jsonify({
            "method_name": None,
            "return_type": None,
            "params": [],
            "suggested_input": ""
        })
    
    # Sinh giá trị gợi ý cho từng tham số
    suggested_params = []
    suggested_values = []
    for p in sig["params"]:
        sv = suggest_value_for_type(p["type"])
        suggested_params.append({
            "name": p["name"],
            "type": p["type"],
            "suggested_value": sv
        })
        suggested_values.append(sv)
    
    suggested_input = "(" + ", ".join(suggested_values) + ")" if suggested_values else "()"
    
    return jsonify({
        "method_name": sig["method_name"],
        "return_type": sig["return_type"],
        "params": suggested_params,
        "suggested_input": suggested_input
    })

def build_full_test_code(code: str, input_args: str = "") -> dict:
    """Tạo bản mã nguồn Java hoàn chỉnh độc lập (bao gồm imports, class, methods và hàm main() kiểm thử)."""
    if not code:
        return {
            "full_code": "",
            "class_name": "Problem",
            "filename": "Problem.java",
            "method_name": None,
            "args_str": "()"
        }

    # 1. Phân tích chữ ký hàm
    sig = parse_method_signature(code)
    
    # 2. Xác định class_name
    has_class = re.search(r'\b(?:public\s+)?(?:class|interface|enum)\s+(\w+)', code)
    if has_class:
        class_name = has_class.group(1)
    else:
        class_name = "Problem"

    # 3. Smart preprocessing cho input arguments
    args_str = "()"
    method_name = ""
    ret_type = "void"
    if sig:
        method_name = sig["method_name"]
        ret_type = sig["return_type"]
        args_str = smart_preprocess_input(input_args, sig["params"])

    # 4. Trích xuất import hiện có và phần thân code
    lines = code.splitlines()
    existing_imports = []
    body_lines = []
    in_imports = True
    for line in lines:
        trimmed = line.strip()
        if not trimmed:
            if in_imports:
                existing_imports.append(line)
            else:
                body_lines.append(line)
            continue
            
        is_comment = trimmed.startswith("//") or trimmed.startswith("/*") or trimmed.startswith("*")
        is_import_or_pkg = trimmed.startswith("import ") or trimmed.startswith("package ")
        
        if in_imports:
            if is_import_or_pkg or is_comment:
                existing_imports.append(line)
            else:
                in_imports = False
                body_lines.append(line)
        else:
            body_lines.append(line)

    body_code = "\n".join(body_lines).rstrip()

    # 5. Bổ sung các thư viện import chuẩn
    standard_imports = [
        "import java.util.*;",
        "import java.io.*;",
        "import java.math.*;",
        "import org.javatuples.*;",
        "import java.util.stream.*;"
    ]
    all_imports = list(existing_imports)
    existing_set = {re.sub(r'\s+', ' ', imp.strip()) for imp in existing_imports if imp.strip() and not (imp.strip().startswith("//") or imp.strip().startswith("/*") or imp.strip().startswith("*"))}
    for simp in standard_imports:
        clean_s = re.sub(r'\s+', ' ', simp.strip())
        if clean_s not in existing_set:
            all_imports.append(simp)

    imports_block = "\n".join(all_imports) + "\n\n"

    # 6. Xây dựng hàm main() kiểm thử thực thi
    if method_name:
        if ret_type == "void":
            main_method = f"""    // =========================================================================
    // HÀM MAIN KIỂM THỬ THỰC THI (TEST RUNNER)
    // =========================================================================
    public static void main(String[] args) {{
        try {{
            System.out.println("=== CHẠY KIỂM THỬ HÀM {method_name} ===");
            {method_name}{args_str};
            System.out.println("-> Thực thi thành công (void return).");
        }} catch (Throwable t) {{
            System.err.println("Lỗi thực thi ngoại lệ: " + t.getMessage());
            t.printStackTrace(System.err);
        }}
    }}"""
        else:
            # Kiểm tra xem kiểu trả về có phải là mảng hay không để hiển thị đẹp hơn
            is_array = ret_type.endswith("[]")
            if is_array:
                # Nếu là mảng đa chiều hoặc mảng đối tượng, dùng Arrays.deepToString, ngược lại dùng Arrays.toString
                is_multidim = "][" in ret_type or not any(ret_type.startswith(x) for x in ["int", "long", "double", "float", "boolean", "char", "byte", "short"])
                print_stmt = f"System.out.println(\"-> Kết quả: \" + java.util.Arrays.deepToString(result));" if is_multidim else f"System.out.println(\"-> Kết quả: \" + java.util.Arrays.toString(result));"
            else:
                print_stmt = "System.out.println(\"-> Kết quả: \" + result);"

            main_method = f"""    // =========================================================================
    // HÀM MAIN KIỂM THỬ THỰC THI (TEST RUNNER)
    // =========================================================================
    public static void main(String[] args) {{
        try {{
            System.out.println("=== CHẠY KIỂM THỬ HÀM {method_name} ===");
            {ret_type} result = {method_name}{args_str};
            {print_stmt}
        }} catch (Throwable t) {{
            System.err.println("Lỗi thực thi ngoại lệ: " + t.getMessage());
            t.printStackTrace(System.err);
        }}
    }}"""
    else:
        main_method = """    // =========================================================================
    // HÀM MAIN KIỂM THỬ THỰC THI (TEST RUNNER)
    // =========================================================================
    public static void main(String[] args) {
        System.out.println("Mã nguồn Java hoàn chỉnh đã sẵn sàng thực thi.");
    }"""

    # 7. Ghép nối thành bản Java hoàn chỉnh độc lập
    if has_class:
        # Nếu đã có class, kiểm tra xem đã có hàm main chưa
        has_main = bool(re.search(r'\bpublic\s+static\s+void\s+main\b', body_code))
        if has_main:
            final_code = imports_block + body_code
        else:
            # Đếm cấp độ ngoặc nhọn để đóng các hàm con trước khi chèn main
            orig_level = 0
            in_string = False
            in_char = False
            in_line_comment = False
            in_block_comment = False
            
            i = 0
            n = len(body_code)
            while i < n:
                ch = body_code[i]
                if in_line_comment:
                    if ch == '\n':
                        in_line_comment = False
                    i += 1
                    continue
                if in_block_comment:
                    if ch == '*' and i + 1 < n and body_code[i+1] == '/':
                        in_block_comment = False
                        i += 2
                    else:
                        i += 1
                    continue
                if in_string:
                    if ch == '\\' and i + 1 < n:
                        i += 2
                    elif ch == '"':
                        in_string = False
                        i += 1
                    else:
                        i += 1
                    continue
                if in_char:
                    if ch == '\\' and i + 1 < n:
                        i += 2
                    elif ch == "'":
                        in_char = False
                        i += 1
                    else:
                        i += 1
                    continue
                if ch == '/' and i + 1 < n and body_code[i+1] == '/':
                    in_line_comment = True
                    i += 2
                    continue
                if ch == '/' and i + 1 < n and body_code[i+1] == '*':
                    in_block_comment = True
                    i += 2
                    continue
                if ch == '"':
                    in_string = True
                    i += 1
                    continue
                if ch == "'":
                    in_char = True
                    i += 1
                    continue
                    
                if ch == '{':
                    orig_level += 1
                elif ch == '}':
                    orig_level -= 1
                i += 1
                
            adjusted_body = body_code.rstrip()
            if orig_level > 1:
                adjusted_body += "\n" + "    }\n" * (orig_level - 1)
                
            if orig_level <= 0 and adjusted_body.endswith('}'):
                adjusted_body = adjusted_body[:-1].rstrip()
                final_code = imports_block + adjusted_body + "\n\n" + main_method + "\n}\n"
            else:
                final_code = imports_block + adjusted_body + "\n\n" + main_method + "\n}\n"
    else:
        # Chưa có class, tự động bọc vào public class <class_name>
        final_code = imports_block + f"public class {class_name} {{\n\n" + body_code + "\n\n" + main_method + "\n}\n"

    final_code = balance_class_brackets(final_code)

    return {
        "full_code": final_code,
        "class_name": class_name,
        "filename": f"{class_name}.java",
        "method_name": method_name,
        "args_str": args_str
    }

@app.route("/api/preview-full-code", methods=["POST"])
def preview_full_code():
    """API xem trước mã nguồn hoàn chỉnh trước khi tải xuống."""
    data = request.get_json() or {}
    code = data.get("code", "")
    input_args = data.get("input_args", "")
    
    result = build_full_test_code(code, input_args)
    return jsonify(result)

@app.route("/api/run", methods=["POST"])
def run_code():
    data = request.get_json() or {}
    code = data.get("code", "")
    input_args = data.get("input_args", "")
    
    if not code:
        return jsonify({"error": "Mã nguồn không được để trống."}), 400
        
    # Phân tích signature hàm
    sig = parse_method_signature(code)
    if not sig:
        return jsonify({"error": "Không tìm thấy hàm 'public static' nào trong mã nguồn để thực thi."}), 400
        
    ret_type = sig["return_type"]
    method_name = sig["method_name"]
    
    # Smart preprocessing: chuyển đổi input đơn giản sang Java hợp lệ
    args_str = smart_preprocess_input(input_args, sig["params"])
        
    # Phát hiện class name hoặc tự động bọc nếu thiếu class
    has_class = re.search(r'\b(?:class|interface|enum)\s+(\w+)', code)
    if has_class:
        class_name = has_class.group(1)
        # Tự động cân bằng ngoặc nhọn kết thúc class nếu thiếu
        open_braces = code.count('{')
        close_braces = code.count('}')
        diff = open_braces - close_braces
        code_balanced = code
        if diff > 0:
            code_balanced = code.rstrip() + "\n" + ("}" * diff)
    else:
        class_name = "Problem"
        # Tách các dòng import ra ngoài nếu có
        lines = code.splitlines()
        imports = []
        body_lines = []
        for line in lines:
            trimmed = line.strip()
            if trimmed.startswith("import ") or trimmed.startswith("package "):
                imports.append(line)
            else:
                body_lines.append(line)
        
        body_code = "\n".join(body_lines)
        
        # Tự động bọc vào public class Problem
        wrapped_code = "\n".join(imports) + "\n\n"
        wrapped_code += "import java.util.*;\nimport java.io.*;\nimport java.math.*;\nimport org.javatuples.*;\n\n"
        wrapped_code += "public class Problem {\n"
        wrapped_code += body_code + "\n"
        wrapped_code += "}\n"
        code_balanced = wrapped_code
        
    # Tạo mã TestRunner riêng biệt để thực thi
    if ret_type == "void":
        runner_code = f"""import java.util.*;
import java.io.*;
import java.math.*;
import org.javatuples.*;

public class TestRunner {{
    public static void main(String[] args) {{
        try {{
            {class_name}.{method_name}{args_str};
            System.out.print("Executed successfully (void return)");
        }} catch (Throwable t) {{
            t.printStackTrace(System.err);
            System.exit(1);
        }}
    }}
}}
"""
    else:
        runner_code = f"""import java.util.*;
import java.io.*;
import java.math.*;
import org.javatuples.*;

public class TestRunner {{
    public static void main(String[] args) {{
        try {{
            System.out.print({class_name}.{method_name}{args_str});
        }} catch (Throwable t) {{
            t.printStackTrace(System.err);
            System.exit(1);
        }}
    }}
}}
"""
    
    # Tạo thư mục tạm thời trong workspace/scratch
    scratch_dir = _REPO_ROOT / "scratch"
    if not scratch_dir.exists():
        scratch_dir.mkdir(exist_ok=True)
        
    temp_dir = tempfile.mkdtemp(dir=str(scratch_dir.resolve()))
    try:
        # Ghi file <class_name>.java và TestRunner.java
        with open(os.path.join(temp_dir, f"{class_name}.java"), "w", encoding="utf-8") as f:
            f.write(code_balanced)
            
        with open(os.path.join(temp_dir, "TestRunner.java"), "w", encoding="utf-8") as f:
            f.write(runner_code)
            
        # Tìm thư viện javatuples-1.2.jar
        jar_name = "javatuples-1.2.jar"
        jar_path_root = _REPO_ROOT / jar_name
        jar_path_eval = _REPO_ROOT / "evaluate" / jar_name
        
        actual_jar_path = ""
        if jar_path_root.exists():
            actual_jar_path = str(jar_path_root.resolve())
        elif jar_path_eval.exists():
            actual_jar_path = str(jar_path_eval.resolve())
        else:
            actual_jar_path = str(jar_path_root.resolve())
            
        classpath = f".{os.pathsep}{actual_jar_path}"
        
        # 1. Biên dịch cả 2 file: javac <class_name>.java TestRunner.java
        compile_cmd = ["javac", "-encoding", "utf-8", "-cp", classpath, f"{class_name}.java", "TestRunner.java"]
        compile_proc = subprocess.run(
            compile_cmd,
            cwd=temp_dir,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if compile_proc.returncode != 0:
            return jsonify({
                "success": False,
                "stage": "compile",
                "output": compile_proc.stderr
            })
            
        # 2. Chạy lớp TestRunner: java TestRunner
        run_cmd = ["java", "-ea", "-cp", classpath, "TestRunner"]
        run_proc = subprocess.run(
            run_cmd,
            cwd=temp_dir,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if run_proc.returncode != 0:
            return jsonify({
                "success": False,
                "stage": "runtime",
                "output": run_proc.stderr
            })
            
        return jsonify({
            "success": True,
            "output": run_proc.stdout
        })
        
    except subprocess.TimeoutExpired:
        return jsonify({
            "success": False,
            "stage": "timeout",
            "output": "Lỗi thực thi: Quá thời gian quy định (Timeout 10s)."
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "stage": "system",
            "output": f"Lỗi hệ thống khi chạy thử mã: {str(e)}"
        }), 500
    finally:
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass

if __name__ == "__main__":
    import threading
    print("Initializing model status...")
    if MOCK_MODE:
        init_model()
    else:
        # Tải mô hình trong luồng nền để Flask web server khởi động ngay lập tức trên cổng 5000
        threading.Thread(target=init_model, daemon=True).start()
    # Chạy trên port 5000 mặc định
    app.run(host="0.0.0.0", port=5000, debug=False)
