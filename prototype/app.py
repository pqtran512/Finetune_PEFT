import os
import sys
import time
import argparse
import re
import subprocess
import tempfile
import shutil
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
    
    if not params or not args_str:
        # Không có thông tin tham số, trả về nguyên gốc
        if not args_str.startswith('('):
            return f"({args_str})"
        return args_str
    
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
        compile_cmd = ["javac", "-cp", classpath, f"{class_name}.java", "TestRunner.java"]
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
    print("Initializing model status...")
    init_model()
    # Chạy trên port 5000 mặc định
    app.run(host="0.0.0.0", port=5000, debug=False)
