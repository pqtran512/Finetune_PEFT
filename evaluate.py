import os
import subprocess
import json
import re
import urllib.request

def download_javatuples():
    """Tự động tải thư viện javatuples nếu chưa có"""
    jar_name = "javatuples-1.2.jar"
    url = "https://repo1.maven.org/maven2/org/javatuples/javatuples/1.2/javatuples-1.2.jar"
    if not os.path.exists(jar_name):
        print(f"📥 Đang tải {jar_name}...")
        urllib.request.urlretrieve(url, jar_name)
        print("✅ Đã tải xong.")
    return jar_name

def clean_java_completion(completion):
    # 1. Cắt bỏ class Main hoặc hàm main tự chế
    patterns = [r"public\s+class\s+Main", r"class\s+Main", r"public\s+static\s+void\s+main"]
    for pattern in patterns:
        match = re.search(pattern, completion)
        if match:
            completion = completion[:match.start()]

    lines = completion.split('\n')

    # 2. Trích xuất body của method chính (dừng khi brace count < 0)
    body_lines = []
    bracket_count = 0
    cutoff = len(lines)

    for i, line in enumerate(lines):
        bracket_count += line.count('{') - line.count('}')
        if bracket_count < 0:
            cutoff = i
            break
        body_lines.append(line)

    method_body = "\n".join(body_lines).strip()

    # 3. Tìm helper methods được GỌI từ method body trong phần còn lại
    remaining = "\n".join(lines[cutoff:])
    helpers_code = _find_needed_helpers(method_body, remaining)

    # 4. Ghép kết quả
    if helpers_code:
        # method_body + } (đóng method chính) + helpers
        final_code = method_body + "\n    }\n" + helpers_code
        # Bỏ } cuối cùng cho đến khi balance == 0 (tests sẽ cung cấp } đóng method cuối)
        while final_code.count('{') < final_code.count('}') and final_code.endswith('}'):
            idx = final_code.rfind('}')
            final_code = (final_code[:idx] + final_code[idx+1:]).rstrip()
    else:
        final_code = method_body

    return final_code


def _extract_all_methods(text):
    """Trích xuất tất cả method definitions hoàn chỉnh từ text, trả về dict {name: code}."""
    methods = {}
    sig_re = re.compile(
        r'^[ \t]*(?:(?:public|private|protected|static|final)\s+)*'
        r'(\w+(?:<[^>]*>)?(?:\[\])*)\s+(\w+)\s*\(',
        re.MULTILINE
    )
    skip = {'main', 'if', 'for', 'while', 'switch', 'catch', 'try', 'new', 'return', 'class', 'else'}

    for match in sig_re.finditer(text):
        name = match.group(2)
        if name in skip:
            continue

        # Tìm dấu { mở đầu method
        rest = text[match.end():]
        brace_idx = rest.find('{')
        if brace_idx == -1 or brace_idx > 200:
            continue

        abs_brace = match.end() + brace_idx
        line_start = text.rfind('\n', 0, match.start())
        line_start = 0 if line_start == -1 else line_start + 1

        # Đếm ngoặc để tìm method hoàn chỉnh
        brace_count = 0
        end = None
        for j in range(abs_brace, len(text)):
            if text[j] == '{':
                brace_count += 1
            elif text[j] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = j + 1
                    break

        if end and name not in methods:
            methods[name] = text[line_start:end].strip()

    return methods


def _find_needed_helpers(method_body, remaining_text):
    """Tìm các helper methods được gọi (trực tiếp hoặc gián tiếp) từ method_body."""
    if not remaining_text.strip():
        return ""

    available = _extract_all_methods(remaining_text)
    if not available:
        return ""

    # Tìm helpers cần thiết theo chuỗi gọi (transitive)
    needed = []
    visited = set()
    code_to_scan = method_body

    found = True
    while found:
        found = False
        calls = set(re.findall(r'\b([a-zA-Z_]\w*)\s*\(', code_to_scan))
        code_to_scan = ""
        for name in calls:
            if name not in visited and name in available:
                visited.add(name)
                needed.append(name)
                code_to_scan += "\n" + available[name]
                found = True

    if needed:
        return "\n".join(available[n] for n in needed)
    return ""

def evaluate(input_file="java_qwen_inference.jsonl"):
    jar_path = download_javatuples()
    
    if not os.path.exists(input_file):
        print(f"❌ Không thấy file {input_file}!")
        return

    passed, total = 0, 0
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)

            # CHỈ CHẠY TASK 5 (HumanEval_4_mean_absolute_deviation)
            # Lưu ý: Task 5 trong danh sách của bạn có task_id là HumanEval_4...
            # if "HumanEval_4_" not in data["task_id"]:
            #     continue

            # Làm sạch code
            cleaned_comp = clean_java_completion(data["completion"])
            
            # Ghép code: Prompt + Cleaned Completion + Tests
            full_code = data["prompt"] + "\n" + cleaned_comp + "\n" + data["tests"]
            
            # Luôn dùng Problem.java
            file_name = "Problem.java"
            with open(file_name, "w", encoding="utf-8") as jf:
                jf.write(full_code)

            # 1. Biên dịch (javac) - Thêm classpath chứa JAR
            # Lưu ý dấu ; dùng cho Windows
            compile_cmd = f'javac -cp ".;{jar_path}" {file_name}'
            cp = subprocess.run(compile_cmd, capture_output=True, text=True, shell=True)
            
            status = "FAILED"
            if cp.returncode != 0:
                status = "COMPILE_ERROR"
                # Nếu muốn xem lỗi cụ thể, bỏ comment dòng dưới:
                print(f"Lỗi tại {data['task_id']}:\n{cp.stderr}") 
            else:
                # 2. Chạy test (java) - Thêm classpath và -ea
                try:
                    run_cmd = f'java -ea -cp ".;{jar_path}" Problem'
                    rp = subprocess.run(run_cmd, capture_output=True, text=True, timeout=10, shell=True)
                    if rp.returncode == 0:
                        status = "PASSED"
                        passed += 1
                except subprocess.TimeoutExpired:
                    status = "TIMEOUT"

            total += 1
            print(f"[{total}] {data['task_id']}: {status}")

            # Dọn dẹp file tạm
            for f_temp in ["Problem.java", "Problem.class"]:
                if os.path.exists(f_temp): os.remove(f_temp)

    print("\n" + "="*40)
    print(f"🏆 KẾT QUẢ CUỐI CÙNG: {(passed/total)*100:.2f}% ({passed}/{total})")
    print("="*40)

if __name__ == "__main__":
    import sys
    fname = sys.argv[1] if len(sys.argv) > 1 else "java_qwen_inference_v4.jsonl"
    evaluate(fname)