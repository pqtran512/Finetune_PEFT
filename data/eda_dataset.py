import json
import re
import sys
from collections import Counter
from pathlib import Path

# Thư mục chứa dữ liệu
DATA_DIR = Path(__file__).resolve().parent

def run_eda(file_path: Path):
    if not file_path.exists():
        print(f"❌ File {file_path} không tồn tại!")
        return

    print(f"🔍 Đang phân tích file: {file_path.name}...\n")

    total_samples = 0
    prefix_lengths = []
    target_lengths = []
    
    # Thống kê định dạng class/imports
    has_imports = 0
    has_class_problem = 0
    has_class_solution = 0
    has_class_generic = 0  # Bất kỳ class nào khác
    is_naked_method = 0    # Không có class wrapper nào cả
    
    # Thống kê về nhiễu và định dạng lỗi
    has_backticks = 0      # Chứa ``` hoặc ```java
    has_inst_tags = 0      # Chứa [INST] hoặc [/INST]
    has_eos_in_target = 0  # Chứa <eos> hoặc </s>
    unbalanced_braces = 0  # Lệch ngoặc nhọn { } trong target
    
    # Kiểm tra helper dependencies
    helper_calls_detected = 0
    
    # Regex cho class và imports
    import_re = re.compile(r"\bimport\s+[\w\.]+;")
    class_re = re.compile(r"\bclass\s+(\w+)\b")
    
    # Regex phát hiện gọi hàm (ví dụ: helperFunc(arg1))
    call_re = re.compile(r"\b([a-zA-Z_]\w*)\s*\(")
    # Các từ khóa thông dụng cần loại bỏ khỏi danh sách gọi hàm phụ trợ
    KEYWORDS_SKIP = {
        "if", "for", "while", "switch", "catch", "try", "new", "return", 
        "super", "this", "assert", "main", "equals", "hashCode", "toString",
        "print", "println", "printf", "add", "remove", "get", "put", "size",
        "length", "charAt", "substring", "contains", "indexOf", "split", "replace",
        "Math", "min", "max", "abs", "sqrt", "pow", "round", "floor", "ceil",
        "Arrays", "asList", "Collections", "sort", "reverse", "List", "ArrayList",
        "Map", "HashMap", "Set", "HashSet", "Integer", "String", "Double", "Float",
        "Long", "Boolean", "Character", "System", "out", "in", "Stream", "stream"
    }

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            total_samples += 1
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            prefix = data.get("prefix", "")
            target = data.get("target", "")
            
            prefix_lengths.append(len(prefix))
            target_lengths.append(len(target))
            
            # 1. Kiểm tra cấu trúc bao bọc (Class/Imports)
            has_imp = bool(import_re.search(prefix))
            if has_imp:
                has_imports += 1
                
            classes = class_re.findall(prefix)
            if classes:
                if "Problem" in classes:
                    has_class_problem += 1
                elif "Solution" in classes:
                    has_class_solution += 1
                else:
                    has_class_generic += 1
            else:
                is_naked_method += 1
                
            # 2. Kiểm tra nhiễu cú pháp và tag thừa
            if "```" in prefix or "```" in target:
                has_backticks += 1
            if "[INST]" in prefix or "[/INST]" in prefix or "[INST]" in target:
                has_inst_tags += 1
            if "</s>" in target or "<eos>" in target:
                has_eos_in_target += 1
                
            # 3. Kiểm tra cân bằng ngoặc nhọn trong target
            # Vì target kết thúc bằng } đóng method nên số { phải bằng số } nếu method tự chứa block đầy đủ
            open_braces = target.count("{")
            close_braces = target.count("}")
            if open_braces != close_braces:
                unbalanced_braces += 1
                
            # 4. Kiểm tra xem target có gọi hàm phụ trợ bên ngoài không
            # Rất thô sơ: Tìm tất cả các lời gọi hàm, lọc bỏ keywords chuẩn, xem có tồn tại lời gọi hàm lạ không
            calls = set(call_re.findall(target))
            helpers = [c for c in calls if c not in KEYWORDS_SKIP and not c[0].isupper()]
            if helpers:
                helper_calls_detected += 1

    # In kết quả báo cáo EDA
    print(f"==================================================")
    print(f"📊 BÁO CÁO EDA DATASET: {file_path.name}")
    print(f"==================================================")
    print(f" Tổng số mẫu (samples)          : {total_samples}")
    print(f" Độ dài Prefix trung bình (char) : {sum(prefix_lengths)/max(1, total_samples):.1f}")
    print(f" Độ dài Target trung bình (char) : {sum(target_lengths)/max(1, total_samples):.1f}")
    print(f"--------------------------------------------------")
    print(f" Cấu trúc File & Wrappers:")
    print(f"  - Có chứa imports             : {has_imports} ({has_imports/max(1, total_samples)*100:.1f}%)")
    print(f"  - Có class Problem            : {has_class_problem} ({has_class_problem/max(1, total_samples)*100:.1f}%)")
    print(f"  - Có class Solution           : {has_class_solution} ({has_class_solution/max(1, total_samples)*100:.1f}%)")
    print(f"  - Có class khác               : {has_class_generic} ({has_class_generic/max(1, total_samples)*100:.1f}%)")
    print(f"  - Hàm độc lập (Naked Method)   : {is_naked_method} ({is_naked_method/max(1, total_samples)*100:.1f}%)")
    print(f"--------------------------------------------------")
    print(f" Kiểm tra độ nhiễu và định dạng:")
    print(f"  - Chứa ký tự markdown (```)   : {has_backticks} ({has_backticks/max(1, total_samples)*100:.1f}%)")
    print(f"  - Chứa tag chat ([INST])      : {has_inst_tags} ({has_inst_tags/max(1, total_samples)*100:.1f}%)")
    print(f"  - Target chứa thẻ dừng (</s>) : {has_eos_in_target} ({has_eos_in_target/max(1, total_samples)*100:.1f}%)")
    print(f"  - Mất cân bằng ngoặc nhọn      : {unbalanced_braces} ({unbalanced_braces/max(1, total_samples)*100:.1f}%)")
    print(f"--------------------------------------------------")
    print(f" Phân tích hàm phụ trợ (Helpers):")
    print(f"  - Target gọi hàm ngoài        : {helper_calls_detected} ({helper_calls_detected/max(1, total_samples)*100:.1f}%)")
    print(f"==================================================")

if __name__ == "__main__":
    target_file = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_DIR / "jsonl" / "evol_java_completion_train.jsonl"
    run_eda(target_file)
