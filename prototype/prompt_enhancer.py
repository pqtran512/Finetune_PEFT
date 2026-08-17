"""Module Prompt Enhancer & Normalizer cho Java Code Generator Prototype.

Hiện thực hóa 6 quy tắc Prompt Engineering từ Báo cáo Chuyên đề ND4:
- RULE-INCL-INSTR (Quy tắc 1): Đóng gói mô tả vào Javadoc comment chuẩn.
- RULE-JAVA-DECL-NO-DOC (Quy tắc 3): Sinh block Imports chuẩn và khung class Problem.
- RULE-TRAINED-KW-JAVA (Quy tắc 5): Hỗ trợ chèn từ khóa ngữ cảnh `// language: Java`.
- RULE-REPEAT-INSTR-AT-END (Quy tắc 9): Đặt Javadoc sát ngay phía trên chữ ký hàm.
- RULE-CLEAR-SYNTAX-OUTPUT (Quy tắc 12): Chuẩn hóa điểm kết thúc prompt (`public static ` hoặc ` {\n`).
- RULE-ZEROSHOT-CHAIN-OF-THOUGHT (Quy tắc 14): Định dạng các bước suy luận CoT trong Javadoc.
"""
from __future__ import annotations

import re
from typing import TypedDict


# Danh sách các thư viện import chuẩn (khớp 100% với tập huấn luyện)
STANDARD_IMPORTS = [
    "import java.util.*;",
    "import java.lang.reflect.*;",
    "import org.javatuples.*;",
    "import java.security.*;",
    "import java.math.*;",
    "import java.io.*;",
    "import java.util.stream.*;",
]

STANDARD_IMPORTS_HEADER = "\n".join(STANDARD_IMPORTS) + "\n"

# Regex nhận diện chữ ký hàm Java (ví dụ: public static int add(int a, int b))
METHOD_SIG_RE = re.compile(
    r"(?:public|private|protected|static|\s)*"
    r"(?:(?:public|private|protected|static|final|abstract|synchronized|native)\s+)*"
    r"(?:[\w<>\[\],\s\?]+?)\s+"
    r"(\w+)\s*\([^)]*\)\s*"
    r"(?:throws[^{]*)?",
    re.MULTILINE,
)

# Regex nhận diện khai báo class/interface
CLASS_DECL_RE = re.compile(r"\b(?:public\s+)?(?:class|interface|enum)\s+(\w+)", re.MULTILINE)

# Regex trích xuất comment
LINE_COMMENT_RE = re.compile(r"//\s*(.*)")
BLOCK_COMMENT_RE = re.compile(r"/\*+([\s\S]*?)\*+/", re.DOTALL)


class EnhancedPromptResult(TypedDict):
    prefix: str
    input_type: str
    is_enhanced: bool
    raw_input: str


# ==============================================================================
# 1. NHẬN DIỆN KIỂU ĐẦU VÀO (INPUT TYPE DETECTION)
# ==============================================================================

def detect_input_type(text: str) -> str:
    """Xác định kiểu dữ liệu của text đầu vào:
    - 'FULL_JAVA': Đã có khai báo class/interface Java hoàn chỉnh
    - 'METHOD_ONLY': Có chữ ký hàm Java (nhưng chưa có class bao ngoài)
    - 'NL': Văn bản tự nhiên thuần (tiếng Việt hoặc tiếng Anh)
    """
    cleaned = text.strip()
    if not cleaned:
        return "NL"

    # Kiểm tra nếu có khai báo class
    if CLASS_DECL_RE.search(cleaned):
        return "FULL_JAVA"

    # Bỏ comment tạm thời để kiểm tra code
    code_without_comments = BLOCK_COMMENT_RE.sub("", cleaned)
    code_without_comments = LINE_COMMENT_RE.sub("", code_without_comments).strip()

    # Kiểm tra nếu có chữ ký hàm Java với các từ khóa đặc trưng
    has_method_keywords = bool(re.search(
        r"\b(?:public|private|protected|static|void|int|double|float|long|boolean|char|String|List<|Map<|Set<)\b",
        code_without_comments
    ))
    has_method_sig = bool(re.search(r"\b\w+\s*\([^)]*\)\s*(?:throws[^{]*)?(?:\{|$)", code_without_comments))

    if has_method_keywords and has_method_sig:
        return "METHOD_ONLY"

    return "NL"


# ==============================================================================
# 2. HIỆN THỰC HÓA CÁC QUY TẮC PROMPT ENGINEERING
# ==============================================================================

def format_cot_instruction(nl_text: str, enable_cot: bool = False) -> str:
    """RULE-14 (RULE-ZEROSHOT-COT):
    Định dạng các bước suy luận từng bước (Step 1, Step 2...) nếu có trong văn bản
    hoặc khi kích hoạt chế độ CoT.
    """
    text = nl_text.strip()
    if not text:
        return text

    # Kiểm tra nếu text có dạng liệt kê số (1. ... 2. ...)
    numbered_steps = re.findall(r"(?:^|\n)\s*(\d+[\.\)]\s*[^\n]+)", text)
    if numbered_steps and len(numbered_steps) >= 2:
        # Chuẩn hóa thành Step 1:, Step 2:
        step_lines = []
        for step in numbered_steps:
            cleaned_step = re.sub(r"^\d+[\.\)]\s*", "", step).strip()
            step_lines.append(cleaned_step)

        # Lấy phần mô tả chung (nếu có trước bước 1)
        first_step_match = re.search(r"\d+[\.\)]", text)
        intro = text[:first_step_match.start()].strip() if first_step_match else ""

        formatted_steps = []
        if intro:
            formatted_steps.append(intro)
        for i, s in enumerate(step_lines, 1):
            formatted_steps.append(f"Step {i}: {s}")

        return "\n".join(formatted_steps)

    # Nếu người dùng có các từ khóa chia bước tiếng Việt
    step_keywords = ["đầu tiên", "sau đó", "tiếp theo", "cuối cùng"]
    has_step_kw = sum(1 for kw in step_keywords if kw in text.lower()) >= 2
    if has_step_kw and enable_cot:
        # Giữ nguyên và thêm ghi chú suy luận
        return text + "\nLet's solve this step by step."

    return text


def format_instruction_as_docstring(instruction_text: str, indent: str = "    ") -> str:
    """RULE-1 (RULE-INCL-INSTR):
    Đóng gói toàn bộ câu lệnh/mô tả yêu cầu vào khối Javadoc comment chuẩn:
    /**
     * [Mô tả]
     */
    """
    text = instruction_text.strip()
    if not text:
        return ""

    # Chuẩn hóa nhiều dòng nếu có
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return ""

    doc_lines = [f"{indent}/**"]
    for l in lines:
        # Tránh lặp lại dấu * nếu user đã gõ kiểu javadoc
        cleaned = re.sub(r"^/\*+\s*|\s*\*+/\s*|^\*\s*", "", l).strip()
        if cleaned:
            doc_lines.append(f"{indent} * {cleaned}")
    doc_lines.append(f"{indent} */")

    return "\n".join(doc_lines)


def build_standard_boilerplate(class_name: str = "Problem") -> str:
    """RULE-3 (RULE-JAVA-DECL-NO-DOC):
    Tạo khối Imports chuẩn và mở khai báo `public class <class_name> {`.
    """
    return f"{STANDARD_IMPORTS_HEADER}\npublic class {class_name} {{\n"


def inject_language_keyword(prompt: str, enabled: bool = False) -> str:
    """RULE-5 (RULE-TRAINED-KW-JAVA):
    Chèn từ khóa chỉ định ngôn ngữ `// language: Java` ở đầu prompt nếu được kích hoạt.
    """
    if not enabled:
        return prompt

    if prompt.startswith("// language: Java"):
        return prompt

    return f"// language: Java\n{prompt}"


def align_instruction_before_signature(
    header_code: str,
    docstring: str,
    signature_or_priming: str
) -> str:
    """RULE-9 (RULE-REPEAT-INSTR-AT-END):
    Đặt Javadoc comment ở vị trí sát ngay phía trên chữ ký hàm (khoảng cách 0 dòng thừa),
    tối ưu hóa cơ chế Attention của mô hình.
    """
    parts = [header_code.rstrip()]
    if docstring:
        parts.append(docstring)
    parts.append(signature_or_priming)
    return "\n".join(parts)


def normalize_signature_syntax(sig_str: str, indent: str = "    ") -> str:
    """RULE-12 (RULE-CLEAR-SYNTAX-OUTPUT):
    Chuẩn hóa cú pháp chữ ký hàm:
    - Loại bỏ phần thân hàm thừa nếu có.
    - Đảm bảo kết thúc bằng đúng ` {\n`.
    - Chuẩn hóa thụt đầu dòng (indent).
    """
    sig = sig_str.strip()

    # Bỏ dấu mở ngoặc { ở cuối nếu có để chuẩn hóa lại
    if sig.endswith("{"):
        sig = sig[:-1].rstrip()

    # Nếu có phần thân hàm đằng sau {, cắt bỏ
    if "{" in sig:
        sig = sig.split("{", 1)[0].strip()

    # Chuẩn hóa khoảng trắng nội bộ
    sig_clean = re.sub(r"\s+", " ", sig)

    return f"{indent}{sig_clean} {{\n"


# ==============================================================================
# 3. TRÍCH XUẤT VÀ CHUẨN HÓA COMMENT TRONG MÃ NGUỒN JAVA
# ==============================================================================

def extract_comments_from_code(code: str) -> tuple[str, str]:
    """Tách comment và mã nguồn sạch ra khỏi đoạn code Java.
    Trả về: (extracted_comment_text, clean_code)
    """
    comments = []

    # 1. Trích xuất block comments (/** ... */ hoặc /* ... */)
    for m in BLOCK_COMMENT_RE.finditer(code):
        content = m.group(1).strip()
        lines = [re.sub(r"^\s*\*\s?", "", l).strip() for l in content.splitlines()]
        cleaned_block = "\n".join(l for l in lines if l)
        if cleaned_block:
            comments.append(cleaned_block)

    code_no_block = BLOCK_COMMENT_RE.sub("", code)

    # 2. Trích xuất line comments (// ...)
    line_comment_texts = []
    clean_lines = []
    for line in code_no_block.splitlines():
        trimmed = line.strip()
        if trimmed.startswith("//"):
            line_comment_texts.append(trimmed[2:].strip())
        else:
            clean_lines.append(line)

    if line_comment_texts:
        comments.append("\n".join(line_comment_texts))

    full_comment = "\n\n".join(c for c in comments if c).strip()
    clean_code = "\n".join(clean_lines).strip()

    return full_comment, clean_code


# ==============================================================================
# 4. HÀM ĐIỀU PHỐI CHÍNH (PROMPT ENHANCER ORCHESTRATOR)
# ==============================================================================

def enhance_prompt(
    user_input: str,
    enable_language_tag: bool = False,
    enable_cot: bool = False,
    default_class_name: str = "Problem"
) -> EnhancedPromptResult:
    """Tối ưu hóa và chuẩn hóa prompt người dùng theo 6 quy tắc ND4.
    
    Hỗ trợ mọi dạng đầu vào:
    1. NL (Ngôn ngữ tự nhiên): Bọc Javadoc + Imports + Class + mồi `public static `
    2. METHOD_ONLY (Chỉ có chữ ký hàm): Bổ sung Imports + bọc Class + di chuyển Comment sát hàm
    3. FULL_JAVA (Mã Java có class): Bổ sung Imports thiếu + chuẩn hóa comment và dấu {
    """
    raw = user_input.strip()
    if not raw:
        return {
            "prefix": "",
            "input_type": "NL",
            "is_enhanced": False,
            "raw_input": user_input
        }

    input_type = detect_input_type(raw)

    # ==========================================================================
    # NHÁNH 1: ĐẦU VÀO LÀ VĂN BẢN TỰ NHIÊN (NL)
    # ==========================================================================
    if input_type == "NL":
        # RULE-14: Chuẩn hóa bước CoT nếu có
        cot_text = format_cot_instruction(raw, enable_cot=enable_cot)

        # RULE-1: Đóng gói vào Javadoc comment
        docstring = format_instruction_as_docstring(cot_text, indent="    ")

        # RULE-3: Tạo block imports chuẩn và mở class Problem
        boilerplate = build_standard_boilerplate(class_name=default_class_name)

        # RULE-12: Mồi tiền tố `    public static ` để mô hình tự sinh tiếp signature + body
        priming_seed = "    public static "

        # RULE-9: Đặt Javadoc sát ngay phía trên tiền tố sinh mã
        full_prefix = align_instruction_before_signature(
            header_code=boilerplate,
            docstring=docstring,
            signature_or_priming=priming_seed
        )

        # RULE-5: Thêm tag ngôn ngữ nếu được kích hoạt
        final_prefix = inject_language_keyword(full_prefix, enabled=enable_language_tag)

        return {
            "prefix": final_prefix,
            "input_type": "NL",
            "is_enhanced": True,
            "raw_input": user_input
        }

    # ==========================================================================
    # NHÁNH 2: ĐẦU VÀO LÀ CHỮ KÝ HÀM (METHOD_ONLY)
    # ==========================================================================
    if input_type == "METHOD_ONLY":
        # Tách comment có sẵn và code sạch
        comment_text, clean_sig = extract_comments_from_code(raw)

        # RULE-1 & RULE-14: Chuẩn hóa comment thành Javadoc
        docstring = ""
        if comment_text:
            cot_text = format_cot_instruction(comment_text, enable_cot=enable_cot)
            docstring = format_instruction_as_docstring(cot_text, indent="    ")

        # RULE-3: Tạo block imports chuẩn và mở class Problem
        boilerplate = build_standard_boilerplate(class_name=default_class_name)

        # RULE-12: Chuẩn hóa chữ ký hàm và dấu mở ngoặc {
        normalized_sig = normalize_signature_syntax(clean_sig, indent="    ")

        # RULE-9: Đặt Javadoc sát ngay phía trên chữ ký hàm
        full_prefix = align_instruction_before_signature(
            header_code=boilerplate,
            docstring=docstring,
            signature_or_priming=normalized_sig
        )

        final_prefix = inject_language_keyword(full_prefix, enabled=enable_language_tag)

        return {
            "prefix": final_prefix,
            "input_type": "METHOD_ONLY",
            "is_enhanced": True,
            "raw_input": user_input
        }

    # ==========================================================================
    # NHÁNH 3: ĐẦU VÀO ĐÃ CÓ CLASS JAVA (FULL_JAVA)
    # ==========================================================================
    # Trích xuất comment, imports và thân class
    lines = raw.splitlines()
    existing_imports = []
    class_body_lines = []
    in_imports = True

    for line in lines:
        trimmed = line.strip()
        if in_imports and (trimmed.startswith("import ") or trimmed.startswith("package ")):
            existing_imports.append(line)
        elif trimmed:
            in_imports = False
            class_body_lines.append(line)
        else:
            if not in_imports:
                class_body_lines.append(line)

    # Bổ sung các imports chuẩn còn thiếu (RULE-3)
    combined_imports = list(existing_imports)
    existing_imports_set = {re.sub(r"\s+", " ", imp.strip()) for imp in existing_imports}
    for std_imp in STANDARD_IMPORTS:
        clean_std = re.sub(r"\s+", " ", std_imp.strip())
        if clean_std not in existing_imports_set:
            combined_imports.append(std_imp)

    header = "\n".join(combined_imports) + "\n\n" if combined_imports else ""
    body_code = "\n".join(class_body_lines)

    # Kiểm tra và đảm bảo dấu mở ngoặc ở cuối (RULE-12)
    body_code_stripped = body_code.rstrip()
    if not body_code_stripped.endswith("{") and not body_code_stripped.endswith("}"):
        # Nếu dòng cuối cùng là chữ ký hàm chưa mở ngoặc
        body_code_stripped += " {"

    full_prefix = header + body_code_stripped + "\n"
    final_prefix = inject_language_keyword(full_prefix, enabled=enable_language_tag)

    return {
        "prefix": final_prefix,
        "input_type": "FULL_JAVA",
        "is_enhanced": True,
        "raw_input": user_input
    }
