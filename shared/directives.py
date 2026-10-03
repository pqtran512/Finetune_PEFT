"""Shared prompt directives and stop sequences for Java LLM code generation.
"""

# Directive được chèn vào trên đầu mỗi prompt dưới dạng comment Java.
# Giữ ở dạng comment để mô hình coi như ngữ cảnh tự nhiên thay vì instruction tag.
#
# Cả CodeLlama và CodeQwen đều dừng tại "\n    }\n" (stop chuẩn của MultiPL-E,
# đóng method). Rule "viết helper sau dấu }" vì thế không được sinh ra ở cả hai.
# Directive CodeLlama bỏ rule đó và yêu cầu viết hết logic trong thân hàm.
PROMPT_DIRECTIVE = """// Implementation rules (must follow):
// 1. The body must be self-contained. Do NOT call any method that you do not
//    define in this same file.
// 2. If you need a helper, define it inside the same class, AFTER the closing
//    brace of the target method, as a private static method.
// 3. Do NOT rely on external libraries beyond java.util.*, java.util.stream.*,
//    java.util.regex.*, java.lang.Math, and org.javatuples.Pair (which exposes
//    getValue0()/getValue1(), not getFirst()/getSecond()).
// 4. Return the completed method body only; do NOT redeclare the class or main.
"""

QWEN_PROMPT_DIRECTIVE = PROMPT_DIRECTIVE

# Khớp với stop-at-method-close: mọi logic phải nằm trong thân hàm đang viết.
CODELLAMA_PROMPT_DIRECTIVE = """// Implementation rules:
// 1. Write the full logic inside this method. Do not call any method that is not already defined above this line.
// 2. Use only java.util.*, java.util.stream.*, java.util.regex.*, java.lang.Math, and org.javatuples.Pair (getValue0()/getValue1()).
// 3. Do not redeclare the class or main.
"""

REPAIR_INSTRUCTION = (
    "// The Java code below failed to compile because it calls helper methods\n"
    "// that are not defined. Define ONLY the missing helpers listed below as\n"
    "// private static methods. Output ONLY the helper method bodies — no\n"
    "// class declaration, no import, no main, no markdown fences. Use the\n"
    "// same indentation as the existing code (4 spaces).\n"
)

STOP_STRINGS = [
    "\n    }\n",
    "\n}\n",
    "\npublic static void main",
    "\n```",
]


def apply_directive(prompt: str, directive: str = PROMPT_DIRECTIVE) -> str:
    """Prepend a prompt directive to the input prompt."""
    return directive + "\n" + prompt
