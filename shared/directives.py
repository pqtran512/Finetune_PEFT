"""Shared prompt directives and stop sequences for Java LLM code generation.
"""

# Directive được chèn vào trên đầu mỗi prompt dưới dạng comment Java.
# Giữ ở dạng comment để mô hình coi như ngữ cảnh tự nhiên thay vì instruction tag.
# CodeLlama và Qwen dùng chung một directive.
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
