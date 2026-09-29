"""Shared Java completion post-processing and cleaning utilities.
"""

import re


def _extract_all_methods(text: str) -> dict:
    """Trích xuất tất cả method definitions hoàn chỉnh từ text, trả về dict {name: code}."""
    methods = {}
    sig_re = re.compile(
        r"^[ \t]*(?:(?:public|private|protected|static|final)\s+)*"
        r"(\w+(?:<[^>]*>)?(?:\[\])*)\s+(\w+)\s*\(",
        re.MULTILINE,
    )
    for m in sig_re.finditer(text):
        m_name = m.group(2)
        if m_name in ("if", "for", "while", "switch", "catch", "synchronized"):
            continue
        start_idx = m.start()
        brace_start = text.find("{", start_idx)
        if brace_start == -1:
            continue
        cnt = 0
        end_idx = None
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                cnt += 1
            elif text[i] == "}":
                cnt -= 1
                if cnt == 0:
                    end_idx = i + 1
                    break
        if end_idx is not None and m_name not in methods:
            methods[m_name] = text[start_idx:end_idx]
    return methods


def _find_needed_helpers(method_body: str, remaining_code: str) -> str:
    """Quét các method calls trong method_body, nếu xuất hiện trong remaining_code thì giữ lại."""
    available = _extract_all_methods(remaining_code)
    if not available:
        return ""
    needed = []
    visited = set()
    code_to_scan = method_body
    found = True
    while found:
        found = False
        calls = set(re.findall(r"\b([a-zA-Z_]\w*)\s*\(", code_to_scan))
        code_to_scan = ""
        for name in calls:
            if name not in visited and name in available:
                visited.add(name)
                needed.append(name)
                code_to_scan += "\n" + available[name]
                found = True
    return "\n".join(available[n] for n in needed) if needed else ""


def clean_java_completion(completion: str) -> str:
    """Cắt completion thành body method chính + helper methods cần thiết."""
    patterns = [r"public\s+class\s+Main", r"class\s+Main", r"public\s+static\s+void\s+main"]
    for p in patterns:
        m = re.search(p, completion)
        if m:
            completion = completion[: m.start()]

    lines = completion.split("\n")
    body_lines = []
    bracket_count = 0
    cutoff = len(lines)
    for i, line in enumerate(lines):
        bracket_count += line.count("{") - line.count("}")
        if bracket_count < 0:
            cutoff = i
            break
        body_lines.append(line)

    method_body = "\n".join(body_lines).strip()
    remaining = "\n".join(lines[cutoff:])
    helpers_code = _find_needed_helpers(method_body, remaining)

    if helpers_code:
        final = method_body + "\n    }\n" + helpers_code
        while final.count("{") < final.count("}") and final.endswith("}"):
            idx = final.rfind("}")
            final = (final[:idx] + final[idx + 1 :]).rstrip()
    else:
        final = method_body

    # Convert default/package-private static methods to public static to prevent test access/reflection errors
    pattern = r'(?<!public\s)(?<!private\s)(?<!protected\s)\bstatic\s+([\w<>, ?[\]]+)\s+(\w+)\s*\('
    final = re.sub(pattern, r'public static \1 \2(', final)

    return final


def clean_helper_output(gen_text: str) -> str:
    """Strip code fences and cut at end-of-helpers markers for repair output."""
    if "```java" in gen_text:
        gen_text = gen_text.split("```java", 1)[1].split("```", 1)[0]
    elif "```" in gen_text:
        gen_text = gen_text.split("```", 1)[1].split("```", 1)[0]
    cut_markers = [
        "\npublic class ",
        "\nclass ",
        "\n// Test",
        "\npublic static void main",
        "\nimport ",
        "\npackage ",
        "\n// ===",
    ]
    for m in cut_markers:
        i = gen_text.find(m)
        if i != -1:
            gen_text = gen_text[:i]
    return gen_text.rstrip()


def clean_output(gen_text: str) -> str:
    """Lọc lấy code Java, dừng lại nếu thấy định nghĩa class mới hoặc EOF."""
    if "```java" in gen_text:
        gen_text = gen_text.split("```java", 1)[1].split("```", 1)[0]
    elif "```" in gen_text:
        gen_text = gen_text.split("```", 1)[1].split("```", 1)[0]
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
