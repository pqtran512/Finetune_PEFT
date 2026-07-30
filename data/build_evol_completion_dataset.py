"""Build method-body completion JSONL from Evol + Magicoder sources (Java).

Sources:
  - nickrosh/Evol-Instruct-Code-80k-v1
  - ise-uiuc/Magicoder-Evol-Instruct-110K
  - ise-uiuc/Magicoder-OSS-Instruct-75K

Task format (generic — not HumanEval-specific):
  prefix = optional // docs + method signature + "{"
  target = method body + closing "}"

All sources are remapped to the same completion schema (not left as chat/INST).
Only the first parseable method in each code block is kept (safer context).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

from datasets import load_dataset


def load_env(path: Path) -> None:
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


DATA_DIR = Path(__file__).resolve().parent
REPO_ROOT = DATA_DIR.parent
load_env(DATA_DIR / ".env")
load_env(REPO_ROOT / ".env")

CACHE_DIR = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
DEFAULT_OUT = DATA_DIR / "jsonl" / "evol_java_completion_train.jsonl"

JAVA_TAG = re.compile(r"```java\b", re.IGNORECASE)
JAVA_MARKERS = re.compile(
    r"(import\s+java\.|import\s+javax\.|System\.out\.|@Override|"
    r"throws\s+(?:Exception|IOException|RuntimeException)|"
    r"new\s+ArrayList<|new\s+HashMap<|new\s+LinkedList<|"
    r"public\s+static\s+void\s+main\s*\(\s*String)"
)
CSHARP_SIGNALS = re.compile(
    r"(\busing\s+System\b|Console\.Write|\bnamespace\s+\w+\s*\{|"
    r"\bvar\s+\w+\s*=|=>|\bIService\w+|\bClaimsIdentity\b|\bMock<|"
    r"\bawait\s+\w+|\basync\s+Task|\b[A-Z]\w+\.cs\b)"
)
NON_JAVA_SIGNALS = re.compile(
    r"(\bfun\s+\w+\s*\(|\bval\s+\w+\s*=|\blet\s+\w+\s*=|"
    r"\bdef\s+\w+\s*\(|\bfunction\s+\w+\s*\(|"
    r"\bconst\s+\w+\s*=|\bprintln!\(|\bfmt\.Print)"
)

# Generic method signature: ≥1 modifier so we skip if/for/while.
METHOD_RE = re.compile(
    r"(/\*\*.*?\*/\s*)?"
    r"((?:public|private|protected|static)"
    r"(?:\s+(?:public|private|protected|static|final|abstract|synchronized|native))*"
    r"\s+[\w<>,\[\]\?\s]+?"
    r"\s+\w+\s*\([^)]*\)\s*"
    r"(?:throws[^{]*)?\s*\{)",
    re.DOTALL,
)


def is_real_java(text: str) -> bool:
    if not text:
        return False
    if "javascript" in text.lower():
        return False
    if CSHARP_SIGNALS.search(text):
        return False
    if NON_JAVA_SIGNALS.search(text):
        return False
    return bool(JAVA_TAG.search(text) or JAVA_MARKERS.search(text))


def clean_backticks(text: str) -> str:
    text = re.sub(r"```java\b", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")
    return text.strip()


def extract_java_code(text: str) -> str | None:
    if not text:
        return None
    m = re.search(r"```java\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        code = m.group(1)
        if CSHARP_SIGNALS.search(code) or NON_JAVA_SIGNALS.search(code):
            return None
        return code.strip()
    
    cleaned = clean_backticks(text)
    if JAVA_MARKERS.search(cleaned) and not CSHARP_SIGNALS.search(cleaned):
        return cleaned
    return None


def instruction_to_line_comments(instruction: str, indent: str = "    ") -> str:
    """Turn natural-language instruction into generic line comments above a method."""
    lines = []
    for raw in instruction.strip().splitlines():
        cleaned = raw.strip()
        if not cleaned:
            continue
        # Keep short; Evol instructions can be long.
        if len(cleaned) > 160:
            cleaned = cleaned[:157] + "..."
        lines.append(f"{indent}// {cleaned}")
        if len(lines) >= 8:
            break
    return "\n".join(lines)


def javadoc_to_line_comments(jd: str, indent: str = "    ") -> str:
    if not jd:
        return ""
    text = jd.strip()
    text = re.sub(r"^/\*+", "", text)
    text = re.sub(r"\*+/\s*$", "", text)
    out = []
    for line in text.splitlines():
        cleaned = re.sub(r"^\s*\*\s?", "", line).strip()
        if not cleaned or cleaned.startswith("@"):
            continue
        out.append(f"{indent}// {cleaned}")
    return "\n".join(out)


def normalize_body_indent(body: str, target_indent: int = 8) -> str:
    lines = body.expandtabs(4).split("\n")
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return body
    min_indent = min(len(l) - len(l.lstrip(" ")) for l in non_empty)
    target = " " * target_indent
    out = []
    for l in lines:
        if not l.strip():
            out.append("")
        else:
            out.append(target + l[min_indent:])
    return "\n".join(out)


def match_method_body(java_src: str, brace_open: int) -> tuple[str, int] | None:
    """Return (body_without_braces, index_after_closing_brace) or None."""
    depth, i = 1, brace_open + 1
    while i < len(java_src) and depth > 0:
        ch = java_src[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    body = java_src[brace_open + 1 : i - 1].rstrip()
    return body, i


CALL_RE = re.compile(r"\b([a-zA-Z_]\w*)\s*\(")
_STRIP_LINE_COMMENT = re.compile(r"//[^\n]*")
_STRIP_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_STRIP_STRING_LIT = re.compile(r'"(?:\\.|[^"\\\n])*"')
SKIP_NAMES = {"main", "if", "for", "while", "switch", "catch", "try", "new",
              "return", "class", "else", "do", "synchronized"}


def _strip_noise(src: str) -> str:
    src = _STRIP_LINE_COMMENT.sub("", src)
    src = _STRIP_BLOCK_COMMENT.sub("", src)
    return _STRIP_STRING_LIT.sub('""', src)


def extract_all_methods_in_src(java_src: str, exclude_name: str) -> dict[str, str]:
    """Find all method definitions in src, return {name: full_code_str}."""
    sig_re = re.compile(
        r"^[ \t]*(?:(?:public|private|protected)\s+)*"
        r"(?:(?:static|final|abstract|synchronized|native)\s+)*"
        r"(?:\w[\w<>,\s\[\]\?]*?)\s+"
        r"(\w+)\s*\(",
        re.MULTILINE,
    )
    methods = {}
    for m in sig_re.finditer(java_src):
        name = m.group(1)
        if name in SKIP_NAMES or name == exclude_name:
            continue
        rest = java_src[m.end():]
        brace_idx = rest.find("{")
        if brace_idx == -1 or brace_idx > 200:
            continue
        abs_brace = m.end() + brace_idx
        line_start = java_src.rfind("\n", 0, m.start())
        line_start = 0 if line_start == -1 else line_start + 1
        depth, end = 0, None
        for j in range(abs_brace, len(java_src)):
            if java_src[j] == "{":
                depth += 1
            elif java_src[j] == "}":
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break
        if end and name not in methods:
            methods[name] = java_src[line_start:end]
    return methods


def collect_needed_helpers(java_src: str, body: str, own_name: str) -> list[str]:
    """Return list of helper method codes (transitively reachable from body),
    each re-indented to 4-space base (method-level inside class)."""
    methods = extract_all_methods_in_src(java_src, exclude_name=own_name)
    if not methods:
        return []
    needed_names = []
    visited = set()
    queue = list(set(CALL_RE.findall(_strip_noise(body))) & set(methods.keys()))
    while queue:
        name = queue.pop()
        if name in visited:
            continue
        visited.add(name)
        needed_names.append(name)
        sub_calls = set(CALL_RE.findall(_strip_noise(methods[name])))
        queue.extend((sub_calls & set(methods.keys())) - visited)
    out = []
    for n in needed_names:
        code = methods[n]
        lines = code.expandtabs(4).split("\n")
        non_empty = [l for l in lines if l.strip()]
        if not non_empty:
            continue
        min_ind = min(len(l) - len(l.lstrip(" ")) for l in non_empty)
        re_indented = "\n".join(
            ("    " + l[min_ind:]) if l.strip() else "" for l in lines
        )
        out.append(re_indented)
    return out


def to_completion_sample(
    java_src: str,
    *,
    instruction: str = "",
    min_body: int = 20,
    max_body: int = 2000,
) -> dict[str, str] | None:
    """Map Java source to generic method-body completion {prefix, target}."""
    if not java_src:
        return None
    m = METHOD_RE.search(java_src)
    if not m:
        return None
    sig = m.group(2)
    if re.search(r"\(\s*this\s+\w", sig):
        return None

    brace_open = m.end() - 1  # index of '{'
    matched = match_method_body(java_src, brace_open)
    if matched is None:
        return None
    body, _ = matched
    if len(body) < min_body or len(body) > max_body:
        return None
    if CSHARP_SIGNALS.search(body) or NON_JAVA_SIGNALS.search(body):
        return None

    own_name_m = re.search(r"\b(\w+)\s*\(", sig)
    own_name = own_name_m.group(1) if own_name_m else ""
    helpers = collect_needed_helpers(java_src, body, own_name)

    inline_doc = (m.group(1) or "").strip()
    doc = javadoc_to_line_comments(inline_doc)
    if not doc and instruction:
        doc = instruction_to_line_comments(instruction)

    sig_compact = re.sub(r"\s+", " ", sig.strip())
    # Drop trailing "{" from compact sig; we add it explicitly.
    if sig_compact.endswith("{"):
        sig_compact = sig_compact[:-1].rstrip()

    header = (
        "import java.util.*;\n"
        "import java.lang.reflect.*;\n"
        "import org.javatuples.*;\n"
        "import java.security.*;\n"
        "import java.math.*;\n"
        "import java.io.*;\n"
        "import java.util.stream.*;\n"
        "class Problem {\n"
    )
    sig_line = f"    {sig_compact} {{\n"

    prefix_parts = [header]
    if doc:
        prefix_parts.append(doc + "\n")
    prefix_parts.append(sig_line)
    prefix = "".join(prefix_parts)

    body_n = normalize_body_indent(body, target_indent=8)
    target = body_n + "\n    }\n"
    if helpers:
        target += "\n" + "\n\n".join(helpers) + "\n"
    return {"prefix": prefix, "target": target}


def load_he_method_keywords() -> set[str]:
    """Optional decontam keywords (hygiene only — does not shape the format)."""
    he = load_dataset(
        "nuprl/MultiPL-E", "humaneval-java", split="test", cache_dir=CACHE_DIR
    )
    keywords: set[str] = set()
    for name in he["name"]:
        parts = name.split("_", 2)
        if len(parts) >= 3:
            keywords.add(parts[2].replace("_", "").lower())
    return {k for k in keywords if len(k) > 6}


def has_leak(prefix: str, target: str, keywords: set[str]) -> bool:
    blob = (prefix + target).lower().replace("_", "")
    return any(k in blob for k in keywords)


def _looks_java(instruction: str, code_blob: str) -> bool:
    blob = instruction + "\n" + code_blob
    return (
        "java" in instruction.lower()
        or "java" in code_blob.lower()
        or is_real_java(blob)
    )


def _map_row(instruction: str, code_text: str) -> dict[str, str] | None:
    """Extract Java + map to prefix/target method-body completion."""
    blob = instruction + "\n" + code_text
    code = (
        extract_java_code(code_text)
        or extract_java_code(blob)
        or (code_text if is_real_java(code_text) else None)
    )
    if not code:
        return None
    return to_completion_sample(code, instruction=instruction)


def collect_evol() -> list[dict[str, str]]:
    print("--- nickrosh/Evol-Instruct-Code-80k-v1 ---")
    ds = load_dataset(
        "nickrosh/Evol-Instruct-Code-80k-v1", split="train", cache_dir=CACHE_DIR
    )
    print(f"  Raw rows: {len(ds)}")
    java_rows = 0
    samples: list[dict[str, str]] = []
    for ex in ds:
        instruction = ex.get("instruction", "") or ""
        output = ex.get("output", "") or ""
        if not _looks_java(instruction, output):
            continue
        java_rows += 1
        out = _map_row(instruction, output)
        if out:
            samples.append(out)
    print(f"  Java-filtered: {java_rows}")
    print(f"  Mapped completion: {len(samples)}")
    return samples


def collect_magicoder_evol() -> list[dict[str, str]]:
    print("--- ise-uiuc/Magicoder-Evol-Instruct-110K ---")
    try:
        ds = load_dataset(
            "ise-uiuc/Magicoder-Evol-Instruct-110K",
            split="train",
            cache_dir=CACHE_DIR,
        )
    except Exception as e:
        print(f"  Skip Magicoder-Evol: {e}")
        return []
    print(f"  Raw rows: {len(ds)}")
    java_rows = 0
    samples: list[dict[str, str]] = []
    for ex in ds:
        instruction = ex.get("instruction", "") or ""
        response = ex.get("response", "") or ex.get("output", "") or ""
        if not _looks_java(instruction, response):
            continue
        java_rows += 1
        out = _map_row(instruction, response)
        if out:
            samples.append(out)
    print(f"  Java-filtered: {java_rows}")
    print(f"  Mapped completion: {len(samples)}")
    return samples


def collect_magicoder_oss() -> list[dict[str, str]]:
    print("--- ise-uiuc/Magicoder-OSS-Instruct-75K ---")
    try:
        ds = load_dataset(
            "ise-uiuc/Magicoder-OSS-Instruct-75K",
            split="train",
            cache_dir=CACHE_DIR,
        )
    except Exception as e:
        print(f"  Skip Magicoder-OSS: {e}")
        return []
    print(f"  Raw rows: {len(ds)}")
    java_rows = 0
    samples: list[dict[str, str]] = []
    for ex in ds:
        problem = ex.get("problem", "") or ""
        solution = ex.get("solution", "") or ""
        if not _looks_java(problem, solution):
            continue
        java_rows += 1
        out = _map_row(problem, solution)
        if out:
            samples.append(out)
    print(f"  Java-filtered: {java_rows}")
    print(f"  Mapped completion: {len(samples)}")
    return samples


def build_samples(*, decontaminate: bool) -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    samples += collect_evol()
    samples += collect_magicoder_evol()
    samples += collect_magicoder_oss()
    print(f"\nTotal mapped (before dedup): {len(samples)}")

    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, str]] = []
    for s in samples:
        key = (s["prefix"][:200], s["target"][:200])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)
    print(f"After dedup: {len(deduped)}")

    if decontaminate:
        print("Decontaminate vs MultiPL-E humaneval-java method-name keywords...")
        he_kw = load_he_method_keywords()
        before = len(deduped)
        deduped = [s for s in deduped if not has_leak(s["prefix"], s["target"], he_kw)]
        print(f"Removed leaks: {before - len(deduped)}; keep {len(deduped)}")

    random.seed(42)
    random.shuffle(deduped)
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evol + Magicoder-Evol/OSS Java → generic method-body completion JSONL"
        )
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output jsonl path",
    )
    parser.add_argument(
        "--no-decontam",
        action="store_true",
        help="Skip HumanEval-Java keyword decontamination",
    )
    args = parser.parse_args()

    samples = build_samples(decontaminate=not args.no_decontam)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(samples)} → {args.out}")
    if samples:
        print("\n--- sample PREFIX ---")
        print(samples[0]["prefix"][:500])
        print("--- sample TARGET ---")
        print(samples[0]["target"][:500])


if __name__ == "__main__":
    main()
