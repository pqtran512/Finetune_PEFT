import json
import os
import re
import random
from pathlib import Path


def load_env_file(env_path: Path):
    """Load KEY=VALUE pairs từ .env vào os.environ (không ghi đè biến đã có)."""
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# Load .env trước khi import datasets/huggingface_hub
load_env_file(Path(__file__).parent / ".env")

# Login HF nếu có token
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
if hf_token:
    from huggingface_hub import login
    login(token=hf_token, add_to_git_credential=False)
    print("✅ HF login OK")
else:
    print("⚠️  Không thấy HF_TOKEN trong .env — gated datasets sẽ skip")

from datasets import load_dataset

CACHE_DIR = "D:/cache/hugging_face"
OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUTPUT_DIR / "java_completion_train.jsonl"

random.seed(42)

# Whitelist patterns rất Java-specific
JAVA_TAG = re.compile(r"```java\b", re.IGNORECASE)
JAVA_MARKERS = re.compile(
    r"(import\s+java\.|import\s+javax\.|System\.out\.|@Override|"
    r"throws\s+(?:Exception|IOException|RuntimeException)|"
    r"new\s+ArrayList<|new\s+HashMap<|new\s+LinkedList<|"
    r"public\s+static\s+void\s+main\s*\(\s*String)"
)

# Blacklist: signals từ C#, Kotlin, JS, TypeScript
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


def is_real_java(text: str) -> bool:
    """Strict check: phải có Java marker, không có C#/Kotlin/JS marker."""
    if not text:
        return False
    if "javascript" in text.lower():
        return False
    if CSHARP_SIGNALS.search(text):
        return False
    if NON_JAVA_SIGNALS.search(text):
        return False
    return bool(JAVA_TAG.search(text) or JAVA_MARKERS.search(text))


METHOD_RE = re.compile(
    r"(/\*\*.*?\*/\s*)?"
    r"(public\s+static\s+\S+\s+\w+\s*\([^)]*\)\s*(?:throws[^{]*)?\{)",
    re.DOTALL,
)


def javadoc_to_line_comments(jd: str) -> str:
    """Convert /** ... */ Javadoc or raw doc text into HumanEval-Java-style // lines."""
    if not jd:
        return ""
    text = jd.strip()
    text = re.sub(r"^/\*+", "", text)
    text = re.sub(r"\*+/\s*$", "", text)
    out = []
    for line in text.splitlines():
        cleaned = re.sub(r"^\s*\*\s?", "", line).strip()
        if not cleaned:
            continue
        if cleaned.startswith("@"):  # drop @param/@return/@throws/@see/...
            continue
        out.append(cleaned)
    if not out:
        return ""
    return "\n".join(f"    // {l}" for l in out)


def normalize_body_indent(body: str, target_indent: int = 8) -> str:
    """Re-indent body so non-empty lines start at exactly target_indent spaces.
    Eliminates train/inference indent mismatch that causes the \\n    }\\n stop
    to fire on inner block closes when source code uses 2-space indentation."""
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


METHOD_DEF_FILTER_RE = re.compile(
    r"^\s*(?:(?:public|private|protected)\s+)?"
    r"(?:(?:static|final|abstract|synchronized|native)\s+)*"
    r"(?:\w[\w<>,\s\[\]\?]*?)\s+"   # return type
    r"(\w+)\s*\(",
    re.MULTILINE,
)
CALL_RE = re.compile(r"\b([a-zA-Z_]\w*)\s*\(")
_STRIP_LINE_COMMENT = re.compile(r"//[^\n]*")
_STRIP_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_STRIP_STRING_LIT = re.compile(r'"(?:\\.|[^"\\\n])*"')


def _strip_noise(src: str) -> str:
    src = _STRIP_LINE_COMMENT.sub("", src)
    src = _STRIP_BLOCK_COMMENT.sub("", src)
    return _STRIP_STRING_LIT.sub('""', src)


def calls_external_helper(java_src: str, body: str, own_name: str) -> bool:
    """True if body calls a method defined in java_src other than own_name.
    Drives the Path-1 filter: skip samples whose extracted method depends on
    sibling helpers, so the fine-tuned model learns to inline everything."""
    defined = set(METHOD_DEF_FILTER_RE.findall(_strip_noise(java_src)))
    defined.discard(own_name)
    if not defined:
        return False
    called = set(CALL_RE.findall(_strip_noise(body)))
    return bool(called & defined)


def extract_method_completion(java_src: str, external_doc: str = ""):
    if not java_src:
        return None
    m = METHOD_RE.search(java_src)
    if not m:
        return None
    sig = m.group(2)
    # Reject C# extension methods (first arg starts with 'this ')
    if re.search(r"\(\s*this\s+\w", sig):
        return None
    sig_end = m.end()
    depth, i = 1, sig_end
    while i < len(java_src) and depth > 0:
        ch = java_src[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    body = java_src[sig_end : i - 1].rstrip()
    if len(body) < 20 or len(body) > 2000:
        return None

    # Re-validate body chỉ là Java
    if CSHARP_SIGNALS.search(body) or NON_JAVA_SIGNALS.search(body):
        return None

    # Path 1 filter: drop samples that call sibling helpers defined elsewhere
    # in the same source. Forces the model to learn inlined solutions.
    own_name_m = re.search(r"\b(\w+)\s*\(", sig)
    own_name = own_name_m.group(1) if own_name_m else ""
    if calls_external_helper(java_src, body, own_name):
        return None

    # Normalize body to 8-space base indent so the method-close stop string
    # `\n    }\n` cannot be confused with inner-block closes (root cause of
    # truncated bodies / missing-return failures).
    body = normalize_body_indent(body, target_indent=8)

    inline_doc = (m.group(1) or "").strip()
    doc_block = javadoc_to_line_comments(inline_doc) or javadoc_to_line_comments(external_doc)

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
    sig_line = f"    {sig}\n"
    prefix = header + (doc_block + "\n" if doc_block else "") + sig_line
    target = body + "\n    }\n"
    return {"prefix": prefix, "target": target}


def extract_from_markdown(text: str):
    """Lấy block code Java đầu tiên trong text markdown."""
    m = re.search(r"```java\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    code = m.group(1)
    if CSHARP_SIGNALS.search(code) or NON_JAVA_SIGNALS.search(code):
        return None
    return extract_method_completion(code)


def load_he_method_keywords():
    he = load_dataset("nuprl/MultiPL-E", "humaneval-java", split="test", cache_dir=CACHE_DIR)
    keywords = set()
    for name in he["name"]:
        parts = name.split("_", 2)
        if len(parts) >= 3:
            keywords.add(parts[2].replace("_", "").lower())
    return {k for k in keywords if len(k) > 6}


def has_leak(prefix: str, target: str, keywords: set[str]) -> bool:
    blob = (prefix + target).lower().replace("_", "")
    return any(k in blob for k in keywords)


def collect_code_search_net(target_n=30000):
    """code_search_net: function-level Java with docstrings (public, no auth)."""
    print("--- code_search_net (java) ---")
    samples = []
    ds = None
    for repo in ["code-search-net/code_search_net", "code_search_net"]:
        try:
            ds = load_dataset(
                repo, "java", split="train", cache_dir=CACHE_DIR, trust_remote_code=True
            )
            break
        except Exception as e:
            print(f"  Try {repo}: {e}")
    if ds is None:
        print("  Skip code_search_net")
        return samples

    for ex in ds:
        if len(samples) >= target_n:
            break
        code = ex.get("func_code_string") or ex.get("whole_func_string") or ""
        external_doc = ex.get("func_documentation_string") or ""
        if not is_real_java(code):
            # Vẫn thử extract nếu code trông giống Java function thuần
            if not re.search(r"\bpublic\s+static\b", code):
                continue
        out = extract_method_completion(code, external_doc=external_doc)
        if out:
            samples.append(out)
    print(f"  collected: {len(samples)}")
    return samples


def collect_magicoder(target_n=8000):
    print("--- Magicoder-OSS-Instruct-75K (Java) ---")
    samples = []
    try:
        ds = load_dataset(
            "ise-uiuc/Magicoder-OSS-Instruct-75K",
            split="train",
            cache_dir=CACHE_DIR,
        )
    except Exception as e:
        print(f"  Skip Magicoder: {e}")
        return samples
    for ex in ds:
        if len(samples) >= target_n:
            break
        problem = ex.get("problem", "") or ""
        solution = ex.get("solution", "") or ""
        text = problem + "\n" + solution
        if not is_real_java(text):
            continue
        out = extract_from_markdown(solution) or extract_from_markdown(text) or extract_method_completion(solution)
        if out:
            samples.append(out)
    print(f"  collected: {len(samples)}")
    return samples


def collect_evol(target_n=5000):
    print("--- Evol-Instruct (filtered Java) ---")
    samples = []
    try:
        ds = load_dataset(
            "nickrosh/Evol-Instruct-Code-80k-v1",
            split="train",
            cache_dir=CACHE_DIR,
        )
    except Exception as e:
        print(f"  Skip Evol: {e}")
        return samples
    for ex in ds:
        if len(samples) >= target_n:
            break
        instruction = ex.get("instruction", "") or ""
        output = ex.get("output", "") or ""
        text = instruction + "\n" + output
        if not is_real_java(text):
            continue
        out = extract_from_markdown(output) or extract_from_markdown(text) or extract_method_completion(output)
        if out:
            samples.append(out)
    print(f"  collected: {len(samples)}")
    return samples


def collect_the_stack_smol(target_n=20000):
    """bigcode/the-stack-smol Java (gated — cần HF token + accept license)."""
    print("--- the-stack-smol (java) ---")
    samples = []
    try:
        ds = load_dataset(
            "bigcode/the-stack-smol",
            data_dir="data/java",
            split="train",
            cache_dir=CACHE_DIR,
        )
    except Exception as e:
        print(f"  Skip the-stack-smol: {e}")
        return samples
    for ex in ds:
        if len(samples) >= target_n:
            break
        src = ex.get("content", "") or ""
        if len(src) > 50000:
            continue
        if not is_real_java(src):
            continue
        out = extract_method_completion(src)
        if out:
            samples.append(out)
    print(f"  collected: {len(samples)}")
    return samples


def collect_codeparrot_github(target_n=20000):
    """codeparrot/github-code Java subset, dùng streaming (public)."""
    print("--- codeparrot/github-code (Java, streaming) ---")
    samples = []
    try:
        ds = load_dataset(
            "codeparrot/github-code",
            "Java-all",
            split="train",
            streaming=True,
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"  Skip github-code: {e}")
        return samples
    seen = 0
    for ex in ds:
        seen += 1
        if seen > 200000:
            break
        if len(samples) >= target_n:
            break
        code = ex.get("code", "") or ""
        if len(code) > 50000:
            continue
        if not is_real_java(code):
            continue
        out = extract_method_completion(code)
        if out:
            samples.append(out)
    print(f"  collected: {len(samples)} (scanned {seen})")
    return samples


def main():
    print("=== Build Java function-completion dataset ===")
    print(f"Output: {OUT_PATH}")

    print("\nLoading HumanEval-Java keywords for decontamination...")
    he_keywords = load_he_method_keywords()
    print(f"  {len(he_keywords)} keywords")

    all_samples = []
    all_samples += collect_code_search_net(target_n=25000)
    all_samples += collect_the_stack_smol(target_n=20000)
    all_samples += collect_codeparrot_github(target_n=10000)
    all_samples += collect_magicoder(target_n=8000)
    all_samples += collect_evol(target_n=5000)

    print(f"\nTotal raw: {len(all_samples)}")

    # Dedup
    seen = set()
    deduped = []
    for s in all_samples:
        key = (s["prefix"][:200], s["target"][:200])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)
    print(f"After dedup: {len(deduped)}")

    # Decontamination
    clean = [s for s in deduped if not has_leak(s["prefix"], s["target"], he_keywords)]
    leaked = len(deduped) - len(clean)
    print(f"Removed leaks: {leaked}")
    print(f"Final: {len(clean)}")

    random.shuffle(clean)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for s in clean:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(clean)} samples -> {OUT_PATH}")
    if clean:
        print("\n--- Sample 0 ---")
        print("PREFIX:", clean[0]["prefix"][:400])
        print("TARGET:", clean[0]["target"][:400])


if __name__ == "__main__":
    main()
