"""Plan B — Data rebuild đầy đủ.

Khác với baseline:
- METHOD_RE loosened: chấp nhận mọi visibility + optional static + optional final.
  Yêu cầu ít nhất 1 modifier để loại trừ control-flow constructs.
- Quality filter: drop mẫu có annotation lạ (@Nullable...), external class refs,
  hoặc indent jump > 8 spaces giữa 2 dòng liên tiếp.
- Bỏ filter "inline everything", thay bằng append helpers vào target.
- Bỏ codeparrot/github-code (datasets v2.17+ block loading scripts).
- Thêm nguồn algorithmic: ise-uiuc/Magicoder-Evol-Instruct-110K.
- code_search_net bump cap 25k → 50k (sau khi quality filter active).

Nguồn (train): code_search_net, the-stack-smol, Magicoder-OSS, Magicoder-Evol,
  Evol-Instruct. Decontaminate: MultiPL-E humaneval-java.
"""
import json
import os
import re
import random
from pathlib import Path


def load_env_file(env_path: Path):
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


load_env_file(Path(__file__).parent / ".env")

hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
if hf_token:
    from huggingface_hub import login
    login(token=hf_token, add_to_git_credential=False)
    print("HF login OK")
else:
    print("Không thấy HF_TOKEN — gated datasets sẽ skip")

from datasets import load_dataset


def resolve_cache_dir() -> str:
    """HF cache: HF_CACHE_DIR > mặc định dưới user home (tránh hardcode D:/)."""
    raw = os.environ.get("HF_CACHE_DIR") or str(
        Path.home() / ".cache" / "huggingface" / "datasets"
    )
    path = Path(raw).expanduser()
    # Ổ không tồn tại (vd. D:/ trên máy không có D) → fallback rõ ràng
    drive = path.drive
    if drive and not Path(drive + "/").exists():
        fallback = Path.home() / ".cache" / "huggingface" / "datasets"
        print(f"⚠️  HF_CACHE_DIR={raw} không dùng được (ổ {drive} không tồn tại).")
        print(f"    Dùng fallback: {fallback}")
        path = fallback
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


CACHE_DIR = resolve_cache_dir()
OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUTPUT_DIR / "java_completion_train.jsonl"

random.seed(42)

# ---------- Java vs non-Java detection ----------
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


# ---------- METHOD_RE loosened ----------
# Require ≥1 modifier (public/private/protected/static) to exclude control flow.
# Allows Allman brace style via `\s*\{` (newline whitespace before brace).
METHOD_RE = re.compile(
    r"(/\*\*.*?\*/\s*)?"
    r"((?:public|private|protected|static)"
    r"(?:\s+(?:public|private|protected|static|final|abstract|synchronized|native))*"
    r"\s+[\w<>,\[\]\?\s]+?"     # return type
    r"\s+\w+\s*\([^)]*\)\s*"    # name(args)
    r"(?:throws[^{]*)?\s*\{)",  # optional throws + brace
    re.DOTALL,
)


# ---------- Quality filter ----------
# Annotations not in MultiPL-E imports → drop. These are framework-specific.
EXTERNAL_ANNOTATION_RE = re.compile(
    r"@(?:Nullable|NonNull|NotNull|Nonnull|CheckForNull|"
    r"Inject|Component|Autowired|Service|Repository|Controller|RestController|"
    r"GetMapping|PostMapping|PutMapping|DeleteMapping|RequestMapping|RequestBody|RequestParam|PathVariable|"
    r"Test|Before|After|BeforeEach|AfterEach|BeforeAll|AfterAll|"
    r"JsonProperty|JsonIgnore|JsonInclude|JsonAlias|"
    r"Generated|VisibleForTesting|"
    r"Builder|Data|Getter|Setter|Slf4j|EqualsAndHashCode|AllArgsConstructor|NoArgsConstructor|RequiredArgsConstructor|"
    r"Entity|Table|Column|Id|GeneratedValue|"
    r"XmlElement|XmlAttribute|XmlRootElement|"
    r"Mock|MockBean|InjectMocks|Captor|"
    r"WebSocket|OnOpen|OnClose|OnMessage)\b"
)

# Whitelist: classes accessible via MultiPL-E header imports (java.util.*, java.lang.*,
# java.io.*, java.math.*, java.security.*, org.javatuples.*, java.util.stream.*).
JAVA_STD_CLASSES = frozenset({
    # java.lang
    "String", "Integer", "Double", "Float", "Long", "Short", "Byte", "Boolean", "Character",
    "Object", "Class", "Number", "Math", "System", "Throwable", "Thread", "Runtime",
    "StringBuilder", "StringBuffer", "Void",
    # java.util
    "Arrays", "Collections", "List", "Map", "Set", "HashMap", "HashSet", "ArrayList",
    "LinkedList", "TreeMap", "TreeSet", "LinkedHashMap", "LinkedHashSet", "Queue",
    "Stack", "Deque", "ArrayDeque", "PriorityQueue", "Optional", "Comparator",
    "Comparable", "Iterator", "Iterable", "Collection", "Spliterator", "EnumSet",
    "EnumMap", "Random", "Date", "Calendar", "TimeZone", "Locale", "UUID", "Objects",
    "Scanner", "StringTokenizer", "BitSet", "Vector", "Hashtable", "Properties",
    "AbstractMap", "AbstractList", "AbstractSet", "AbstractCollection",
    # java.math
    "BigInteger", "BigDecimal", "MathContext", "RoundingMode",
    # java.util.stream
    "Stream", "IntStream", "LongStream", "DoubleStream", "Collectors", "Collector",
    # java.io
    "InputStream", "OutputStream", "Reader", "Writer", "File", "BufferedReader",
    "BufferedWriter", "PrintStream", "PrintWriter", "IOException",
    # java.security
    "MessageDigest", "SecureRandom", "Signature",
    # org.javatuples
    "Pair", "Triplet", "Quartet", "Quintet", "Sextet", "Septet", "Octet", "Ennead", "Decade", "Tuple", "Unit",
    # Common exceptions (java.lang)
    "Exception", "RuntimeException", "IllegalArgumentException", "IllegalStateException",
    "NullPointerException", "IndexOutOfBoundsException", "ArithmeticException",
    "ArrayIndexOutOfBoundsException", "ClassCastException", "NumberFormatException",
    "UnsupportedOperationException", "ConcurrentModificationException",
    "StringIndexOutOfBoundsException", "NoSuchElementException", "Error", "AssertionError",
    # java.lang.reflect (header includes it)
    "Method", "Field", "Constructor", "Modifier", "Array",
    # Common primitive boxes already above
})

EXTERNAL_CLASS_REF_RE = re.compile(
    r"(?:^|[^\w.])(?:new\s+([A-Z]\w*)|([A-Z]\w*)\s*\.)",
    re.MULTILINE,
)


def has_external_refs(body: str) -> bool:
    """True if body uses annotations or classes outside MultiPL-E whitelist."""
    if EXTERNAL_ANNOTATION_RE.search(body):
        return True
    for m in EXTERNAL_CLASS_REF_RE.finditer(body):
        name = m.group(1) or m.group(2)
        if name and name not in JAVA_STD_CLASSES:
            return True
    return False


def has_indent_jumps(body: str, max_jump: int = 8) -> bool:
    """True if any consecutive non-empty lines have indent jump > max_jump."""
    prev = None
    for line in body.expandtabs(4).split("\n"):
        if not line.strip():
            continue
        ind = len(line) - len(line.lstrip(" "))
        if prev is not None and abs(ind - prev) > max_jump:
            return True
        prev = ind
    return False


# ---------- Indent normalization ----------
def javadoc_to_line_comments(jd: str) -> str:
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
        if cleaned.startswith("@"):
            continue
        out.append(cleaned)
    if not out:
        return ""
    return "\n".join(f"    // {l}" for l in out)


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


# ---------- Helper extraction (replaces inline-everything filter) ----------
_STRIP_LINE_COMMENT = re.compile(r"//[^\n]*")
_STRIP_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_STRIP_STRING_LIT = re.compile(r'"(?:\\.|[^"\\\n])*"')
CALL_RE = re.compile(r"\b([a-zA-Z_]\w*)\s*\(")
SKIP_NAMES = {"main", "if", "for", "while", "switch", "catch", "try", "new",
              "return", "class", "else", "do", "synchronized", "throw"}


def _strip_noise(src: str) -> str:
    src = _STRIP_LINE_COMMENT.sub("", src)
    src = _STRIP_BLOCK_COMMENT.sub("", src)
    return _STRIP_STRING_LIT.sub('""', src)


def extract_all_methods_in_src(java_src: str, exclude_name: str) -> dict:
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


def collect_needed_helpers(java_src: str, body: str, own_name: str) -> list:
    """Helpers transitively called from body, re-indented to 4-space base."""
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
        sub = set(CALL_RE.findall(_strip_noise(methods[name])))
        queue.extend((sub & set(methods.keys())) - visited)
    out = []
    for n in needed_names:
        code = methods[n]
        # Optional: drop helper if it has external refs (would break compile in eval)
        if has_external_refs(code):
            continue
        lines = code.expandtabs(4).split("\n")
        non_empty = [l for l in lines if l.strip()]
        if not non_empty:
            continue
        min_ind = min(len(l) - len(l.lstrip(" ")) for l in non_empty)
        out.append("\n".join(
            ("    " + l[min_ind:]) if l.strip() else "" for l in lines
        ))
    return out


# ---------- Main extraction ----------
def extract_method_completion(java_src: str, external_doc: str = ""):
    if not java_src:
        return None
    m = METHOD_RE.search(java_src)
    if not m:
        return None
    sig = m.group(2)
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

    if CSHARP_SIGNALS.search(body) or NON_JAVA_SIGNALS.search(body):
        return None

    # NEW Plan B: quality filter
    if has_external_refs(body):
        return None
    if has_indent_jumps(body):
        return None

    own_name_m = re.search(r"\b(\w+)\s*\(", sig)
    own_name = own_name_m.group(1) if own_name_m else ""

    # NEW Plan B: include helpers instead of dropping the sample
    helpers = collect_needed_helpers(java_src, body, own_name)

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
    # Normalize the signature line to single-line K&R style at 4-space indent.
    sig_compact = re.sub(r"\s+", " ", sig.strip())
    sig_line = f"    {sig_compact}\n"
    prefix = header + (doc_block + "\n" if doc_block else "") + sig_line

    target = body + "\n    }\n"
    if helpers:
        target += "\n" + "\n\n".join(helpers) + "\n"

    return {"prefix": prefix, "target": target}


def extract_from_markdown(text: str):
    m = re.search(r"```java\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    code = m.group(1)
    if CSHARP_SIGNALS.search(code) or NON_JAVA_SIGNALS.search(code):
        return None
    return extract_method_completion(code)


# ---------- Decontamination ----------
def load_he_method_keywords():
    he = load_dataset("nuprl/MultiPL-E", "humaneval-java", split="test", cache_dir=CACHE_DIR)
    keywords = set()
    for name in he["name"]:
        parts = name.split("_", 2)
        if len(parts) >= 3:
            keywords.add(parts[2].replace("_", "").lower())
    return {k for k in keywords if len(k) > 6}


def has_leak(prefix: str, target: str, keywords: set) -> bool:
    blob = (prefix + target).lower().replace("_", "")
    return any(k in blob for k in keywords)


# ---------- Collectors ----------
def collect_code_search_net(target_n=50000):
    print("--- code_search_net (java) ---")
    samples = []
    ds = None
    for repo in ["code-search-net/code_search_net", "code_search_net"]:
        try:
            ds = load_dataset(repo, "java", split="train", cache_dir=CACHE_DIR, trust_remote_code=True)
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
            if not re.search(r"\b(?:public|private|protected|static)\b", code):
                continue
        out = extract_method_completion(code, external_doc=external_doc)
        if out:
            samples.append(out)
    print(f"  collected: {len(samples)}")
    return samples


def collect_the_stack_smol(target_n=20000):
    print("--- the-stack-smol (java) ---")
    samples = []
    try:
        ds = load_dataset("bigcode/the-stack-smol", data_dir="data/java", split="train", cache_dir=CACHE_DIR)
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


def collect_magicoder(target_n=8000):
    print("--- Magicoder-OSS-Instruct-75K (Java) ---")
    samples = []
    try:
        ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train", cache_dir=CACHE_DIR)
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


def collect_magicoder_evol(target_n=15000):
    """NEW (Plan B): Evol-augmented instruction data, higher diversity."""
    print("--- Magicoder-Evol-Instruct-110K (Java) ---")
    samples = []
    try:
        ds = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train", cache_dir=CACHE_DIR)
    except Exception as e:
        print(f"  Skip Magicoder-Evol: {e}")
        return samples
    for ex in ds:
        if len(samples) >= target_n:
            break
        instruction = ex.get("instruction", "") or ""
        response = ex.get("response", "") or ex.get("output", "") or ""
        text = instruction + "\n" + response
        if not is_real_java(text):
            continue
        out = extract_from_markdown(response) or extract_from_markdown(text) or extract_method_completion(response)
        if out:
            samples.append(out)
    print(f"  collected: {len(samples)}")
    return samples


def collect_evol(target_n=5000):
    print("--- Evol-Instruct (filtered Java) ---")
    samples = []
    try:
        ds = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train", cache_dir=CACHE_DIR)
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


def main():
    print("=== Plan B: Build Java function-completion dataset ===")
    print(f"Output: {OUT_PATH}")

    print("\nLoading HumanEval-Java keywords for decontamination...")
    he_keywords = load_he_method_keywords()
    print(f"  {len(he_keywords)} keywords")

    all_samples = []
    all_samples += collect_code_search_net(target_n=50000)
    all_samples += collect_the_stack_smol(target_n=20000)
    all_samples += collect_magicoder(target_n=8000)
    all_samples += collect_magicoder_evol(target_n=15000)
    all_samples += collect_evol(target_n=5000)

    print(f"\nTotal raw: {len(all_samples)}")

    seen = set()
    deduped = []
    for s in all_samples:
        key = (s["prefix"][:200], s["target"][:200])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)
    print(f"After dedup: {len(deduped)}")

    clean = [s for s in deduped if not has_leak(s["prefix"], s["target"], he_keywords)]
    print(f"Removed leaks: {len(deduped) - len(clean)}")
    print(f"Final: {len(clean)}")

    random.shuffle(clean)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for s in clean:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(clean)} samples -> {OUT_PATH}")
    if clean:
        n_with_helpers = sum(1 for s in clean if s["target"].count("    }") > 1)
        print(f"Samples with helper bodies appended: {n_with_helpers} ({n_with_helpers/len(clean)*100:.1f}%)")
        print("\n--- Sample 0 PREFIX ---")
        print(clean[0]["prefix"][:400])
        print("--- Sample 0 TARGET ---")
        print(clean[0]["target"][:400])


if __name__ == "__main__":
    main()
