import json
import os
import re
import random
from pathlib import Path
from datasets import load_dataset

CACHE_DIR = "D:/cache/hugging_face"
OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUTPUT_DIR / "java_completion_train.jsonl"

random.seed(42)

JAVA_TAG = re.compile(r"```java\b", re.IGNORECASE)
JAVA_KW = re.compile(
    r"\b(public\s+(?:static\s+)?(?:class|void|int|boolean|String)"
    r"|System\.out\.println"
    r"|import\s+java\.)\b"
)


def is_real_java(text: str) -> bool:
    if not text:
        return False
    if "javascript" in text.lower():
        return False
    return bool(JAVA_TAG.search(text) or JAVA_KW.search(text))


METHOD_RE = re.compile(
    r"(/\*\*.*?\*/\s*)?"
    r"(public\s+static\s+\S+\s+\w+\s*\([^)]*\)\s*(?:throws[^{]*)?\{)",
    re.DOTALL,
)


def extract_method_completion(java_src: str):
    m = METHOD_RE.search(java_src)
    if not m:
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
    javadoc = (m.group(1) or "").strip()
    sig = m.group(2)
    prefix = (
        "import java.util.*;\n"
        "import java.lang.reflect.*;\n"
        "import org.javatuples.*;\n"
        "class Problem {\n"
        f"    {javadoc}\n    {sig}\n"
    )
    target = body + "\n    }\n}\n"
    return {"prefix": prefix, "target": target}


def extract_from_markdown(text: str):
    """Lấy block code Java đầu tiên trong text markdown, rồi extract method."""
    m = re.search(r"```java\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    return extract_method_completion(m.group(1))


def load_he_method_keywords():
    """Lấy tên method HumanEval-Java để decontaminate."""
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


def collect_the_stack(target_n=30000):
    print("--- the-stack-smol Java ---")
    samples = []
    try:
        ds = load_dataset(
            "bigcode/the-stack-smol",
            data_dir="data/java",
            split="train",
            cache_dir=CACHE_DIR,
        )
    except Exception as e:
        print(f"Skip the-stack-smol: {e}")
        return samples
    for ex in ds:
        if len(samples) >= target_n:
            break
        src = ex.get("content", "")
        if not is_real_java(src):
            continue
        out = extract_method_completion(src)
        if out:
            samples.append(out)
    print(f"  collected: {len(samples)}")
    return samples


def collect_multipl_t(target_n=15000):
    print("--- MultiPL-T Java ---")
    samples = []
    for cfg in ["java", "Java"]:
        try:
            ds = load_dataset("nuprl/MultiPL-T", cfg, split="train", cache_dir=CACHE_DIR)
            break
        except Exception:
            ds = None
    if ds is None:
        print("  Skip MultiPL-T (config not found)")
        return samples
    for ex in ds:
        if len(samples) >= target_n:
            break
        # MultiPL-T thường có cột 'content' hoặc 'completion'
        src = ex.get("content") or ex.get("solution") or ex.get("completion") or ""
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
        text = (ex.get("problem", "") or "") + "\n" + (ex.get("solution", "") or "")
        if not is_real_java(text):
            continue
        out = extract_from_markdown(text) or extract_method_completion(text)
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
        text = (ex.get("instruction", "") or "") + "\n" + (ex.get("output", "") or "")
        if not is_real_java(text):
            continue
        out = extract_from_markdown(text) or extract_method_completion(text)
        if out:
            samples.append(out)
    print(f"  collected: {len(samples)}")
    return samples


def main():
    print("=== Build Java function-completion dataset ===")
    print(f"Output: {OUT_PATH}")

    print("\nLoading HumanEval-Java keywords for decontamination...")
    he_keywords = load_he_method_keywords()
    print(f"  {len(he_keywords)} keywords")

    all_samples = []
    all_samples += collect_the_stack(target_n=30000)
    all_samples += collect_multipl_t(target_n=15000)
    all_samples += collect_magicoder(target_n=8000)
    all_samples += collect_evol(target_n=5000)

    print(f"\nTotal raw: {len(all_samples)}")

    # Deduplicate by (prefix-truncated, target-truncated) hash
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
        print("PREFIX:", clean[0]["prefix"][:300])
        print("TARGET:", clean[0]["target"][:300])


if __name__ == "__main__":
    main()
