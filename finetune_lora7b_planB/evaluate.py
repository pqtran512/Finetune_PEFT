"""Plan B — Evaluate Java pass@1 on HumanEval-Java.

Thêm: failure-mode breakdown (COMPILE_ERROR vs FAILED vs TIMEOUT vs PASSED).
"""
import os
import subprocess
import json
import re
import sys
import urllib.request
from collections import Counter


def download_javatuples():
    jar_name = "javatuples-1.2.jar"
    url = "https://repo1.maven.org/maven2/org/javatuples/javatuples/1.2/javatuples-1.2.jar"
    if not os.path.exists(jar_name):
        print(f"Downloading {jar_name}...")
        urllib.request.urlretrieve(url, jar_name)
        print("Done.")
    return jar_name


def clean_java_completion(completion: str) -> str:
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
    return final


def _extract_all_methods(text: str) -> dict:
    methods = {}
    sig_re = re.compile(
        r"^[ \t]*(?:(?:public|private|protected|static|final)\s+)*"
        r"(\w+(?:<[^>]*>)?(?:\[\])*)\s+(\w+)\s*\(",
        re.MULTILINE,
    )
    skip = {"main", "if", "for", "while", "switch", "catch", "try", "new", "return", "class", "else"}
    for m in sig_re.finditer(text):
        name = m.group(2)
        if name in skip:
            continue
        rest = text[m.end():]
        brace_idx = rest.find("{")
        if brace_idx == -1 or brace_idx > 200:
            continue
        abs_brace = m.end() + brace_idx
        line_start = text.rfind("\n", 0, m.start())
        line_start = 0 if line_start == -1 else line_start + 1
        brace_count, end = 0, None
        for j in range(abs_brace, len(text)):
            if text[j] == "{":
                brace_count += 1
            elif text[j] == "}":
                brace_count -= 1
                if brace_count == 0:
                    end = j + 1
                    break
        if end and name not in methods:
            methods[name] = text[line_start:end].strip()
    return methods


def _find_needed_helpers(method_body: str, remaining_text: str) -> str:
    if not remaining_text.strip():
        return ""
    available = _extract_all_methods(remaining_text)
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


def evaluate(input_file="java_qwen_inference_planB.jsonl"):
    jar_path = download_javatuples()
    if not os.path.exists(input_file):
        print(f"Không thấy file {input_file}!")
        return

    counts = Counter()
    failures = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            cleaned = clean_java_completion(data["completion"])
            full_code = data["prompt"] + "\n" + cleaned + "\n" + data["tests"]

            file_name = "Problem.java"
            with open(file_name, "w", encoding="utf-8") as jf:
                jf.write(full_code)

            sep = ";" if os.name == "nt" else ":"
            compile_cmd = f'javac -encoding utf-8 -cp ".{sep}{jar_path}" {file_name}'
            cp = subprocess.run(compile_cmd, capture_output=True, text=True, shell=True)

            status = "FAILED"
            if cp.returncode != 0:
                status = "COMPILE_ERROR"
                failures.append((data["task_id"], status, cp.stderr[:300]))
            else:
                try:
                    run_cmd = f'java -ea -cp ".{sep}{jar_path}" Problem'
                    rp = subprocess.run(run_cmd, capture_output=True, text=True, timeout=10, shell=True)
                    if rp.returncode == 0:
                        status = "PASSED"
                    else:
                        failures.append((data["task_id"], "FAILED", rp.stderr[:300]))
                except subprocess.TimeoutExpired:
                    status = "TIMEOUT"
                    failures.append((data["task_id"], status, "timeout 10s"))

            counts[status] += 1
            print(f"[{sum(counts.values())}] {data['task_id']}: {status}")

            for f_temp in ["Problem.java", "Problem.class"]:
                if os.path.exists(f_temp):
                    os.remove(f_temp)

    total = sum(counts.values())
    passed = counts["PASSED"]

    print("\n" + "=" * 50)
    print(f"FINAL pass@1: {passed / total * 100:.2f}% ({passed}/{total})")
    print("Breakdown:")
    for k in ("PASSED", "COMPILE_ERROR", "FAILED", "TIMEOUT"):
        v = counts.get(k, 0)
        print(f"  {k:14s} {v:4d} ({v/total*100:5.2f}%)")
    print("=" * 50)

    with open(f"{input_file}.report.txt", "w", encoding="utf-8") as f:
        f.write(f"pass@1: {passed/total*100:.2f}% ({passed}/{total})\n\n")
        f.write("Failures (task_id | status | stderr[:300]):\n")
        for tid, st, err in failures:
            f.write(f"{tid} | {st} | {err}\n")
    print(f"Report: {input_file}.report.txt")


if __name__ == "__main__":
    fname = sys.argv[1] if len(sys.argv) > 1 else "java_qwen_inference_planB.jsonl"
    evaluate(fname)
