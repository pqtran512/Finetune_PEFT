"""Shared evaluation runner for Java completions on HumanEval-Java benchmark.
Outputs both report text and failures JSONL with parsed missing methods.
Windows cp1252-safe.
"""

import os
import sys
import json
from pathlib import Path
from collections import Counter

# Safe stdout on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from .cleaner import clean_java_completion
from .compiler import download_javatuples, compile_and_run_java, parse_missing_methods


def evaluate_jsonl(
    input_file: str,
    save_failures: bool = True,
    output_report: str = None,
    output_failures: str = None,
    verbose: bool = True,
) -> dict:
    """Đánh giá toàn diện 1 file jsonl sinh mã Java trên HumanEval-Java.
    
    Args:
        input_file: Đường dẫn file jsonl chứa [{'task_id', 'prompt', 'completion', 'tests'}, ...]
        save_failures: Có lưu file .failures.jsonl để chạy repair pass hay không
        output_report: Đường dẫn file report. Mặc định là {input_file}.report.txt
        output_failures: Đường dẫn file failures. Mặc định là {input_file}.failures.jsonl
        verbose: In tiến độ ra console
        
    Returns:
        Dict tổng hợp kết quả
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {input_file}")

    jar_path = download_javatuples()

    counts = Counter()
    failures = []
    failures_full = []

    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    total = len(lines)
    passed = 0

    if verbose:
        print(f"\n[START] Bat dau danh gia: {input_file} ({total} tasks)")
        print(f"[JAR] Javatuples path: {jar_path}")

    for idx, line in enumerate(lines, 1):
        data = json.loads(line)
        task_id = data.get("task_id", f"task_{idx}")
        prompt = data.get("prompt", "")
        raw_completion = data.get("completion", "")
        tests = data.get("tests", "")

        cleaned_completion = clean_java_completion(raw_completion)

        status, stderr = compile_and_run_java(
            prompt=prompt,
            completion=cleaned_completion,
            tests=tests,
            jar_path=jar_path,
        )

        counts[status] += 1
        if status == "PASSED":
            passed += 1
        else:
            failures.append({
                "task_id": task_id,
                "status": status,
                "stderr_short": stderr[:300].strip(),
            })

            missing_methods = parse_missing_methods(stderr) if status == "COMPILE_ERROR" else []
            failures_full.append({
                "task_id": task_id,
                "prompt": prompt,
                "completion": cleaned_completion,
                "tests": tests,
                "status": status,
                "stderr": stderr,
                "missing_methods": missing_methods,
            })

        if verbose:
            print(f"[{idx}/{total}] {task_id}: {status}")

    pass_rate = (passed / total) * 100 if total > 0 else 0.0

    # Xuất report.txt
    report_path = Path(output_report) if output_report else Path(f"{input_file}.report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"pass@1: {pass_rate:.2f}% ({passed}/{total})\n\n")
        f.write("Breakdown:\n")
        for st, c in sorted(counts.items()):
            f.write(f"  {st}: {c} ({c / total * 100:.1f}%)\n")
        f.write("\nFailures (task_id | status | stderr[:300]):\n")
        for fail in failures:
            f.write(f"{fail['task_id']} | {fail['status']} | {fail['stderr_short']}\n\n")

    # Xuất failures.jsonl
    failures_path = Path(output_failures) if output_failures else Path(f"{input_file}.failures.jsonl")
    if save_failures:
        with open(failures_path, "w", encoding="utf-8") as f:
            for fail_item in failures_full:
                f.write(json.dumps(fail_item, ensure_ascii=False) + "\n")

    if verbose:
        print("\n" + "=" * 50)
        print(f"[RESULT] Pass@1: {pass_rate:.2f}% ({passed}/{total})")
        print(f"Breakdown: {dict(counts)}")
        print(f"[REPORT] Saved report to: {report_path}")
        if save_failures:
            print(f"[FAILURES] Saved failures (for repair pass) to: {failures_path}")
        print("=" * 50 + "\n")

    return {
        "pass_rate": pass_rate,
        "passed": passed,
        "total": total,
        "counts": dict(counts),
        "failures": failures,
        "report_file": str(report_path),
        "failures_file": str(failures_path) if save_failures else None,
    }
