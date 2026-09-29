"""Evaluate Java pass@1 on HumanEval-Java using the shared evaluation pipeline.
Generates both .report.txt and .failures.jsonl (for repair pass).
"""

import os
import sys
from pathlib import Path

# Add repo root to sys.path so 'shared' can be imported when running from any CWD
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.cleaner import clean_java_completion, _extract_all_methods, _find_needed_helpers
from shared.compiler import download_javatuples, parse_missing_methods, compile_and_run_java
from shared.runner import evaluate_jsonl


def evaluate(input_file: str = "java_qwen_inference.jsonl"):
    """Evaluate completion results and produce report and failures files."""
    return evaluate_jsonl(input_file)


if __name__ == "__main__":
    fname = sys.argv[1] if len(sys.argv) > 1 else "java_qwen_inference_v4.jsonl"
    evaluate(fname)
