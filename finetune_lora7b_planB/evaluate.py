"""Plan B — Evaluate Java pass@1 on HumanEval-Java using shared pipeline.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.cleaner import clean_java_completion, _extract_all_methods, _find_needed_helpers
from shared.compiler import download_javatuples, parse_missing_methods, compile_and_run_java
from shared.runner import evaluate_jsonl


def evaluate(input_file: str = "java_qwen_inference_planB.jsonl"):
    """Evaluate completion file and output report and failures JSONL."""
    return evaluate_jsonl(input_file=input_file)


if __name__ == "__main__":
    fname = sys.argv[1] if len(sys.argv) > 1 else "java_qwen_inference_planB.jsonl"
    evaluate(fname)
