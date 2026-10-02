"""Chấm complexity, warning và thời gian chạy trên jsonl inference HumanEval-Java.

Một file:
    python evaluate/quality.py inference/jsonl/model_a.jsonl

Hai file (thêm median trên giao các bài cả hai đều pass, và time_a / time_b):
    python evaluate/quality.py model_a.jsonl model_b.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.quality_metrics import (
    compare_passed,
    default_jar_path,
    evaluate_records,
    load_jsonl,
    locate_tools,
    render_compare,
    render_summary,
    summarize_passed,
)


def write_quality(stem: str, rows: list[dict], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{stem}.quality.jsonl"
    report_path = output_dir / f"{stem}.quality.txt"
    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report_path.write_text(render_summary(stem, summarize_passed(rows)), encoding="utf-8")
    return jsonl_path, report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Đánh giá chất lượng completion Java đã pass test.")
    parser.add_argument("inputs", nargs="+", help="Một hoặc hai file jsonl inference")
    parser.add_argument(
        "--output-dir",
        default=str(_REPO_ROOT / "results" / "quality"),
        help="Thư mục ghi .quality.jsonl và .quality.txt",
    )
    parser.add_argument("--limit", type=int, default=None, help="Chỉ lấy N bài đầu")
    parser.add_argument("--repeats", type=int, default=5, help="Số process đo thời gian")
    parser.add_argument("--loops", type=int, default=200, help="Số lần lặp test trong một process")
    parser.add_argument("--warmup", type=int, default=20, help="Số lần lặp bỏ đi trước khi lấy median")
    parser.add_argument("--timeout", type=int, default=5, help="Timeout một lần chạy test, giây")
    parser.add_argument("--timing-timeout", type=int, default=60, help="Timeout một process đo thời gian, giây")
    args = parser.parse_args(argv)

    if len(args.inputs) > 2:
        parser.error("Chỉ nhận 1 hoặc 2 file jsonl")

    try:
        tools = locate_tools()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    jar_path = default_jar_path()
    output_dir = Path(args.output_dir)
    written = []
    for input_path in args.inputs:
        path = Path(input_path)
        records = load_jsonl(path, limit=args.limit)
        print(f"\n[START] {path} ({len(records)} tasks)")
        rows = evaluate_records(
            records,
            tools,
            jar_path,
            repeats=args.repeats,
            loops=args.loops,
            warmup=args.warmup,
            timeout=args.timeout,
            timing_timeout=args.timing_timeout,
            verbose=True,
        )
        jsonl_path, report_path = write_quality(path.stem, rows, output_dir)
        summary = render_summary(path.stem, summarize_passed(rows))
        print(summary)
        print(f"[REPORT] {report_path}")
        print(f"[JSONL]  {jsonl_path}")
        written.append((path.stem, rows))

    if len(written) == 2:
        (stem_a, rows_a), (stem_b, rows_b) = written
        compared = compare_passed(rows_a, rows_b)
        text = render_compare(stem_a, stem_b, compared)
        compare_path = output_dir / f"{stem_a}__vs__{stem_b}.compare.txt"
        compare_path.write_text(text, encoding="utf-8")
        print(text)
        print(f"[COMPARE] {compare_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
