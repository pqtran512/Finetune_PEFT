"""Shared repair pass logic for repairing Java compilation errors across models.
"""

import json
from pathlib import Path
from tqdm import tqdm
from .directives import REPAIR_INSTRUCTION
from .cleaner import clean_helper_output


def build_repair_prompt(entry: dict) -> str:
    """Xây dựng repair prompt yêu cầu mô hình sinh bổ sung các hàm helper còn thiếu."""
    missing = entry.get("missing_methods", [])
    missing_lines = "\n".join(
        f"//   - {m['name']}({m['args']})" for m in missing
    ) or "//   (infer from compiler error below)"

    stderr_snippet = (entry.get("stderr") or "")[:1500]
    full_attempt = entry["prompt"] + "\n" + entry["completion"]

    return (
        REPAIR_INSTRUCTION
        + "// Missing methods to define:\n"
        + missing_lines
        + "\n// Compiler error (truncated):\n"
        + "\n".join("//   " + ln for ln in stderr_snippet.splitlines()[:30])
        + "\n\n"
        + "// === Existing Problem.java (do NOT repeat any of this) ===\n"
        + full_attempt
        + "\n\n// === Append the missing helper method definitions here ===\n"
        + "    private static "
    )


def repair_completions(
    input_path: str,
    failures_path: str,
    output_path: str,
    generate_fn,
) -> int:
    """Chạy Repair Pass chung cho bất kỳ mô hình nào.
    
    Args:
        input_path: Đường dẫn file inference jsonl gốc.
        failures_path: Đường dẫn file failures.jsonl (do evaluate sinh ra).
        output_path: Đường dẫn file output jsonl sau khi đã repair.
        generate_fn: Hàm nhận prompt_text -> trả về generated string.
                     Signature: generate_fn(prompt: str) -> str.
    
    Returns:
        Số lượng task được repair.
    """
    if not Path(input_path).exists():
        print(f"Error: input_path không tồn tại: {input_path}")
        return 0

    if not Path(failures_path).exists():
        print(f"Error: failures_path không tồn tại: {failures_path}")
        return 0

    # 1. Đọc failures và lọc các COMPILE_ERROR có missing_methods
    failures_by_id = {}
    with open(failures_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("status") == "COMPILE_ERROR" and item.get("missing_methods"):
                failures_by_id[item["task_id"]] = item

    if not failures_by_id:
        print("Không có COMPILE_ERROR nào có missing_methods để repair.")
        # Sao chép nguyên input sang output nếu đường dẫn khác nhau
        if Path(input_path).resolve() != Path(output_path).resolve():
            with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
                fout.write(fin.read())
        return 0

    print(f"Tìm thấy {len(failures_by_id)} task COMPILE_ERROR cần repair.")

    # 2. Sinh repair code bằng generate_fn
    repaired_completions = {}
    for task_id, entry in tqdm(failures_by_id.items(), desc="Repairing"):
        prompt_text = build_repair_prompt(entry)
        raw_helper = generate_fn(prompt_text)
        helper_body = clean_helper_output(raw_helper)

        # build_repair_prompt kết thúc bằng mid-token '    private static '
        # Ta cần đảm bảo tiền tố đó xuất hiện trong helper code được ghép
        if not helper_body.strip().startswith("private static") and not helper_body.strip().startswith("public static"):
            full_helper = "    private static " + helper_body.lstrip()
        else:
            full_helper = helper_body

        old_completion = entry["completion"].rstrip()
        new_completion = old_completion + "\n\n" + full_helper
        repaired_completions[task_id] = new_completion

    # 3. Merge vào toàn bộ input gốc và lưu vào output
    total_written = 0
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            tid = data.get("task_id")
            if tid in repaired_completions:
                data["completion"] = repaired_completions[tid]
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            total_written += 1

    print(f"Đã lưu kết quả repaired vào {output_path} ({len(repaired_completions)} tasks được sửa / tổng {total_written} tasks).")
    return len(repaired_completions)
