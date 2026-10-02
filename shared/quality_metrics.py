"""Quality metrics for Java completions that already pass HumanEval-Java tests.

Metrics, computed on the generated method only:
  - cyclomatic and cognitive complexity (PMD)
  - Checkstyle structural warnings
  - SpotBugs warnings
  - in-JVM test runtime (median of repeated process launches)

Medians use PASSED tasks only. Comparing two files uses the intersection
of task ids that both passed. The runtime ratio is time(first) / time(second).
"""

from __future__ import annotations

import json
import re
import statistics
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from .cleaner import clean_java_completion
from .compiler import (
    cleanup_problem_artifacts,
    compile_problem_source,
    download_javatuples,
    run_problem_class,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _REPO_ROOT / "evaluate" / "config"
_TOOLS_DIR = _REPO_ROOT / "evaluate" / "tools"

_METHOD_NAME_RE = re.compile(r"\b(\w+)\s*\([^;{}]*\)\s*\{")
_SKIP_NAMES = {"if", "for", "while", "switch", "catch", "synchronized"}
_MAIN_RE = re.compile(
    r"public\s+static\s+void\s+main\s*\([^)]*\)\s*(?:throws\s+[\w.,\s]+)?\{"
)
_CYCLO_RE = re.compile(
    r"The method '([^']+)' has a cyclomatic complexity of (\d+)",
    re.IGNORECASE,
)
_COG_RE = re.compile(
    r"The method '([^']+)' has a cognitive complexity of (\d+)",
    re.IGNORECASE,
)
_RUNTIME_RE = re.compile(r"QUALITY_RUNTIME_NS=(\d+)")

PMD_VERSION = "7.28.0"
CHECKSTYLE_VERSION = "10.23.1"
SPOTBUGS_VERSION = "4.9.8"


def target_method_name(prompt: str) -> str | None:
    """Tên method đang được completion: signature cuối cùng trong prompt."""
    names = [
        name
        for name in _METHOD_NAME_RE.findall(prompt)
        if name not in _SKIP_NAMES
    ]
    return names[-1] if names else None


def source_without_tests(prompt: str, completion: str) -> str:
    """Ghép prompt + completion và đóng ngoặc còn mở. Không gắn tests."""
    cleaned = clean_java_completion(completion)
    body = prompt + "\n" + cleaned
    deficit = body.count("{") - body.count("}")
    if deficit > 0:
        body += "\n" + ("}\n" * deficit)
    return body


def method_line_span(source: str, method_name: str) -> tuple[int, int] | None:
    """Dòng bắt đầu và kết thúc (1-based, inclusive) của method."""
    pattern = re.compile(rf"\b{re.escape(method_name)}\s*\(")
    for match in pattern.finditer(source):
        brace = source.find("{", match.end())
        if brace < 0:
            continue
        end = _matching_brace(source, brace)
        if end is None:
            continue
        start_line = source.count("\n", 0, match.start()) + 1
        end_line = source.count("\n", 0, end) + 1
        return start_line, end_line
    return None


def instrument_main_for_timing(source: str, loops: int = 200, warmup: int = 20) -> str:
    """Lặp thân main trong cùng một process và in median nano giây sau warmup."""
    if warmup >= loops:
        raise ValueError("warmup must be smaller than loops")
    match = _MAIN_RE.search(source)
    if not match:
        raise ValueError("public static void main not found")
    brace = match.end() - 1
    end = _matching_brace(source, brace)
    if end is None:
        raise ValueError("main method is missing a closing brace")
    body = source[brace + 1 : end]
    timed = f"""
        final int __loops = {int(loops)};
        final int __warmup = {int(warmup)};
        long[] __samples = new long[__loops];
        for (int __i = 0; __i < __loops; __i++) {{
            long __t0 = System.nanoTime();
            {body}
            __samples[__i] = System.nanoTime() - __t0;
        }}
        long[] __measured = java.util.Arrays.copyOfRange(__samples, __warmup, __loops);
        java.util.Arrays.sort(__measured);
        long __median = __measured[__measured.length / 2];
        System.out.println("QUALITY_RUNTIME_NS=" + __median);
"""
    return source[: brace + 1] + timed + source[end:]


def parse_runtime_ns(stdout: str) -> int | None:
    match = _RUNTIME_RE.search(stdout or "")
    return int(match.group(1)) if match else None


def parse_pmd_report(xml_text: str) -> list[dict]:
    """Mỗi phần tử: {method, cyclomatic|None, cognitive|None} gộp theo tên method."""
    by_method: dict[str, dict] = {}
    xml_text = _xml_slice(xml_text, "pmd")
    if not xml_text or "<violation" not in xml_text:
        return []
    root = ET.fromstring(xml_text)
    for node in root.iter():
        if _local(node.tag) != "violation":
            continue
        message = "".join(node.itertext())
        cyclo = _CYCLO_RE.search(message)
        cog = _COG_RE.search(message)
        if cyclo:
            name = _short_method_name(cyclo.group(1))
            entry = by_method.setdefault(name, {"method": name, "cyclomatic": None, "cognitive": None})
            entry["cyclomatic"] = int(cyclo.group(2))
        elif cog:
            name = _short_method_name(cog.group(1))
            entry = by_method.setdefault(name, {"method": name, "cyclomatic": None, "cognitive": None})
            entry["cognitive"] = int(cog.group(2))
    return list(by_method.values())


def parse_checkstyle_report(xml_text: str, line_span: tuple[int, int] | None) -> list[dict]:
    """Warning Checkstyle nằm trong line_span của method được chấm."""
    xml_text = _xml_slice(xml_text, "checkstyle")
    if not xml_text or "<error" not in xml_text:
        return []
    root = ET.fromstring(xml_text)
    out = []
    for node in root.iter():
        if _local(node.tag) != "error":
            continue
        line_raw = node.attrib.get("line")
        if line_raw is None:
            continue
        line = int(line_raw)
        if line_span is not None and not (line_span[0] <= line <= line_span[1]):
            continue
        source = node.attrib.get("source", "")
        out.append(
            {
                "line": line,
                "check": source.rsplit(".", 1)[-1].removesuffix("Check"),
                "message": node.attrib.get("message", ""),
            }
        )
    return out


def parse_spotbugs_report(xml_text: str, target_method: str | None) -> tuple[list[dict], list[dict]]:
    """Tách bug của method đề và bug của helper. main đã bị exclude ở filter XML."""
    target, helpers = [], []
    if not xml_text or "<BugInstance" not in xml_text:
        return target, helpers
    root = ET.fromstring(xml_text)
    for node in root.iter():
        if _local(node.tag) != "BugInstance":
            continue
        method = _spotbugs_method_name(node)
        item = {
            "method": method,
            "type": node.attrib.get("type", ""),
            "rank": node.attrib.get("rank", ""),
        }
        if target_method and method == target_method:
            target.append(item)
        elif method and method != "main":
            helpers.append(item)
    return target, helpers


def median(values: list[int | float]) -> float | None:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return statistics.median(nums)


def summarize_passed(rows: list[dict]) -> dict:
    """Median trên các bài PASSED. Bài không có số liệu bị bỏ khỏi median đó."""
    passed = [row for row in rows if row.get("status") == "PASSED"]
    cyclo = [row.get("cyclomatic") for row in passed]
    cog = [row.get("cognitive") for row in passed]
    checks = [row.get("checkstyle_warnings") for row in passed]
    bugs = [row.get("spotbugs_warnings") for row in passed]
    runtime = [row.get("runtime_ns") for row in passed]
    cyclo_known = [v for v in cyclo if v is not None]
    return {
        "tasks": len(rows),
        "passed": len(passed),
        "median_cyclomatic": median(cyclo_known),
        "median_cognitive": median([v for v in cog if v is not None]),
        "cyclomatic_gt_10": sum(1 for v in cyclo_known if v > 10),
        "cyclomatic_known": len(cyclo_known),
        "median_checkstyle_warnings": median([v for v in checks if v is not None]),
        "tasks_with_checkstyle_warning": sum(1 for v in checks if v),
        "checkstyle_known": sum(1 for v in checks if v is not None),
        "median_spotbugs_warnings": median([v for v in bugs if v is not None]),
        "tasks_with_spotbugs_warning": sum(1 for v in bugs if v),
        "spotbugs_known": sum(1 for v in bugs if v is not None),
        "median_runtime_ns": median([v for v in runtime if v is not None]),
        "runtime_known": sum(1 for v in runtime if v is not None),
    }


def compare_passed(rows_a: list[dict], rows_b: list[dict]) -> dict:
    """Chỉ các task_id cả hai đều PASSED. Tỷ lệ runtime là A / B trên từng bài, rồi lấy median."""
    by_a = {row["task_id"]: row for row in rows_a if row.get("status") == "PASSED"}
    by_b = {row["task_id"]: row for row in rows_b if row.get("status") == "PASSED"}
    shared_ids = sorted(set(by_a) & set(by_b))
    inter_a = [by_a[task_id] for task_id in shared_ids]
    inter_b = [by_b[task_id] for task_id in shared_ids]
    ratios = []
    for task_id in shared_ids:
        ta = by_a[task_id].get("runtime_ns")
        tb = by_b[task_id].get("runtime_ns")
        if ta is not None and tb not in (None, 0):
            ratios.append(ta / tb)
    summary_a = summarize_passed(inter_a)
    summary_b = summarize_passed(inter_b)
    return {
        "intersection_passed": len(shared_ids),
        "a": summary_a,
        "b": summary_b,
        "median_runtime_ratio_a_over_b": median(ratios),
        "runtime_ratio_known": len(ratios),
    }


def render_summary(path_label: str, summary: dict) -> str:
    cyclo_known = summary["cyclomatic_known"]
    gt = summary["cyclomatic_gt_10"]
    gt_pct = (100.0 * gt / cyclo_known) if cyclo_known else 0.0
    lines = [
        f"file: {path_label}",
        f"tasks: {summary['tasks']}",
        f"passed: {summary['passed']}",
        "median trên các bài PASSED:",
        f"  median_cyclomatic: {_fmt(summary['median_cyclomatic'])}",
        f"  median_cognitive: {_fmt(summary['median_cognitive'])}",
        f"  cyclomatic_gt_10: {gt}/{cyclo_known} ({gt_pct:.1f}%)",
        f"  median_checkstyle_warnings: {_fmt(summary['median_checkstyle_warnings'])}",
        f"  tasks_with_checkstyle_warning: {summary['tasks_with_checkstyle_warning']}/{summary['checkstyle_known']}",
        f"  median_spotbugs_warnings: {_fmt(summary['median_spotbugs_warnings'])}",
        f"  tasks_with_spotbugs_warning: {summary['tasks_with_spotbugs_warning']}/{summary['spotbugs_known']}",
        f"  median_runtime_ns: {_fmt(summary['median_runtime_ns'])}",
        f"  runtime_known: {summary['runtime_known']}/{summary['passed']}",
    ]
    return "\n".join(lines) + "\n"


def render_compare(label_a: str, label_b: str, compared: dict) -> str:
    a = compared["a"]
    b = compared["b"]
    lines = [
        f"a: {label_a}",
        f"b: {label_b}",
        f"intersection_passed: {compared['intersection_passed']}",
        "median trên giao các bài cả hai đều PASSED:",
        f"  median_cyclomatic_a: {_fmt(a['median_cyclomatic'])}",
        f"  median_cyclomatic_b: {_fmt(b['median_cyclomatic'])}",
        f"  median_cognitive_a: {_fmt(a['median_cognitive'])}",
        f"  median_cognitive_b: {_fmt(b['median_cognitive'])}",
        f"  cyclomatic_gt_10_a: {a['cyclomatic_gt_10']}/{a['cyclomatic_known']}",
        f"  cyclomatic_gt_10_b: {b['cyclomatic_gt_10']}/{b['cyclomatic_known']}",
        f"  median_checkstyle_warnings_a: {_fmt(a['median_checkstyle_warnings'])}",
        f"  median_checkstyle_warnings_b: {_fmt(b['median_checkstyle_warnings'])}",
        f"  median_spotbugs_warnings_a: {_fmt(a['median_spotbugs_warnings'])}",
        f"  median_spotbugs_warnings_b: {_fmt(b['median_spotbugs_warnings'])}",
        f"  median_runtime_ns_a: {_fmt(a['median_runtime_ns'])}",
        f"  median_runtime_ns_b: {_fmt(b['median_runtime_ns'])}",
        f"  median_runtime_ratio_a_over_b: {_fmt(compared['median_runtime_ratio_a_over_b'])}",
        f"  runtime_ratio_known: {compared['runtime_ratio_known']}/{compared['intersection_passed']}",
        "ratio > 1 nghĩa là file a chậm hơn file b trên cùng test.",
    ]
    return "\n".join(lines) + "\n"


def locate_tools() -> dict[str, Path]:
    pmd_bat = _TOOLS_DIR / "pmd" / f"pmd-bin-{PMD_VERSION}" / "bin" / "pmd.bat"
    checkstyle_jar = _TOOLS_DIR / "checkstyle" / f"checkstyle-{CHECKSTYLE_VERSION}-all.jar"
    spotbugs_jar = _TOOLS_DIR / "spotbugs" / f"spotbugs-{SPOTBUGS_VERSION}" / "lib" / "spotbugs.jar"
    missing = [
        str(path)
        for path in (pmd_bat, checkstyle_jar, spotbugs_jar)
        if not path.exists()
    ]
    if missing:
        script = _TOOLS_DIR / "download_tools.ps1"
        raise FileNotFoundError(
            "Chưa có binary PMD/Checkstyle/SpotBugs. Chạy:\n"
            f"  powershell -ExecutionPolicy Bypass -File \"{script}\"\n"
            "Thiếu:\n  " + "\n  ".join(missing)
        )
    return {
        "pmd_bat": pmd_bat,
        "checkstyle_jar": checkstyle_jar,
        "spotbugs_jar": spotbugs_jar,
        "pmd_ruleset": _CONFIG_DIR / "pmd-ruleset.xml",
        "checkstyle_config": _CONFIG_DIR / "checkstyle.xml",
        "spotbugs_exclude": _CONFIG_DIR / "spotbugs-exclude.xml",
    }


def load_jsonl(path: str | Path, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def evaluate_records(
    records: list[dict],
    tools: dict[str, Path],
    jar_path: str,
    repeats: int = 5,
    loops: int = 200,
    warmup: int = 20,
    timeout: int = 5,
    timing_timeout: int = 60,
    verbose: bool = True,
) -> list[dict]:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if warmup >= loops:
        raise ValueError("warmup must be smaller than loops")
    results = []
    total = len(records)
    for index, record in enumerate(records, 1):
        task_id = record.get("task_id", f"task_{index}")
        try:
            row = _evaluate_one(
                record,
                tools,
                jar_path,
                repeats=repeats,
                loops=loops,
                warmup=warmup,
                timeout=timeout,
                timing_timeout=timing_timeout,
            )
        except Exception as exc:
            row = _empty_row(task_id, "ERROR")
            row["error"] = f"{type(exc).__name__}: {exc}"
        results.append(row)
        if verbose:
            print(f"[{index}/{total}] {task_id}: {row['status']}")
    return results


def _evaluate_one(
    record: dict,
    tools: dict[str, Path],
    jar_path: str,
    repeats: int,
    loops: int,
    warmup: int,
    timeout: int,
    timing_timeout: int,
) -> dict:
    task_id = record.get("task_id", "")
    prompt = record.get("prompt", "")
    completion = record.get("completion", "")
    tests = record.get("tests", "")
    method = target_method_name(prompt)
    cleaned = clean_java_completion(completion)
    static_source = source_without_tests(prompt, completion)
    full_source = prompt + "\n" + cleaned + "\n" + tests
    span = method_line_span(static_source, method) if method else None

    row = _empty_row(task_id, "ERROR")
    row["target_method"] = method

    with tempfile.TemporaryDirectory(prefix="quality_") as tmp:
        working = Path(tmp)
        _fill_static_metrics(row, static_source, span, method, tools, working)
        compile_status, compile_err = compile_problem_source(full_source, jar_path, working)
        if compile_status != "OK":
            row["status"] = compile_status
            row["error"] = (compile_err or "")[:500]
            return row

        _fill_spotbugs(row, tools, jar_path, working, method)
        run_status, _stdout, run_err = run_problem_class(
            jar_path, working, timeout=timeout, enable_assertions=True
        )
        row["status"] = run_status
        if run_status != "PASSED":
            row["error"] = (run_err or "")[:500]
            return row

        _fill_runtime(
            row,
            full_source,
            jar_path,
            working,
            tools_repeats=repeats,
            loops=loops,
            warmup=warmup,
            timing_timeout=timing_timeout,
        )
        cleanup_problem_artifacts(working)
    return row


def _fill_static_metrics(row, static_source, span, method, tools, working: Path) -> None:
    java_path = working / "Problem.java"
    java_path.write_text(static_source, encoding="utf-8")
    pmd_text, pmd_err = _run_pmd(java_path, tools)
    if pmd_err:
        row["pmd_error"] = pmd_err
    methods = parse_pmd_report(pmd_text)
    by_name = {item["method"]: item for item in methods}
    if method and method in by_name:
        row["cyclomatic"] = by_name[method]["cyclomatic"]
        row["cognitive"] = by_name[method]["cognitive"]
    row["helpers"] = [
        item for name, item in by_name.items() if name != method
    ]
    check_text, check_err = _run_checkstyle(java_path, tools)
    if check_err:
        row["checkstyle_error"] = check_err
    warnings = parse_checkstyle_report(check_text, span)
    row["checkstyle_warnings"] = len(warnings) if not check_err else None
    row["checkstyle_details"] = warnings[:30]


def _fill_spotbugs(row, tools, jar_path, working: Path, method: str | None) -> None:
    xml_text, err = _run_spotbugs(working, jar_path, tools)
    if err:
        row["spotbugs_error"] = err
        row["spotbugs_warnings"] = None
        return
    target, helpers = parse_spotbugs_report(xml_text, method)
    row["spotbugs_warnings"] = len(target)
    row["spotbugs_details"] = target[:30]
    row["helper_spotbugs"] = helpers[:30]


def _fill_runtime(
    row,
    full_source: str,
    jar_path: str,
    working: Path,
    tools_repeats: int,
    loops: int,
    warmup: int,
    timing_timeout: int,
) -> None:
    try:
        timed_source = instrument_main_for_timing(full_source, loops=loops, warmup=warmup)
    except ValueError as exc:
        row["runtime_error"] = str(exc)
        return
    compile_status, compile_err = compile_problem_source(timed_source, jar_path, working)
    if compile_status != "OK":
        row["runtime_error"] = (compile_err or compile_status)[:500]
        return
    samples = []
    for _ in range(tools_repeats):
        status, stdout, err = run_problem_class(
            jar_path, working, timeout=timing_timeout, enable_assertions=True
        )
        if status != "PASSED":
            row["runtime_error"] = (err or status)[:500]
            continue
        value = parse_runtime_ns(stdout)
        if value is None:
            row["runtime_error"] = "missing QUALITY_RUNTIME_NS marker"
            continue
        samples.append(value)
    row["runtime_ns_samples"] = samples
    mid = median(samples)
    row["runtime_ns"] = int(round(mid)) if mid is not None else None


def _run_pmd(java_path: Path, tools: dict[str, Path]) -> tuple[str, str | None]:
    cmd = [
        "cmd",
        "/c",
        str(tools["pmd_bat"]),
        "check",
        "-d",
        str(java_path),
        "-R",
        str(tools["pmd_ruleset"]),
        "-f",
        "xml",
        "--no-cache",
    ]
    return _tool_output(cmd, ok_codes={0, 4}, xml_tag="<pmd")


def _run_checkstyle(java_path: Path, tools: dict[str, Path]) -> tuple[str, str | None]:
    cmd = [
        "java",
        "-jar",
        str(tools["checkstyle_jar"]),
        "-c",
        str(tools["checkstyle_config"]),
        "-f",
        "xml",
        str(java_path),
    ]
    return _tool_output(cmd, ok_codes={0, 1, 2}, xml_tag="<checkstyle")


def _run_spotbugs(working: Path, jar_path: str, tools: dict[str, Path]) -> tuple[str, str | None]:
    out_path = working / "spotbugs.xml"
    cmd = [
        "java",
        "-jar",
        str(tools["spotbugs_jar"]),
        "-textui",
        "-xml:withMessages",
        "-effort:max",
        "-exclude",
        str(tools["spotbugs_exclude"]),
        "-auxclasspath",
        jar_path,
        "-output",
        str(out_path),
        str(working),
    ]
    _text, err = _tool_output(cmd, ok_codes={0, 1, 2, 3}, xml_tag=None)
    if not out_path.exists():
        return "", err or "SpotBugs did not write a report"
    xml_text = out_path.read_text(encoding="utf-8", errors="replace")
    if err and "<BugCollection" not in xml_text:
        return xml_text, err
    return xml_text, None


def _tool_output(cmd: list[str], ok_codes: set[int], xml_tag: str | None) -> tuple[str, str | None]:
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "", "tool timeout"
    except OSError as exc:
        return "", str(exc)
    text = completed.stdout or ""
    if completed.returncode in ok_codes and (xml_tag is None or xml_tag in text or completed.returncode == 0):
        return text, None
    if xml_tag and xml_tag in text:
        return text, None
    err = (completed.stderr or text or f"exit {completed.returncode}")[:500]
    return text, err


def _empty_row(task_id: str, status: str) -> dict:
    return {
        "task_id": task_id,
        "status": status,
        "target_method": None,
        "cyclomatic": None,
        "cognitive": None,
        "helpers": [],
        "checkstyle_warnings": None,
        "checkstyle_details": [],
        "spotbugs_warnings": None,
        "spotbugs_details": [],
        "helper_spotbugs": [],
        "runtime_ns_samples": [],
        "runtime_ns": None,
        "error": None,
    }


def _matching_brace(source: str, open_index: int) -> int | None:
    depth = 0
    for index in range(open_index, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def _short_method_name(name: str) -> str:
    """PMD đôi khi in 'foo(ArgType)' thay vì 'foo'."""
    return name.split("(", 1)[0].strip()


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _xml_slice(text: str, root_tag: str) -> str:
    if not text:
        return ""
    idx = text.find("<?xml")
    if idx < 0:
        idx = text.find(f"<{root_tag}")
    return text[idx:] if idx >= 0 else text


def _spotbugs_method_name(bug_instance: ET.Element) -> str | None:
    for child in list(bug_instance):
        if _local(child.tag) == "Method":
            return child.attrib.get("name")
    for node in bug_instance.iter():
        if _local(node.tag) == "Method":
            return node.attrib.get("name")
    return None


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def default_jar_path() -> str:
    return download_javatuples()
