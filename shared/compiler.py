"""Shared Java compile and execution runner with compiler stderr parsing.
"""

import os
import subprocess
import re
import urllib.request
from pathlib import Path

_MISSING_METHOD_RE = re.compile(
    r"cannot find symbol\s*\n\s*(?:symbol:\s*)?method\s+(\w+)\(([^)]*)\)",
    re.IGNORECASE,
)


def download_javatuples(target_dir=None) -> str:
    """Tự động kiểm tra hoặc tải javatuples-1.2.jar."""
    jar_name = "javatuples-1.2.jar"
    repo_root = Path(__file__).resolve().parent.parent

    # Check candidates
    candidates = [
        Path(target_dir) / jar_name if target_dir else None,
        repo_root / jar_name,
        repo_root / "evaluate" / jar_name,
        Path.cwd() / jar_name,
    ]
    for c in candidates:
        if c and c.exists():
            return str(c.resolve())

    # Download to target or repo_root / "evaluate"
    save_dir = Path(target_dir) if target_dir else repo_root / "evaluate"
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / jar_name
    print(f"Downloading {jar_name} to {out_path}...")
    url = "https://repo1.maven.org/maven2/org/javatuples/javatuples/1.2/javatuples-1.2.jar"
    urllib.request.urlretrieve(url, str(out_path))
    print("Done.")
    return str(out_path.resolve())


def parse_missing_methods(stderr: str) -> list[dict]:
    """Trích danh sách missing methods [{'name': ..., 'args': ...}] từ compiler stderr."""
    seen = set()
    out = []
    for m in _MISSING_METHOD_RE.finditer(stderr):
        name = m.group(1)
        args = m.group(2).strip()
        key = (name, args)
        if key in seen:
            continue
        seen.add(key)
        out.append({"name": name, "args": args})
    return out


def compile_and_run_java(
    prompt: str,
    completion: str,
    tests: str,
    jar_path: str,
    timeout: int = 5,
    temp_dir: str = None,
) -> tuple[str, str]:
    """Biên dịch và chạy thử 1 test case HumanEval Java.
    
    Returns:
        (status, stderr):
            status in ["PASSED", "COMPILE_ERROR", "FAILED", "TIMEOUT"]
            stderr: text output of javac / java
    """
    working_dir = Path(temp_dir) if temp_dir else Path.cwd()
    java_file = working_dir / "Problem.java"
    class_file = working_dir / "Problem.class"

    full_code = prompt + "\n" + completion + "\n" + tests
    with open(java_file, "w", encoding="utf-8") as f:
        f.write(full_code)

    classpath = f".{os.pathsep}{jar_path}"
    compile_cmd = ["javac", "-encoding", "utf-8", "-cp", classpath, "Problem.java"]

    try:
        cp = subprocess.run(
            compile_cmd,
            cwd=str(working_dir),
            capture_output=True,
            text=True,
            timeout=15,
        )
        if cp.returncode != 0:
            return "COMPILE_ERROR", cp.stderr

        # Run test
        run_cmd = ["java", "-cp", classpath, "Problem"]
        rp = subprocess.run(
            run_cmd,
            cwd=str(working_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if rp.returncode == 0:
            return "PASSED", ""
        else:
            return "FAILED", rp.stderr
    except subprocess.TimeoutExpired:
        return "TIMEOUT", "Timeout expired"
    finally:
        # Cleanup temp artifacts
        if java_file.exists():
            try:
                os.remove(java_file)
            except OSError:
                pass
        if class_file.exists():
            try:
                os.remove(class_file)
            except OSError:
                pass
        # Remove inner class files like Problem$1.class if any
        for f in working_dir.glob("Problem$*.class"):
            try:
                os.remove(f)
            except OSError:
                pass
