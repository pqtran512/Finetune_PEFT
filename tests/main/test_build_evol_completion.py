"""Unit tests for Evol → generic method-body completion mapping."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data"))

from build_evol_completion_dataset import (  # noqa: E402
    to_completion_sample,
    extract_java_code,
)


def test_to_completion_sample_splits_after_brace():
    src = """
public class Demo {
    /** Sum two ints. */
    public static int add(int a, int b) {
        return a + b;
    }
}
"""
    out = to_completion_sample(src)
    assert out is not None
    assert out["prefix"].rstrip().endswith("{")
    assert "return a + b;" in out["target"]
    assert out["target"].rstrip().endswith("}")
    assert "class Problem" not in out["prefix"]
    assert "org.javatuples" not in out["prefix"]


def test_instruction_becomes_line_comments_when_no_javadoc():
    src = """
    public static int twice(int x) {
        return x * 2;
    }
"""
    out = to_completion_sample(src, instruction="Double the input value")
    assert out is not None
    assert "// Double the input value" in out["prefix"]
    assert "return x * 2;" in out["target"]


def test_extract_java_from_markdown():
    text = "Here:\n```java\npublic static void f() {\n  int x = 1;\n}\n```\n"
    code = extract_java_code(text)
    assert code is not None
    assert "int x = 1;" in code


def test_magicoder_style_response_maps_to_completion():
    """Magicoder response field is remapped like Evol — not left as chat text."""
    response = (
        "Sure.\n```java\n"
        "public static int square(int n) {\n"
        "    if (n < 0) {\n"
        "        return -n * n;\n"
        "    }\n"
        "    return n * n;\n"
        "}\n```\n"
    )
    code = extract_java_code(response)
    assert code is not None
    out = to_completion_sample(code, instruction="Return n squared")
    assert out is not None
    assert out["prefix"].rstrip().endswith("{")
    assert "return n * n;" in out["target"]
    assert "[INST]" not in out["prefix"]
