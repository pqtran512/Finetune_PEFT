import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.quality_metrics import (  # noqa: E402
    compare_passed,
    instrument_main_for_timing,
    median,
    method_line_span,
    parse_checkstyle_report,
    parse_pmd_report,
    parse_runtime_ns,
    parse_spotbugs_report,
    source_without_tests,
    summarize_passed,
    target_method_name,
)


PROMPT = """import java.util.*;
class Problem {
    public static boolean hasCloseElements(ArrayList<Float> numbers, float threshold) {
"""

COMPLETION = """
        for (int i = 0; i < numbers.size(); i++) {
            if (numbers.get(i) == null) {
                return false;
            }
        }
        return false;
    }
"""

TESTS = """
    }
    public static void main(String[] args) {
    assert(hasCloseElements(new ArrayList<Float>(), (0.5f)) == (false));
    }

}
"""


def test_target_method_is_last_signature():
    assert target_method_name(PROMPT) == "hasCloseElements"


def test_source_without_tests_closes_class_and_method():
    source = source_without_tests(PROMPT, COMPLETION)
    assert source.count("{") == source.count("}")
    assert "void main" not in source
    assert "hasCloseElements" in source


def test_method_line_span_covers_body():
    source = source_without_tests(PROMPT, COMPLETION)
    span = method_line_span(source, "hasCloseElements")
    assert span is not None
    start, end = span
    assert start < end
    snippet = "\n".join(source.splitlines()[start - 1 : end])
    assert "return false" in snippet


def test_instrument_main_prints_marker_and_keeps_assert():
    full = PROMPT + COMPLETION + TESTS
    # completion already closes the method; tests close it again in this fixture.
    # Use the cleaned shape the runner uses: prompt opens the method, tests close it.
    from shared.cleaner import clean_java_completion

    full = PROMPT + "\n" + clean_java_completion(COMPLETION) + "\n" + TESTS
    timed = instrument_main_for_timing(full, loops=10, warmup=2)
    assert "QUALITY_RUNTIME_NS=" in timed
    assert "assert(hasCloseElements" in timed
    assert timed.count("{") == timed.count("}")


def test_parse_pmd_merges_complexity_for_target_method():
    xml = """<?xml version="1.0"?>
    <pmd version="7.0.0">
      <file name="Problem.java">
        <violation beginline="3" rule="CyclomaticComplexity">
          The method 'hasCloseElements(ArrayList&lt;Float&gt;, float)' has a cyclomatic complexity of 4.
        </violation>
        <violation beginline="3" rule="CognitiveComplexity">
          The method 'hasCloseElements' has a cognitive complexity of 5, current threshold is 1
        </violation>
        <violation beginline="20" rule="CyclomaticComplexity">
          The method 'helper' has a cyclomatic complexity of 2.
        </violation>
      </file>
    </pmd>
    """
    rows = {item["method"]: item for item in parse_pmd_report(xml)}
    assert rows["hasCloseElements"]["cyclomatic"] == 4
    assert rows["hasCloseElements"]["cognitive"] == 5
    assert rows["helper"]["cyclomatic"] == 2


def test_parse_checkstyle_keeps_only_target_lines():
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <checkstyle version="10.21.4">
      <file name="Problem.java">
        <error line="4" severity="warning" message="magic" source="com.puppycrawl.tools.checkstyle.checks.coding.MagicNumberCheck"/>
        <error line="30" severity="warning" message="in main" source="com.puppycrawl.tools.checkstyle.checks.coding.MagicNumberCheck"/>
      </file>
    </checkstyle>
    """
    warnings = parse_checkstyle_report(xml, (3, 10))
    assert len(warnings) == 1
    assert warnings[0]["line"] == 4
    assert warnings[0]["check"] == "MagicNumber"


def test_parse_spotbugs_splits_target_and_helper():
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <BugCollection>
      <BugInstance type="NP_NULL" rank="2">
        <Method classname="Problem" name="hasCloseElements" signature="()Z"/>
      </BugInstance>
      <BugInstance type="DLS_DEAD" rank="4">
        <Method classname="Problem" name="helper" signature="()V"/>
      </BugInstance>
    </BugCollection>
    """
    target, helpers = parse_spotbugs_report(xml, "hasCloseElements")
    assert [item["type"] for item in target] == ["NP_NULL"]
    assert [item["method"] for item in helpers] == ["helper"]


def test_runtime_marker():
    assert parse_runtime_ns("noise\nQUALITY_RUNTIME_NS=42\n") == 42


def test_summary_ignores_failures_and_compare_uses_intersection():
    rows_a = [
        {"task_id": "a", "status": "PASSED", "cyclomatic": 2, "cognitive": 2,
         "checkstyle_warnings": 1, "spotbugs_warnings": 0, "runtime_ns": 100},
        {"task_id": "b", "status": "PASSED", "cyclomatic": 12, "cognitive": 9,
         "checkstyle_warnings": 0, "spotbugs_warnings": 2, "runtime_ns": 300},
        {"task_id": "c", "status": "FAILED", "cyclomatic": 99, "cognitive": 99,
         "checkstyle_warnings": 9, "spotbugs_warnings": 9, "runtime_ns": 1},
    ]
    rows_b = [
        {"task_id": "a", "status": "PASSED", "cyclomatic": 4, "cognitive": 4,
         "checkstyle_warnings": 0, "spotbugs_warnings": 0, "runtime_ns": 50},
        {"task_id": "b", "status": "FAILED", "cyclomatic": 1, "cognitive": 1,
         "checkstyle_warnings": 0, "spotbugs_warnings": 0, "runtime_ns": 10},
        {"task_id": "d", "status": "PASSED", "cyclomatic": 1, "cognitive": 1,
         "checkstyle_warnings": 0, "spotbugs_warnings": 0, "runtime_ns": 10},
    ]
    summary = summarize_passed(rows_a)
    assert summary["passed"] == 2
    assert summary["median_cyclomatic"] == 7
    assert summary["cyclomatic_gt_10"] == 1
    assert summary["median_runtime_ns"] == 200

    compared = compare_passed(rows_a, rows_b)
    assert compared["intersection_passed"] == 1
    assert compared["a"]["median_cyclomatic"] == 2
    assert compared["median_runtime_ratio_a_over_b"] == 2


def test_median_of_empty_is_none():
    assert median([]) is None


def test_instrument_rejects_warmup_past_loops():
    with pytest.raises(ValueError):
        instrument_main_for_timing("public static void main(String[] args) { }", loops=2, warmup=2)
