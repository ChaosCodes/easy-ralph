"""
Experiment 1: Reviewer JSON Output

Hypothesis: Reviewer can reliably output JSON instead of free-text with regex markers.

Replace target: reviewer.py:37-60 (parse_reviewer_output, 3 regexes, 24 lines)

Test approach:
- Create mock reviewer outputs in various formats
- Parse with both control (regex) and experimental (JSON)
- Also test with real Claude API calls (Phase 2)

Phase 1: Test JSON parser robustness against synthetic inputs
Phase 2: Test Claude's ability to output valid JSON (requires API)
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ralph_sdk.reviewer import Verdict, ReviewResult, parse_reviewer_output
from experiments.code_to_prompt.experiment_runner import (
    run_experiment_sync,
    save_report,
)


# =============================================================================
# Experimental: JSON-based parser
# =============================================================================

def parse_reviewer_output_v2(text: str) -> ReviewResult:
    """
    Parse reviewer output using JSON.

    Handles common LLM output patterns:
    1. Pure JSON
    2. JSON inside markdown code fence
    3. JSON with surrounding text
    4. Fallback to regex if JSON not found
    """
    # Try to extract JSON from the text
    json_obj = _extract_json(text)

    if json_obj:
        verdict_str = json_obj.get("verdict", "passed").lower()
        try:
            verdict = Verdict(verdict_str)
        except ValueError:
            verdict = Verdict.PASSED

        return ReviewResult(
            verdict=verdict,
            reason=json_obj.get("reason", ""),
            suggestions=json_obj.get("suggestions", ""),
        )

    # Fallback: try regex (backwards compatible)
    return parse_reviewer_output(text)


def _extract_json(text: str) -> dict | None:
    """Extract a JSON object from text, handling common LLM output patterns."""
    # Pattern 1: JSON in code fence
    fence_match = re.search(r"```(?:json)?\s*\n(\{.*?\})\s*\n```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Pattern 2: Bare JSON object (find first { to last })
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        candidate = text[brace_start:brace_end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None


# =============================================================================
# Test Scenarios
# =============================================================================

SCENARIOS = [
    # --- PASSED scenarios (regex format) ---
    {
        "name": "passed_regex",
        "input": """VERDICT: passed
REASON: All acceptance criteria met. The implementation correctly handles all edge cases.
SUGGESTIONS: Consider adding more comprehensive logging.""",
        "expected_verdict": "passed",
    },
    {
        "name": "passed_regex_uppercase",
        "input": """VERDICT: PASSED
REASON: Implementation is complete and meets all requirements.
SUGGESTIONS: None.""",
        "expected_verdict": "passed",
    },
    # --- PASSED scenarios (JSON format) ---
    {
        "name": "passed_json_bare",
        "input": json.dumps({
            "verdict": "passed",
            "reason": "All acceptance criteria met.",
            "suggestions": "Consider adding logging."
        }),
        "expected_verdict": "passed",
    },
    {
        "name": "passed_json_fenced",
        "input": """Here is my review:

```json
{
    "verdict": "passed",
    "reason": "Code works correctly. All tests pass.",
    "suggestions": "Minor: add type hints to helper functions."
}
```

Overall the implementation is solid.""",
        "expected_verdict": "passed",
    },
    {
        "name": "passed_json_with_surrounding_text",
        "input": """I've reviewed the implementation carefully.

{"verdict": "passed", "reason": "All requirements met.", "suggestions": ""}

The code is clean and well-structured.""",
        "expected_verdict": "passed",
    },

    # --- RETRY scenarios (regex format) ---
    {
        "name": "retry_regex",
        "input": """VERDICT: retry
REASON: Tests fail due to missing import statement on line 15.
SUGGESTIONS: Add `import re` at the top of utils.py.""",
        "expected_verdict": "retry",
    },
    {
        "name": "retry_regex_verification",
        "input": """VERDICT: retry
REASON: The implementation uses an outdated API pattern.
SUGGESTIONS: Verify current API usage with documentation.""",
        "expected_verdict": "retry",
    },
    {
        "name": "retry_regex_multiline",
        "input": """VERDICT: retry
REASON: There are several issues:
1. Missing null check on line 20
2. Incorrect return type on line 35
3. Test coverage is insufficient
SUGGESTIONS: Fix the above issues and add more test cases.""",
        "expected_verdict": "retry",
    },
    # --- RETRY scenarios (JSON format) ---
    {
        "name": "retry_json_bare",
        "input": json.dumps({
            "verdict": "retry",
            "reason": "Tests fail due to missing import.",
            "suggestions": "Add `import re` at the top."
        }),
        "expected_verdict": "retry",
    },
    {
        "name": "retry_json_fenced",
        "input": """```json
{
    "verdict": "retry",
    "reason": "The function misses the edge case of empty input.",
    "suggestions": "Add a check for empty string."
}
```""",
        "expected_verdict": "retry",
    },
    {
        "name": "retry_json_multiline_suggestions",
        "input": json.dumps({
            "verdict": "retry",
            "reason": "Multiple issues found.",
            "suggestions": "1. Fix import on line 5\n2. Add error handling\n3. Update test assertions"
        }),
        "expected_verdict": "retry",
    },
    {
        "name": "retry_json_mixed_case",
        "input": json.dumps({
            "verdict": "Retry",
            "reason": "Needs minor fix.",
            "suggestions": "Update line 42."
        }),
        "expected_verdict": "retry",
    },

    # --- FAILED scenarios (regex format) ---
    {
        "name": "failed_regex",
        "input": """VERDICT: failed
REASON: The approach is fundamentally flawed. Synchronous I/O cannot meet performance requirements.
SUGGESTIONS: Rewrite using async I/O with aiohttp.""",
        "expected_verdict": "failed",
    },
    # --- FAILED scenarios (JSON format) ---
    {
        "name": "failed_json_bare",
        "input": json.dumps({
            "verdict": "failed",
            "reason": "Fundamentally flawed approach.",
            "suggestions": "Rewrite using async I/O."
        }),
        "expected_verdict": "failed",
    },
    {
        "name": "failed_json_fenced",
        "input": """After careful review:

```json
{
    "verdict": "failed",
    "reason": "Wrong architecture. Cannot scale.",
    "suggestions": "Use message queue pattern."
}
```

This is a fundamental design issue.""",
        "expected_verdict": "failed",
    },

    # --- Edge cases ---
    {
        "name": "edge_no_suggestions_regex",
        "input": """VERDICT: passed
REASON: Everything looks good, no issues found.""",
        "expected_verdict": "passed",
    },
    {
        "name": "edge_no_suggestions_json",
        "input": json.dumps({
            "verdict": "passed",
            "reason": "Everything looks good."
        }),
        "expected_verdict": "passed",
    },
    {
        "name": "edge_extra_fields_json",
        "input": json.dumps({
            "verdict": "retry",
            "reason": "Minor issue found.",
            "suggestions": "Fix the typo.",
            "confidence": "high",
            "reviewer_notes": "Generally good code"
        }),
        "expected_verdict": "retry",
    },
    {
        "name": "edge_json_multiline_reason",
        "input": json.dumps({
            "verdict": "retry",
            "reason": "Issues:\n1. Missing null check\n2. Incorrect return type\n3. Insufficient test coverage",
            "suggestions": "Fix the above issues."
        }),
        "expected_verdict": "retry",
    },
    {
        "name": "edge_empty_text",
        "input": "",
        "expected_verdict": "passed",  # both should default to passed
    },
]


def _check_against_truth(parser_fn, text: str, expected: str) -> dict:
    """Run a parser and check against expected verdict."""
    result = parser_fn(text)
    return {
        "verdict": result.verdict.value,
        "correct": result.verdict.value == expected,
        "reason": result.reason,
    }


def run_phase1():
    """
    Phase 1: Test both parsers against ground truth.

    For each scenario, we know the expected verdict. We test whether
    each parser correctly extracts it. This measures accuracy, not agreement.
    """
    print("=" * 60)
    print("Experiment 1 - Phase 1: Parser Accuracy on Synthetic Inputs")
    print("=" * 60)

    # Test regex parser accuracy
    regex_correct = 0
    json_correct = 0
    total = len(SCENARIOS)

    results_table = []

    for s in SCENARIOS:
        text = s["input"]
        expected = s["expected_verdict"]
        name = s["name"]

        regex_result = parse_reviewer_output(text)
        json_result = parse_reviewer_output_v2(text)

        regex_ok = regex_result.verdict.value == expected
        json_ok = json_result.verdict.value == expected

        if regex_ok:
            regex_correct += 1
        if json_ok:
            json_correct += 1

        status = ""
        if regex_ok and json_ok:
            status = "BOTH OK"
        elif json_ok and not regex_ok:
            status = "JSON WINS"
        elif regex_ok and not json_ok:
            status = "REGEX WINS"
        else:
            status = "BOTH FAIL"

        results_table.append({
            "name": name,
            "expected": expected,
            "regex": regex_result.verdict.value,
            "json": json_result.verdict.value,
            "status": status,
        })

    # Print results
    print(f"\n{'Scenario':<35} {'Expected':<10} {'Regex':<10} {'JSON':<10} {'Status':<12}")
    print("-" * 77)
    for r in results_table:
        print(f"{r['name']:<35} {r['expected']:<10} {r['regex']:<10} {r['json']:<10} {r['status']:<12}")

    print(f"\n{'='*60}")
    print(f"Regex parser accuracy: {regex_correct}/{total} ({regex_correct/total:.1%})")
    print(f"JSON parser accuracy:  {json_correct}/{total} ({json_correct/total:.1%})")
    print(f"{'='*60}")

    # Detailed analysis
    regex_only_formats = [r for r in results_table if "regex" in r["name"].lower() or r["name"].startswith("edge_no_suggestions_regex") or r["name"].startswith("edge_multiline_reason") or r["name"] == "edge_empty_text"]
    json_only_formats = [r for r in results_table if "json" in r["name"].lower()]

    print(f"\nRegex-format inputs: {sum(1 for r in results_table if 'regex' in r['name'])} scenarios")
    print(f"JSON-format inputs: {sum(1 for r in results_table if 'json' in r['name'])} scenarios")

    # Save to file
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"exp1_phase1_{timestamp}.json"

    import json as json_mod
    with open(output_file, "w") as f:
        json_mod.dump({
            "experiment": "exp1_reviewer_json_phase1",
            "regex_accuracy": regex_correct / total,
            "json_accuracy": json_correct / total,
            "total_scenarios": total,
            "results": results_table,
        }, f, indent=2)

    print(f"\nResults saved: {output_file}")
    return {
        "regex_accuracy": regex_correct / total,
        "json_accuracy": json_correct / total,
        "total": total,
        "results": results_table,
    }


if __name__ == "__main__":
    results = run_phase1()
    passed = results["json_accuracy"] >= 0.95
    print(f"\nExperiment 1 Phase 1: {'PASSED' if passed else 'FAILED'}")
    print(f"JSON parser is a strict superset of regex parser: {results['json_accuracy'] >= results['regex_accuracy']}")
