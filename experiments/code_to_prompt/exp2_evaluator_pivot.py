"""
Experiment 2: Evaluator Pivot Detection

Hypothesis: Natural language pivot conditions in the prompt produce
the same decisions as the 5 hardcoded conditions in _assess_pivot_recommendation().

Replace target: evaluator.py:554-601 (5 conditions, 48 lines)

Phase 1: Test code-based _assess_pivot_recommendation against ground truth (sanity check)
Phase 2: Test Claude with natural language pivot conditions (requires API)

Test matrix: 8 scenarios x 5 repeats = 40 trials
"""

import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ralph_sdk.evaluator import (
    EvaluationResult,
    Metric,
    MetricResult,
    MetricType,
    _assess_pivot_recommendation,
)
from claude_code_sdk import AssistantMessage, ClaudeCodeOptions, query


# =============================================================================
# Test Scenarios
# =============================================================================

@dataclass
class PivotScenario:
    name: str
    previous_scores: list[float]
    current_score: float
    attempt_number: int
    hard_failures: list[str]  # Names of failing hard metrics
    expected_pivot: bool
    description: str


SCENARIOS = [
    # --- Should Pivot ---
    PivotScenario(
        name="too_many_attempts_weak_improvement",
        previous_scores=[30, 32],
        current_score=34,
        attempt_number=3,
        hard_failures=[],
        expected_pivot=True,
        description="3 attempts, avg improvement 1.3 per attempt (< 5 threshold)",
    ),
    PivotScenario(
        name="declining_scores",
        previous_scores=[40, 35],
        current_score=30,
        attempt_number=3,
        hard_failures=[],
        expected_pivot=True,
        description="Scores declining: 40 -> 35 -> 30",
    ),
    PivotScenario(
        name="stuck_scores",
        previous_scores=[44, 45],
        current_score=44,
        attempt_number=3,
        hard_failures=[],
        expected_pivot=True,
        description="Scores stuck within 3 points: 44, 45, 44",
    ),
    PivotScenario(
        name="hard_metric_keeps_failing",
        previous_scores=[50],
        current_score=55,
        attempt_number=2,
        hard_failures=["tests_pass"],
        expected_pivot=True,
        description="Hard metric (tests) failing after 2 attempts",
    ),
    PivotScenario(
        name="very_low_after_multiple",
        previous_scores=[30],
        current_score=35,
        attempt_number=2,
        hard_failures=[],
        expected_pivot=True,
        description="Score < 40 after 2 attempts",
    ),

    # --- Should NOT Pivot ---
    PivotScenario(
        name="normal_improvement",
        previous_scores=[60],
        current_score=75,
        attempt_number=2,
        hard_failures=[],
        expected_pivot=False,
        description="Good improvement: 60 -> 75",
    ),
    PivotScenario(
        name="first_attempt",
        previous_scores=[],
        current_score=50,
        attempt_number=1,
        hard_failures=[],
        expected_pivot=False,
        description="First attempt, score 50",
    ),
    PivotScenario(
        name="sufficient_improvement",
        previous_scores=[40, 50],
        current_score=60,
        attempt_number=3,
        hard_failures=[],
        expected_pivot=False,
        description="Consistent improvement: 40 -> 50 -> 60",
    ),
]


def build_eval_result(scenario: PivotScenario) -> EvaluationResult:
    """Build a mock EvaluationResult for a scenario."""
    metrics = []

    # Add a passing soft metric
    metrics.append(MetricResult(
        metric=Metric("code_quality", MetricType.SUBJECTIVE, "Code quality"),
        passed=True,
        score=scenario.current_score,
        reason="Score based on scenario",
    ))

    # Add hard metrics (failing or passing based on scenario)
    if scenario.hard_failures:
        for name in scenario.hard_failures:
            metrics.append(MetricResult(
                metric=Metric(name, MetricType.HARD, f"{name} must pass"),
                passed=False,
                reason=f"{name} is failing",
            ))
    else:
        metrics.append(MetricResult(
            metric=Metric("tests_pass", MetricType.HARD, "All tests pass"),
            passed=True,
            reason="Tests pass",
        ))

    return EvaluationResult(
        task_id="T001",
        overall_passed=not bool(scenario.hard_failures),
        overall_score=scenario.current_score,
        metrics=metrics,
        attempt_number=scenario.attempt_number,
        previous_scores=scenario.previous_scores,
    )


# =============================================================================
# Phase 1: Verify control (code) against ground truth
# =============================================================================

def run_phase1():
    """Test _assess_pivot_recommendation against expected outcomes."""
    print("=" * 60)
    print("Experiment 2 - Phase 1: Code Pivot Detection (Sanity Check)")
    print("=" * 60)

    correct = 0
    total = len(SCENARIOS)

    print(f"\n{'Scenario':<40} {'Expected':<8} {'Got':<8} {'Status'}")
    print("-" * 70)

    for s in SCENARIOS:
        result = build_eval_result(s)
        should_pivot, reason = _assess_pivot_recommendation(
            result=result,
            previous_scores=s.previous_scores,
            attempt_number=s.attempt_number,
            pivot_threshold=3,
            min_improvement=5.0,
        )

        ok = should_pivot == s.expected_pivot
        if ok:
            correct += 1
        status = "OK" if ok else f"MISMATCH (reason: {reason})"
        print(f"{s.name:<40} {str(s.expected_pivot):<8} {str(should_pivot):<8} {status}")

    print(f"\nCode accuracy: {correct}/{total} ({correct/total:.1%})")
    return correct, total


# =============================================================================
# Phase 2: Prompt-based pivot detection via Claude API
# =============================================================================

PIVOT_DETECTION_PROMPT = """You are evaluating whether a software task should PIVOT (change direction) or CONTINUE (keep iterating).

## Pivot Conditions

Recommend PIVOT when ANY of these conditions are met:

1. **Too many attempts with weak progress**: attempt_number >= 3 AND average improvement < 5 points per attempt
   (average improvement = (current_score - first_score) / attempt_number)

2. **Scores declining**: The last 3 scores show consistent decline (each lower than the previous)

3. **Scores stuck**: The last 3 scores are all within 3 points of each other

4. **Hard constraint keeps failing**: A hard metric (like tests_pass or builds) fails after 2+ attempts

5. **Very low score after multiple attempts**: Score < 40 after 2+ attempts

If NONE of these conditions are met, recommend CONTINUE.

## Output Format

Respond with EXACTLY one line:
DECISION: PIVOT or CONTINUE

Then explain briefly:
REASON: <your reasoning>
"""


def build_scenario_prompt(scenario: PivotScenario) -> str:
    """Build the prompt for a single scenario."""
    history_str = ", ".join(f"{s:.0f}" for s in scenario.previous_scores)
    if history_str:
        history_str = f"[{history_str}] -> {scenario.current_score:.0f}"
    else:
        history_str = f"{scenario.current_score:.0f} (first attempt)"

    hard_str = ", ".join(scenario.hard_failures) if scenario.hard_failures else "none"

    return f"""Evaluate this task's progress:

- **Attempt number**: {scenario.attempt_number}
- **Score history**: {history_str}
- **Current score**: {scenario.current_score:.0f}/100
- **Failing hard metrics**: {hard_str}

Should this task PIVOT or CONTINUE?"""


async def run_single_trial(scenario: PivotScenario, trial_num: int) -> dict:
    """Run a single API trial."""
    prompt = build_scenario_prompt(scenario)
    result_text = ""

    try:
        async for message in query(
            prompt=prompt,
            options=ClaudeCodeOptions(
                system_prompt=PIVOT_DETECTION_PROMPT,
                max_turns=1,
            ),
        ):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if hasattr(block, "text") and block.text:
                        result_text += block.text
    except Exception as e:
        return {
            "scenario": scenario.name,
            "trial": trial_num,
            "expected": "pivot" if scenario.expected_pivot else "continue",
            "got": None,
            "correct": False,
            "raw_output": "",
            "error": str(e),
        }

    # Parse decision
    import re
    decision_match = re.search(r"DECISION:\s*(PIVOT|CONTINUE)", result_text, re.IGNORECASE)
    if decision_match:
        decision = decision_match.group(1).lower()
    else:
        # Fallback: look for keywords
        text_lower = result_text.lower()
        if "pivot" in text_lower and "continue" not in text_lower:
            decision = "pivot"
        elif "continue" in text_lower and "pivot" not in text_lower:
            decision = "continue"
        else:
            decision = "unknown"

    expected = "pivot" if scenario.expected_pivot else "continue"
    correct = decision == expected

    return {
        "scenario": scenario.name,
        "trial": trial_num,
        "expected": expected,
        "got": decision,
        "correct": correct,
        "raw_output": result_text[:300],
        "error": None,
    }


async def run_phase2(repeats: int = 5):
    """Phase 2: Test Claude's pivot detection via API."""
    print(f"\n{'='*60}")
    print(f"Experiment 2 - Phase 2: Prompt-based Pivot Detection")
    print(f"Scenarios: {len(SCENARIOS)} x {repeats} repeats = {len(SCENARIOS) * repeats} trials")
    print(f"{'='*60}")

    all_results = []

    for scenario in SCENARIOS:
        print(f"\n--- {scenario.name} (expected: {'pivot' if scenario.expected_pivot else 'continue'}) ---")
        print(f"    {scenario.description}")

        for trial in range(1, repeats + 1):
            print(f"  Trial {trial}/{repeats}...", end=" ", flush=True)
            result = await run_single_trial(scenario, trial)
            all_results.append(result)

            if result["error"]:
                print(f"ERROR: {result['error']}")
            else:
                ok = "OK" if result["correct"] else "WRONG"
                print(f"{result['got']} ({ok})")

    # Aggregate
    total = len(all_results)
    errors = sum(1 for r in all_results if r["error"])
    valid = total - errors
    correct = sum(1 for r in all_results if r["correct"])

    print(f"\n{'='*60}")
    print(f"Results ({valid} valid trials, {errors} errors):")
    print(f"  Prompt accuracy: {correct}/{valid} ({correct/valid:.1%})" if valid else "  No valid trials")
    print(f"  Threshold: >= 90%")
    print(f"  PASSED: {'YES' if valid and correct/valid >= 0.9 else 'NO'}")
    print(f"{'='*60}")

    # Per-scenario breakdown
    print(f"\nPer-scenario breakdown:")
    for scenario in SCENARIOS:
        s_results = [r for r in all_results if r["scenario"] == scenario.name and not r["error"]]
        if s_results:
            c = sum(1 for r in s_results if r["correct"])
            n = len(s_results)
            print(f"  {scenario.name:<40} {c}/{n} correct")

    # Analyze failure patterns
    failures = [r for r in all_results if not r["correct"] and not r["error"]]
    if failures:
        print(f"\nFailure analysis:")
        for f in failures:
            print(f"  [{f['scenario']}] expected={f['expected']} got={f['got']}")
            print(f"    Output: {f['raw_output'][:200]}")

    # Save
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"exp2_phase2_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump({
            "experiment": "exp2_evaluator_pivot_phase2",
            "total_trials": total,
            "errors": errors,
            "accuracy": correct / valid if valid else 0,
            "passed": (correct / valid >= 0.9) if valid else False,
            "results": all_results,
        }, f, indent=2)

    print(f"\nResults saved: {output_file}")
    return (correct / valid >= 0.9) if valid else False


async def main():
    """Run both phases."""
    # Phase 1: sanity check
    code_correct, code_total = run_phase1()
    if code_correct < code_total:
        print("\nWARNING: Code-based detection doesn't match expected outcomes!")
        print("Fix the test scenarios or the code before proceeding to Phase 2.")

    # Phase 2: API test
    passed = await run_phase2(repeats=5)
    print(f"\nExperiment 2 overall: {'PASSED' if passed else 'FAILED'}")


if __name__ == "__main__":
    asyncio.run(main())
