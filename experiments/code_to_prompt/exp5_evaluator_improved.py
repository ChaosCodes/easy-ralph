"""
Experiment 5: Evaluator Pivot Detection — Improved Prompt

Round 2 improvement over Experiment 2.
Key change: Added HARD_METRIC_PIVOT_RULE to prompt, teaching Claude that
hard metrics are binary (pass/fail) and independent of score trends.

Round 1 result: 87.5% (35/40), failed on hard_metric_keeps_failing (0/5)
Target: >= 90% overall, hard_metric_keeps_failing >= 60% (3/5)

Test matrix: 8 scenarios x 5 repeats = 40 trials
"""

import asyncio
import json
import re
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
from ralph_sdk.prompts import HARD_METRIC_PIVOT_RULE
from claude_code_sdk import AssistantMessage, ClaudeCodeOptions, query


# =============================================================================
# Test Scenarios (same as exp2)
# =============================================================================

@dataclass
class PivotScenario:
    name: str
    previous_scores: list[float]
    current_score: float
    attempt_number: int
    hard_failures: list[str]
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
        description="Hard metric (tests) failing after 2 attempts, even though score improved",
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


# =============================================================================
# Improved Prompt (Round 2)
# =============================================================================

PIVOT_DETECTION_PROMPT_V2 = f"""You are evaluating whether a software task should PIVOT (change direction) or CONTINUE (keep iterating).

## Pivot Conditions

Recommend PIVOT when ANY of these conditions are met:

1. **Too many attempts with weak progress**: attempt_number >= 3 AND average improvement < 5 points per attempt
   (average improvement = (current_score - first_score) / attempt_number)

2. **Scores declining**: The last 3 scores show consistent decline (each lower than the previous)

3. **Scores stuck**: The last 3 scores are all within 3 points of each other

4. **Hard constraint keeps failing**: A hard metric (like tests_pass or builds) fails after 2+ attempts
   **IMPORTANT**: This condition is evaluated INDEPENDENTLY of score trends.

5. **Very low score after multiple attempts**: Score < 40 after 2+ attempts

If NONE of these conditions are met, recommend CONTINUE.

{HARD_METRIC_PIVOT_RULE}

## Output Format

Respond with EXACTLY one line:
DECISION: PIVOT or CONTINUE

Then explain briefly:
REASON: <your reasoning>
"""

# Original prompt from exp2 for comparison
PIVOT_DETECTION_PROMPT_V1 = """You are evaluating whether a software task should PIVOT (change direction) or CONTINUE (keep iterating).

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


async def run_single_trial(scenario: PivotScenario, trial_num: int, system_prompt: str) -> dict:
    """Run a single API trial."""
    prompt = build_scenario_prompt(scenario)
    result_text = ""

    try:
        async for message in query(
            prompt=prompt,
            options=ClaudeCodeOptions(
                system_prompt=system_prompt,
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
    decision_match = re.search(r"DECISION:\s*(PIVOT|CONTINUE)", result_text, re.IGNORECASE)
    if decision_match:
        decision = decision_match.group(1).lower()
    else:
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
        "raw_output": result_text[:500],
        "error": None,
    }


async def run_experiment(prompt_version: str, system_prompt: str, repeats: int = 5):
    """Run the experiment with a given prompt version."""
    print(f"\n{'='*60}")
    print(f"Experiment 5 - {prompt_version}: Evaluator Pivot (Improved Prompt)")
    print(f"Scenarios: {len(SCENARIOS)} x {repeats} repeats = {len(SCENARIOS) * repeats} trials")
    print(f"{'='*60}")

    all_results = []

    for scenario in SCENARIOS:
        print(f"\n--- {scenario.name} (expected: {'pivot' if scenario.expected_pivot else 'continue'}) ---")
        print(f"    {scenario.description}")

        for trial in range(1, repeats + 1):
            print(f"  Trial {trial}/{repeats}...", end=" ", flush=True)
            result = await run_single_trial(scenario, trial, system_prompt)
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
    print(f"Results - {prompt_version} ({valid} valid trials, {errors} errors):")
    if valid:
        print(f"  Prompt accuracy: {correct}/{valid} ({correct/valid:.1%})")
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

    # Failure analysis
    failures = [r for r in all_results if not r["correct"] and not r["error"]]
    if failures:
        print(f"\nFailure analysis:")
        for f in failures:
            print(f"  [{f['scenario']}] expected={f['expected']} got={f['got']}")
            print(f"    Output: {f['raw_output'][:200]}")

    return {
        "prompt_version": prompt_version,
        "total": total,
        "errors": errors,
        "valid": valid,
        "correct": correct,
        "accuracy": correct / valid if valid else 0,
        "passed": (correct / valid >= 0.9) if valid else False,
        "results": all_results,
    }


async def main():
    """Run improved prompt experiment and compare with Round 1."""
    print("=" * 70)
    print("Experiment 5: Evaluator Pivot — Improved Prompt (Round 2)")
    print("=" * 70)
    print("\nKey change: Added HARD_METRIC_PIVOT_RULE to prompt")
    print("Round 1 result: 87.5% (0/5 on hard_metric_keeps_failing)")
    print("Target: >= 90% overall, hard_metric_keeps_failing >= 60%\n")

    # Run with improved prompt (V2)
    v2_results = await run_experiment("V2_improved", PIVOT_DETECTION_PROMPT_V2, repeats=5)

    # Summary comparison
    print(f"\n{'='*70}")
    print("COMPARISON: Round 1 vs Round 2")
    print(f"{'='*70}")
    print(f"  Round 1 (V1): 87.5% (35/40)")
    print(f"  Round 2 (V2): {v2_results['accuracy']:.1%} ({v2_results['correct']}/{v2_results['valid']})")

    # Check hard_metric_keeps_failing specifically
    hmkf = [r for r in v2_results["results"] if r["scenario"] == "hard_metric_keeps_failing" and not r["error"]]
    hmkf_correct = sum(1 for r in hmkf if r["correct"])
    print(f"\n  hard_metric_keeps_failing:")
    print(f"    Round 1: 0/5 (0%)")
    print(f"    Round 2: {hmkf_correct}/{len(hmkf)} ({hmkf_correct/len(hmkf):.0%})" if hmkf else "    Round 2: N/A")

    improvement = v2_results["accuracy"] - 0.875
    print(f"\n  Improvement: {improvement:+.1%}")
    print(f"  Merge to prompts.py: {'YES' if improvement >= 0.05 else 'NO (< 5% improvement)'}")

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"exp5_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump({
            "experiment": "exp5_evaluator_improved",
            "round": 2,
            "round1_accuracy": 0.875,
            "round1_hmkf": "0/5",
            "v2": {
                "accuracy": v2_results["accuracy"],
                "passed": v2_results["passed"],
                "total": v2_results["total"],
                "correct": v2_results["correct"],
                "hmkf_correct": hmkf_correct,
                "hmkf_total": len(hmkf),
            },
            "results": v2_results["results"],
        }, f, indent=2)

    print(f"\nResults saved: {output_file}")
    return v2_results["passed"]


if __name__ == "__main__":
    asyncio.run(main())
