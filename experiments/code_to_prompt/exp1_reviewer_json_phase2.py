"""
Experiment 1 - Phase 2: Test Claude's ability to output JSON review verdicts.

Uses real Claude API calls via claude-code-sdk.
Sends 4 scenarios x 5 repeats = 20 trials.
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from claude_code_sdk import AssistantMessage, ClaudeCodeOptions, query
from ralph_sdk.reviewer import ReviewResult, Verdict, parse_reviewer_output

from experiments.code_to_prompt.exp1_reviewer_json import parse_reviewer_output_v2

# Modified reviewer prompt that requests JSON output
REVIEWER_JSON_PROMPT = """You are a task reviewer. Review the implementation and provide your verdict.

## Output Format

You MUST output your verdict as a JSON object (no other text before or after):

```json
{
    "verdict": "passed" | "retry" | "failed",
    "reason": "explanation of your verdict",
    "suggestions": "what to improve (empty string if passed)"
}
```

## Verdicts
- **passed**: Task complete, all criteria met
- **retry**: Non-fundamental issue, can be fixed with another attempt
- **failed**: Fundamental issue, needs different approach
"""

SCENARIOS = [
    {
        "name": "passed",
        "prompt": """Review this implementation:

Goal: Add a function that adds two numbers
Implementation: Created `add(a, b)` function in utils.py that returns a + b. Added tests for positive, negative, and zero inputs. All tests pass.

Provide your review verdict.""",
        "expected_verdict": "passed",
    },
    {
        "name": "retry_minor_bug",
        "prompt": """Review this implementation:

Goal: Add email validation function
Implementation: Created `validate_email(email)` in utils.py. It checks for @ symbol and domain with a dot. However, tests show it incorrectly accepts emails with spaces like "user @domain.com". This is a minor fix needed.

Provide your review verdict.""",
        "expected_verdict": "retry",
    },
    {
        "name": "retry_verification",
        "prompt": """Review this implementation:

Goal: Integrate with OpenAI API for text generation
Implementation: Used `openai.Completion.create()` API to generate text. The code works but uses the legacy completions API. No [已搜索验证] annotation found. The current OpenAI API uses `openai.chat.completions.create()` with a different parameter structure.

Provide your review verdict.""",
        "expected_verdict": "retry",
    },
    {
        "name": "failed",
        "prompt": """Review this implementation:

Goal: Build a real-time websocket server handling 10000 concurrent connections
Implementation: Used Flask's built-in development server with synchronous request handling. Each connection spawns a thread. This approach cannot scale to the required 10000 concurrent connections due to thread overhead and GIL limitations.

Provide your review verdict.""",
        "expected_verdict": "failed",
    },
]


async def run_single_trial(scenario: dict, trial_num: int) -> dict:
    """Run a single trial with Claude API."""
    result_text = ""

    try:
        async for message in query(
            prompt=scenario["prompt"],
            options=ClaudeCodeOptions(
                system_prompt=REVIEWER_JSON_PROMPT,
                max_turns=1,
            ),
        ):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if hasattr(block, "text") and block.text:
                        result_text += block.text
    except Exception as e:
        return {
            "scenario": scenario["name"],
            "trial": trial_num,
            "expected": scenario["expected_verdict"],
            "raw_output": "",
            "json_parsed": False,
            "json_verdict": None,
            "regex_verdict": None,
            "json_correct": False,
            "regex_correct": False,
            "error": str(e),
        }

    # Parse with both methods
    regex_result = parse_reviewer_output(result_text)
    json_result = parse_reviewer_output_v2(result_text)

    expected = scenario["expected_verdict"]

    return {
        "scenario": scenario["name"],
        "trial": trial_num,
        "expected": expected,
        "raw_output": result_text[:500],
        "json_parsed": json_result.verdict.value != "passed" or expected == "passed",  # rough check
        "json_verdict": json_result.verdict.value,
        "regex_verdict": regex_result.verdict.value,
        "json_correct": json_result.verdict.value == expected,
        "regex_correct": regex_result.verdict.value == expected,
        "error": None,
    }


async def run_phase2(repeats: int = 5):
    """Run Phase 2: Real Claude API calls."""
    print("=" * 60)
    print("Experiment 1 - Phase 2: Claude JSON Output Reliability")
    print(f"Scenarios: {len(SCENARIOS)} x {repeats} repeats = {len(SCENARIOS) * repeats} trials")
    print("=" * 60)

    all_results = []

    for scenario in SCENARIOS:
        print(f"\n--- Scenario: {scenario['name']} (expected: {scenario['expected_verdict']}) ---")
        for trial in range(1, repeats + 1):
            print(f"  Trial {trial}/{repeats}...", end=" ", flush=True)
            result = await run_single_trial(scenario, trial)
            all_results.append(result)

            if result["error"]:
                print(f"ERROR: {result['error']}")
            else:
                j_ok = "OK" if result["json_correct"] else "WRONG"
                r_ok = "OK" if result["regex_correct"] else "WRONG"
                print(f"JSON={result['json_verdict']}({j_ok}) Regex={result['regex_verdict']}({r_ok})")

    # Aggregate
    total = len(all_results)
    errors = sum(1 for r in all_results if r["error"])
    json_correct = sum(1 for r in all_results if r["json_correct"])
    regex_correct = sum(1 for r in all_results if r["regex_correct"])

    valid = total - errors

    print(f"\n{'='*60}")
    print(f"Results ({valid} valid trials, {errors} errors):")
    print(f"  JSON parser accuracy:  {json_correct}/{valid} ({json_correct/valid:.1%})" if valid else "  No valid trials")
    print(f"  Regex parser accuracy: {regex_correct}/{valid} ({regex_correct/valid:.1%})" if valid else "")
    print(f"{'='*60}")

    # Per-scenario breakdown
    print(f"\nPer-scenario breakdown:")
    for scenario in SCENARIOS:
        scenario_results = [r for r in all_results if r["scenario"] == scenario["name"] and not r["error"]]
        if scenario_results:
            j = sum(1 for r in scenario_results if r["json_correct"])
            r = sum(1 for r in scenario_results if r["regex_correct"])
            n = len(scenario_results)
            print(f"  {scenario['name']:<25} JSON: {j}/{n}  Regex: {r}/{n}")

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"exp1_phase2_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump({
            "experiment": "exp1_reviewer_json_phase2",
            "total_trials": total,
            "errors": errors,
            "json_accuracy": json_correct / valid if valid else 0,
            "regex_accuracy": regex_correct / valid if valid else 0,
            "passed": (json_correct / valid >= 0.9) if valid else False,
            "results": all_results,
        }, f, indent=2)

    print(f"\nResults saved: {output_file}")

    passed = (json_correct / valid >= 0.9) if valid else False
    print(f"\nExperiment 1 Phase 2: {'PASSED' if passed else 'FAILED'}")
    return passed


if __name__ == "__main__":
    asyncio.run(run_phase2(repeats=5))
