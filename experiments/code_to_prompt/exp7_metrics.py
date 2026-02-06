"""
Experiment 7: Metrics → Prompt

Tests whether metrics system logic can be prompt-ified:
- 7a: Category detection (keyword matching → Claude classification)
- 7b: Metrics extraction from goal text (regex parsing → Claude JSON extraction)

Test matrix:
- 7a: 10 goal descriptions x 3 repeats = 30 trials
- 7b: 5 goal descriptions x 3 repeats = 15 trials
"""

import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ralph_sdk.metrics import (
    TaskCategory,
    detect_category,
    parse_metrics_from_goal,
)
from ralph_sdk.prompts import METRICS_CATEGORY_DETECTION_PROMPT, METRICS_EXTRACTION_PROMPT
from claude_code_sdk import AssistantMessage, ClaudeCodeOptions, query


# =============================================================================
# 7a: Category Detection Scenarios
# =============================================================================

CATEGORY_SCENARIOS = [
    {
        "name": "algorithm_sort",
        "goal": "Implement a merge sort algorithm that handles large datasets efficiently. Need to optimize for memory usage and support parallel sorting.",
        "expected_category": "algorithm",
    },
    {
        "name": "algorithm_ml",
        "goal": "Build a prediction model using machine learning to forecast sales data. Need high accuracy with XGBoost or similar.",
        "expected_category": "algorithm",
    },
    {
        "name": "web_react",
        "goal": "Create a React dashboard with responsive charts showing user analytics. Must work on mobile and desktop browsers.",
        "expected_category": "web",
    },
    {
        "name": "web_landing",
        "goal": "Build a landing page with HTML/CSS featuring a hero section, pricing table, and contact form. Must be responsive.",
        "expected_category": "web",
    },
    {
        "name": "api_rest",
        "goal": "Build a REST API for user management with CRUD endpoints, JWT authentication, and rate limiting middleware.",
        "expected_category": "api",
    },
    {
        "name": "api_graphql",
        "goal": "Create a GraphQL backend server with database integration for an e-commerce product catalog.",
        "expected_category": "api",
    },
    {
        "name": "cli_tool",
        "goal": "Build a command line tool using Click that processes CSV files and outputs summary statistics to the terminal.",
        "expected_category": "cli",
    },
    {
        "name": "library_sdk",
        "goal": "Create a Python SDK package for interacting with the FooBar API. Should be pip-installable with clean API surface and good documentation.",
        "expected_category": "library",
    },
    {
        "name": "general_refactor",
        "goal": "Refactor the existing codebase to improve maintainability. Clean up dead code, add type hints, and improve error handling.",
        "expected_category": "general",
    },
    {
        "name": "general_ambiguous",
        "goal": "Set up the development environment and write initial project scaffolding with proper configuration files.",
        "expected_category": "general",
    },
]


# =============================================================================
# 7b: Metrics Extraction Scenarios
# =============================================================================

METRICS_EXTRACTION_SCENARIOS = [
    {
        "name": "explicit_metrics",
        "goal": """Build a REST API for user management.

## Success Metrics

### Hard Constraints (must pass)
- **tests_pass**: All unit and integration tests pass
- **type_check**: mypy type checking passes with no errors

### Optimization Targets
| Metric | Target | Priority |
|--------|--------|----------|
| response_time | <= 200ms | HIGH |
| test_coverage | >= 85% | MEDIUM |

### Quality Criteria (AI-evaluated)
- **api_design**: RESTful conventions, consistent naming, proper error codes
- **code_quality**: Clean code, good structure, proper error handling
""",
        "expected_hard": ["tests_pass", "type_check"],
        "expected_soft": ["response_time", "test_coverage"],
        "expected_subjective": ["api_design", "code_quality"],
        "has_custom_metrics": True,
    },
    {
        "name": "partial_metrics",
        "goal": """Build a CLI tool for data processing.

## Success Metrics

### Hard Constraints (must pass)
- **runs**: CLI runs without errors
- **help_works**: --help shows usage information

### Quality Criteria (AI-evaluated)
- **cli_ux**: Clear output, helpful error messages
""",
        "expected_hard": ["runs", "help_works"],
        "expected_soft": [],
        "expected_subjective": ["cli_ux"],
        "has_custom_metrics": True,
    },
    {
        "name": "no_metrics",
        "goal": """Build a simple calculator application.

The calculator should support basic operations: addition, subtraction,
multiplication, and division. Handle edge cases like division by zero.
""",
        "expected_hard": [],
        "expected_soft": [],
        "expected_subjective": [],
        "has_custom_metrics": False,
    },
    {
        "name": "inline_metrics",
        "goal": """Create a web scraper that collects product prices.

Requirements:
- Must handle at least 100 pages per minute
- Error rate should be below 5%
- Must pass all tests
- Code should be well-structured and maintainable

## Success Metrics

### Hard Constraints (must pass)
- **tests_pass**: All tests pass
- **no_runtime_errors**: No unhandled exceptions

### Optimization Targets
| Metric | Target | Priority |
|--------|--------|----------|
| throughput | >= 100 pages/min | HIGH |
| error_rate | <= 5% | MEDIUM |

### Quality Criteria (AI-evaluated)
- **code_quality**: Code readability and maintainability
""",
        "expected_hard": ["tests_pass", "no_runtime_errors"],
        "expected_soft": ["throughput", "error_rate"],
        "expected_subjective": ["code_quality"],
        "has_custom_metrics": True,
    },
    {
        "name": "with_checkpoints",
        "goal": """Build a recommendation engine for an e-commerce platform.

## Success Metrics

### Hard Constraints (must pass)
- **tests_pass**: All tests pass
- **no_runtime_errors**: No runtime errors during inference

### Optimization Targets
| Metric | Target | Priority |
|--------|--------|----------|
| accuracy | >= 90% | HIGH |
| latency | <= 100ms | HIGH |

### Quality Criteria (AI-evaluated)
- **algorithm_elegance**: Clean, efficient implementation
- **code_quality**: Readable and well-structured

### Checkpoints
- After data preprocessing is complete
- After model training is done
- After API integration
""",
        "expected_hard": ["tests_pass", "no_runtime_errors"],
        "expected_soft": ["accuracy", "latency"],
        "expected_subjective": ["algorithm_elegance", "code_quality"],
        "has_custom_metrics": True,
    },
]


def _extract_json_from_text(text: str) -> dict | None:
    """Extract JSON from text output."""
    fence_match = re.search(r"```(?:json)?\s*\n(\{.*?\})\s*\n```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass

    return None


# =============================================================================
# 7a: Category Detection
# =============================================================================

async def run_category_trial(scenario: dict, trial_num: int) -> dict:
    """Run a single category detection trial."""
    prompt = f"Classify this project goal:\n\n{scenario['goal']}"
    text_result = ""

    try:
        async for message in query(
            prompt=prompt,
            options=ClaudeCodeOptions(
                system_prompt=METRICS_CATEGORY_DETECTION_PROMPT,
                max_turns=1,
            ),
        ):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if hasattr(block, "text") and block.text:
                        text_result += block.text
    except Exception as e:
        return {
            "scenario": scenario["name"],
            "trial": trial_num,
            "expected": scenario["expected_category"],
            "got": None,
            "correct": False,
            "json_parsed": False,
            "error": str(e),
        }

    json_obj = _extract_json_from_text(text_result)
    json_parsed = json_obj is not None

    if json_parsed:
        got_category = json_obj.get("category", "unknown").lower()
    else:
        got_category = "parse_failed"

    # Also run the code-based detection for comparison
    code_category = detect_category(scenario["goal"]).value

    correct = got_category == scenario["expected_category"]

    return {
        "scenario": scenario["name"],
        "trial": trial_num,
        "expected": scenario["expected_category"],
        "got_prompt": got_category,
        "got_code": code_category,
        "correct": correct,
        "code_correct": code_category == scenario["expected_category"],
        "json_parsed": json_parsed,
        "raw_text": text_result[:300],
        "error": None,
    }


async def run_exp7a(repeats: int = 3):
    """Run category detection experiment."""
    print(f"\n{'='*60}")
    print(f"Experiment 7a: Category Detection (Prompt vs Code)")
    print(f"Scenarios: {len(CATEGORY_SCENARIOS)} x {repeats} repeats = {len(CATEGORY_SCENARIOS) * repeats} trials")
    print(f"{'='*60}")

    all_results = []

    for scenario in CATEGORY_SCENARIOS:
        print(f"\n--- {scenario['name']} (expected: {scenario['expected_category']}) ---")

        for trial in range(1, repeats + 1):
            print(f"  Trial {trial}/{repeats}...", end=" ", flush=True)
            result = await run_category_trial(scenario, trial)
            all_results.append(result)

            if result["error"]:
                print(f"ERROR: {result['error']}")
            else:
                p_ok = "OK" if result["correct"] else "WRONG"
                c_ok = "OK" if result["code_correct"] else "WRONG"
                print(f"prompt={result['got_prompt']}({p_ok}) code={result['got_code']}({c_ok})")

    # Aggregate
    total = len(all_results)
    errors = sum(1 for r in all_results if r["error"])
    valid = total - errors
    prompt_correct = sum(1 for r in all_results if r["correct"])
    code_correct = sum(1 for r in all_results if r.get("code_correct"))
    json_parsed = sum(1 for r in all_results if r.get("json_parsed") and not r["error"])

    print(f"\n{'='*60}")
    print(f"Results ({valid} valid trials, {errors} errors):")
    if valid:
        print(f"  Prompt accuracy: {prompt_correct}/{valid} ({prompt_correct/valid:.1%})")
        print(f"  Code accuracy:   {code_correct}/{valid} ({code_correct/valid:.1%})")
        print(f"  JSON parse rate: {json_parsed}/{valid} ({json_parsed/valid:.1%})")
    print(f"  Threshold: >= 90%")
    print(f"  PASSED: {'YES' if valid and prompt_correct/valid >= 0.9 else 'NO'}")
    print(f"{'='*60}")

    # Per-scenario breakdown
    print(f"\nPer-scenario breakdown:")
    for scenario in CATEGORY_SCENARIOS:
        s_results = [r for r in all_results if r["scenario"] == scenario["name"] and not r["error"]]
        if s_results:
            pc = sum(1 for r in s_results if r["correct"])
            cc = sum(1 for r in s_results if r["code_correct"])
            n = len(s_results)
            print(f"  {scenario['name']:<25} prompt={pc}/{n}  code={cc}/{n}")

    return {
        "total": total,
        "errors": errors,
        "valid": valid,
        "prompt_correct": prompt_correct,
        "code_correct": code_correct,
        "prompt_accuracy": prompt_correct / valid if valid else 0,
        "code_accuracy": code_correct / valid if valid else 0,
        "passed": (prompt_correct / valid >= 0.9) if valid else False,
        "results": all_results,
    }


# =============================================================================
# 7b: Metrics Extraction
# =============================================================================

async def run_extraction_trial(scenario: dict, trial_num: int) -> dict:
    """Run a single metrics extraction trial."""
    prompt = f"Extract metrics from this project goal:\n\n{scenario['goal']}"
    text_result = ""

    try:
        async for message in query(
            prompt=prompt,
            options=ClaudeCodeOptions(
                system_prompt=METRICS_EXTRACTION_PROMPT,
                max_turns=1,
            ),
        ):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if hasattr(block, "text") and block.text:
                        text_result += block.text
    except Exception as e:
        return {
            "scenario": scenario["name"],
            "trial": trial_num,
            "correct": False,
            "json_parsed": False,
            "error": str(e),
        }

    json_obj = _extract_json_from_text(text_result)
    json_parsed = json_obj is not None

    if not json_parsed:
        return {
            "scenario": scenario["name"],
            "trial": trial_num,
            "correct": False,
            "json_parsed": False,
            "raw_text": text_result[:300],
            "error": "JSON parse failed",
        }

    # Check has_custom_metrics
    got_has_custom = json_obj.get("has_custom_metrics", False)
    expected_has_custom = scenario["has_custom_metrics"]
    has_custom_correct = got_has_custom == expected_has_custom

    # Check hard constraints
    got_hard = sorted([m.get("name", "") for m in json_obj.get("hard_constraints", [])])
    expected_hard = sorted(scenario["expected_hard"])
    hard_correct = got_hard == expected_hard

    # Check soft targets
    got_soft = sorted([m.get("name", "") for m in json_obj.get("soft_targets", [])])
    expected_soft = sorted(scenario["expected_soft"])
    soft_correct = got_soft == expected_soft

    # Check subjective criteria
    got_subj = sorted([m.get("name", "") for m in json_obj.get("subjective_criteria", [])])
    expected_subj = sorted(scenario["expected_subjective"])
    subj_correct = got_subj == expected_subj

    # Overall correctness: all parts must match
    all_correct = has_custom_correct and hard_correct and soft_correct and subj_correct

    # Also run code-based extraction for comparison
    code_result = parse_metrics_from_goal(scenario["goal"])

    return {
        "scenario": scenario["name"],
        "trial": trial_num,
        "correct": all_correct,
        "has_custom_correct": has_custom_correct,
        "hard_correct": hard_correct,
        "soft_correct": soft_correct,
        "subj_correct": subj_correct,
        "json_parsed": True,
        "got": {
            "has_custom": got_has_custom,
            "hard": got_hard,
            "soft": got_soft,
            "subjective": got_subj,
        },
        "expected": {
            "has_custom": expected_has_custom,
            "hard": expected_hard,
            "soft": expected_soft,
            "subjective": expected_subj,
        },
        "code_found_metrics": code_result is not None,
        "raw_text": text_result[:300],
        "error": None,
    }


async def run_exp7b(repeats: int = 3):
    """Run metrics extraction experiment."""
    print(f"\n{'='*60}")
    print(f"Experiment 7b: Metrics Extraction (Prompt vs Regex)")
    print(f"Scenarios: {len(METRICS_EXTRACTION_SCENARIOS)} x {repeats} repeats = {len(METRICS_EXTRACTION_SCENARIOS) * repeats} trials")
    print(f"{'='*60}")

    all_results = []

    for scenario in METRICS_EXTRACTION_SCENARIOS:
        print(f"\n--- {scenario['name']} (has_custom={scenario['has_custom_metrics']}) ---")

        for trial in range(1, repeats + 1):
            print(f"  Trial {trial}/{repeats}...", end=" ", flush=True)
            result = await run_extraction_trial(scenario, trial)
            all_results.append(result)

            if result["error"]:
                print(f"ERROR: {result['error']}")
            else:
                ok = "OK" if result["correct"] else "WRONG"
                parts = []
                if not result.get("has_custom_correct", True):
                    parts.append("has_custom")
                if not result.get("hard_correct", True):
                    parts.append("hard")
                if not result.get("soft_correct", True):
                    parts.append("soft")
                if not result.get("subj_correct", True):
                    parts.append("subj")
                detail = f" (failed: {', '.join(parts)})" if parts else ""
                print(f"{ok}{detail}")

    # Aggregate
    total = len(all_results)
    errors = sum(1 for r in all_results if r["error"])
    valid = total - errors
    correct = sum(1 for r in all_results if r["correct"])
    json_parsed = sum(1 for r in all_results if r.get("json_parsed") and not r["error"])

    # Sub-metric accuracy
    has_custom_correct = sum(1 for r in all_results if r.get("has_custom_correct") and not r["error"])
    hard_correct = sum(1 for r in all_results if r.get("hard_correct") and not r["error"])
    soft_correct = sum(1 for r in all_results if r.get("soft_correct") and not r["error"])
    subj_correct = sum(1 for r in all_results if r.get("subj_correct") and not r["error"])

    print(f"\n{'='*60}")
    print(f"Results ({valid} valid trials, {errors} errors):")
    if valid:
        print(f"  Overall accuracy:     {correct}/{valid} ({correct/valid:.1%})")
        print(f"  has_custom accuracy:  {has_custom_correct}/{valid} ({has_custom_correct/valid:.1%})")
        print(f"  hard metrics:         {hard_correct}/{valid} ({hard_correct/valid:.1%})")
        print(f"  soft metrics:         {soft_correct}/{valid} ({soft_correct/valid:.1%})")
        print(f"  subjective metrics:   {subj_correct}/{valid} ({subj_correct/valid:.1%})")
        print(f"  JSON parse rate:      {json_parsed}/{valid} ({json_parsed/valid:.1%})")
    print(f"  Threshold: >= 85%")
    print(f"  PASSED: {'YES' if valid and correct/valid >= 0.85 else 'NO'}")
    print(f"{'='*60}")

    # Per-scenario breakdown
    print(f"\nPer-scenario breakdown:")
    for scenario in METRICS_EXTRACTION_SCENARIOS:
        s_results = [r for r in all_results if r["scenario"] == scenario["name"] and not r["error"]]
        if s_results:
            c = sum(1 for r in s_results if r["correct"])
            n = len(s_results)
            print(f"  {scenario['name']:<25} {c}/{n}")

    return {
        "total": total,
        "errors": errors,
        "valid": valid,
        "correct": correct,
        "accuracy": correct / valid if valid else 0,
        "passed": (correct / valid >= 0.85) if valid else False,
        "sub_accuracies": {
            "has_custom": has_custom_correct / valid if valid else 0,
            "hard": hard_correct / valid if valid else 0,
            "soft": soft_correct / valid if valid else 0,
            "subjective": subj_correct / valid if valid else 0,
        },
        "results": all_results,
    }


async def main():
    """Run both 7a and 7b experiments."""
    print("=" * 70)
    print("Experiment 7: Metrics → Prompt")
    print("=" * 70)

    # 7a: Category Detection
    results_7a = await run_exp7a(repeats=3)

    # 7b: Metrics Extraction
    results_7b = await run_exp7b(repeats=3)

    # Summary
    print(f"\n{'='*70}")
    print("EXPERIMENT 7 SUMMARY")
    print(f"{'='*70}")
    print(f"  7a Category Detection: {results_7a['prompt_accuracy']:.1%} (code: {results_7a['code_accuracy']:.1%})")
    print(f"     PASSED: {'YES' if results_7a['passed'] else 'NO'}")
    print(f"  7b Metrics Extraction: {results_7b['accuracy']:.1%}")
    print(f"     PASSED: {'YES' if results_7b['passed'] else 'NO'}")

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"exp7_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump({
            "experiment": "exp7_metrics",
            "7a_category_detection": {
                "prompt_accuracy": results_7a["prompt_accuracy"],
                "code_accuracy": results_7a["code_accuracy"],
                "passed": results_7a["passed"],
                "results": results_7a["results"],
            },
            "7b_metrics_extraction": {
                "accuracy": results_7b["accuracy"],
                "sub_accuracies": results_7b["sub_accuracies"],
                "passed": results_7b["passed"],
                "results": results_7b["results"],
            },
        }, f, indent=2, default=str)

    print(f"\nResults saved: {output_file}")
    return results_7a["passed"] and results_7b["passed"]


if __name__ == "__main__":
    asyncio.run(main())
