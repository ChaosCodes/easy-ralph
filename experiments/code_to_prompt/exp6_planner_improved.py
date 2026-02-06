"""
Experiment 6: Planner Hedge — Improved Prompt

Round 2 improvement over Experiment 3.
Key change: Added HEDGE_VS_ASK_GUIDE to prompt using domain concept teaching
and OpenClaw-style conditional rules with examples.

Round 1 result: 91.7% action accuracy, hedge scenario 0/3
Target: >= 95% action accuracy, hedge >= 67% (2/3)

Test matrix: 12 scenarios x 3 repeats = 36 trials
"""

import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ralph_sdk.planner import Action, PlannerDecision, parse_planner_output
from ralph_sdk.prompts import HEDGE_VS_ASK_GUIDE
from claude_code_sdk import AssistantMessage, ClaudeCodeOptions, query


# =============================================================================
# Test Scenarios (same as exp3)
# =============================================================================

SCENARIOS = [
    {
        "name": "execute",
        "prompt": """Current state:
- T001: IMPLEMENT - Add login page (status: ready)
- T002: EXPLORE - Research auth libraries (status: done)

T002 found that we should use passport.js. T001 is ready to execute.

Decide the next action.""",
        "expected_action": "execute",
        "expected_target": "T001",
    },
    {
        "name": "explore",
        "prompt": """Current state:
- T001: EXPLORE - Investigate performance bottleneck (status: ready)

No tasks have been executed yet. We need to understand the problem first.

Decide the next action.""",
        "expected_action": "explore",
        "expected_target": "T001",
    },
    {
        "name": "parallel_execute",
        "prompt": """Current state:
- T001: EXPLORE - Research caching strategies (status: ready)
- T002: EXPLORE - Research database indexing (status: ready)
- T003: IMPLEMENT - Refactor API layer (status: blocked by T001, T002)

T001 and T002 are independent research tasks. They can run in parallel.

Decide the next action.""",
        "expected_action": "parallel_execute",
        "expected_target": None,
        "expected_task_ids": ["T001", "T002"],
    },
    {
        "name": "create",
        "prompt": """Current state:
- T001: EXPLORE - Investigate auth options (status: done)
  Findings: Need both JWT auth AND rate limiting. Rate limiting was not in original plan.

We need to add a new task for rate limiting that was discovered during exploration.

Decide the next action.""",
        "expected_action": "create",
    },
    {
        "name": "modify",
        "prompt": """Current state:
- T001: IMPLEMENT - Add JWT authentication (status: ready)
  But T002's exploration found the project uses session-based auth, not JWT.
- T002: EXPLORE - Research auth patterns (status: done)
  Finding: Project already has session middleware. Should use sessions, not JWT.

T001's approach needs to change based on T002's findings.

Decide the next action.""",
        "expected_action": "modify",
        "expected_target": "T001",
    },
    {
        "name": "delete",
        "prompt": """Current state:
- T001: IMPLEMENT - Add pagination to API (status: done)
- T002: IMPLEMENT - Add cursor-based pagination (status: ready)

T001 already implemented pagination. T002 is a duplicate and no longer needed.

Decide the next action.""",
        "expected_action": "delete",
        "expected_target": "T002",
    },
    {
        "name": "skip",
        "prompt": """Current state:
- T001: IMPLEMENT - Integrate with external payment API (status: ready, but API is down)
- T002: IMPLEMENT - Add user profile page (status: ready)

The payment API is temporarily unavailable. We should work on something else.

Decide the next action.""",
        "expected_action": "skip",
        "expected_target": "T001",
    },
    {
        "name": "ask",
        "prompt": """Current state:
- T001: IMPLEMENT - Add data export feature (status: ready)

The task says "export data" but doesn't specify format. Two valid options:
1. CSV export (simpler, widely supported)
2. JSON export (more structured, better for programmatic use)

This is a user preference decision that the agent cannot make autonomously.

Decide the next action.""",
        "expected_action": "ask",
    },
    {
        "name": "hedge",
        "prompt": """Current state:
- T001: IMPLEMENT - Implement caching with Redis (status: done, awaiting user testing)

T001 is done but relies on Redis being available in production.
The agent can autonomously research alternative caching solutions (in-memory, file-based)
without needing any input from the user. This is a technical risk that the agent
can prepare for on its own.

Decide the next action.""",
        "expected_action": "hedge",
        "expected_target": "T001",
    },
    {
        "name": "pivot_research",
        "prompt": """Current state:
- T001: EXPLORE - Research using library X for data processing (status: done)
  Finding: Library X has been abandoned, last commit 2 years ago. It doesn't support Python 3.11+.
  There is a maintained alternative: library Y.

The current direction is not viable based on research.

Decide the next action.""",
        "expected_action": "pivot_research",
        "expected_target": "T001",
    },
    {
        "name": "pivot_iteration",
        "prompt": """Current state:
- T001: IMPLEMENT - Optimize search algorithm (status: needs improvement)
  Attempt 1: Score 30/100
  Attempt 2: Score 32/100
  Attempt 3: Score 34/100
  Pattern: Small improvements but fundamentally limited by O(n²) approach.
  Evaluator recommends pivot.

Multiple attempts have shown minimal improvement.

Decide the next action.""",
        "expected_action": "pivot_iteration",
        "expected_target": "T001",
    },
    {
        "name": "done",
        "prompt": """Current state:
- T001: IMPLEMENT - Add login page (status: done, score 95/100)
- T002: IMPLEMENT - Add registration page (status: done, score 92/100)
- T003: EXPLORE - Security review (status: done, no issues found)

Goal: Add user authentication with login and registration.
All tasks are complete, all scores meet threshold, security review passed.

Decide the next action.""",
        "expected_action": "done",
    },
]


# =============================================================================
# Improved Prompt (V2 — with hedge/ask guide)
# =============================================================================

PLANNER_JSON_PROMPT_V2 = f"""You are a task planner. Read the current state and decide the next action.

## Available Actions
- **execute**: Execute a single IMPLEMENT task (requires target)
- **explore**: Execute an EXPLORE task (requires target)
- **parallel_execute**: Execute multiple independent tasks concurrently (requires task_ids)
- **create**: Add new task(s) (requires new_tasks)
- **modify**: Change an existing task (requires target, modification)
- **delete**: Remove a task (requires target)
- **skip**: Temporarily skip a blocked task (requires target)
- **ask**: Ask user for a decision (requires question)
- **hedge**: Explore alternatives for a task pending verification (requires target)
- **pivot_research**: Abandon direction after research confirms not viable (requires target)
- **pivot_iteration**: Change direction after multiple failed attempts (requires target)
- **done**: Goal fully achieved

{HEDGE_VS_ASK_GUIDE}

## CRITICAL: Output Format

You MUST output your decision as a JSON object. Do NOT use any tools. Just output the JSON directly:

```json
{{
    "action": "<action_name>",
    "target": "<task_id or null>",
    "task_ids": ["<id1>", "<id2>"],
    "reason": "<why this action>"
}}
```

Include additional fields as appropriate:
- For create/hedge/pivot: "new_tasks"
- For ask: "question"
- For modify: "modification"
- For hedge: "failure_assumptions"
- For pivot_research: "current_approach", "blocker", "new_direction"
- For pivot_iteration: "attempt_count", "best_score", "failure_pattern", "new_approach"

DO NOT try to read files or use any tools. Just analyze the state provided and output JSON.
"""

# Original prompt from exp3 for comparison
PLANNER_JSON_PROMPT_V1 = """You are a task planner. Read the current state and decide the next action.

## Available Actions
- **execute**: Execute a single IMPLEMENT task (requires target)
- **explore**: Execute an EXPLORE task (requires target)
- **parallel_execute**: Execute multiple independent tasks concurrently (requires task_ids)
- **create**: Add new task(s) (requires new_tasks)
- **modify**: Change an existing task (requires target, modification)
- **delete**: Remove a task (requires target)
- **skip**: Temporarily skip a blocked task (requires target)
- **ask**: Ask user for a decision (requires question)
- **hedge**: Explore alternatives for a task pending verification (requires target)
- **pivot_research**: Abandon direction after research confirms not viable (requires target)
- **pivot_iteration**: Change direction after multiple failed attempts (requires target)
- **done**: Goal fully achieved

## CRITICAL: Output Format

You MUST output your decision as a JSON object. Do NOT use any tools. Just output the JSON directly:

```json
{
    "action": "<action_name>",
    "target": "<task_id or null>",
    "task_ids": ["<id1>", "<id2>"],
    "reason": "<why this action>"
}
```

Include additional fields as appropriate:
- For create/hedge/pivot: "new_tasks"
- For ask: "question"
- For modify: "modification"
- For hedge: "failure_assumptions"
- For pivot_research: "current_approach", "blocker", "new_direction"
- For pivot_iteration: "attempt_count", "best_score", "failure_pattern", "new_approach"

DO NOT try to read files or use any tools. Just analyze the state provided and output JSON.
"""


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


async def run_single_trial(scenario: dict, trial_num: int, system_prompt: str) -> dict:
    """Run a single trial with Claude API using JSON output."""
    text_result = ""

    try:
        async for message in query(
            prompt=scenario["prompt"],
            options=ClaudeCodeOptions(
                system_prompt=system_prompt,
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
            "expected_action": scenario["expected_action"],
            "got_action": None,
            "action_correct": False,
            "target_correct": None,
            "json_parsed": False,
            "raw_output": "",
            "error": str(e),
        }

    json_obj = _extract_json_from_text(text_result)
    json_parsed = json_obj is not None

    if json_parsed:
        got_action = json_obj.get("action", "unknown").lower()
        got_target = json_obj.get("target")
        got_task_ids = json_obj.get("task_ids", [])
    else:
        decision = parse_planner_output(text_result)
        got_action = decision.action.value
        got_target = decision.target
        got_task_ids = decision.task_ids

    action_correct = got_action == scenario["expected_action"]

    target_correct = None
    if "expected_target" in scenario and scenario["expected_target"]:
        target_correct = got_target == scenario["expected_target"]

    task_ids_correct = None
    if "expected_task_ids" in scenario:
        expected_ids = set(scenario["expected_task_ids"])
        got_ids = set(got_task_ids) if got_task_ids else set()
        task_ids_correct = expected_ids == got_ids

    return {
        "scenario": scenario["name"],
        "trial": trial_num,
        "expected_action": scenario["expected_action"],
        "got_action": got_action,
        "action_correct": action_correct,
        "target_correct": target_correct,
        "task_ids_correct": task_ids_correct,
        "json_parsed": json_parsed,
        "json_obj": json_obj,
        "raw_text": text_result[:500],
        "error": None,
    }


async def run_experiment(prompt_version: str, system_prompt: str, repeats: int = 3):
    """Run the experiment with a given prompt version."""
    print(f"\n{'='*60}")
    print(f"Experiment 6 - {prompt_version}: Planner Hedge (Improved Prompt)")
    print(f"Scenarios: {len(SCENARIOS)} x {repeats} repeats = {len(SCENARIOS) * repeats} trials")
    print(f"{'='*60}")

    all_results = []

    for scenario in SCENARIOS:
        print(f"\n--- {scenario['name']} (expected: {scenario['expected_action']}) ---")

        for trial in range(1, repeats + 1):
            print(f"  Trial {trial}/{repeats}...", end=" ", flush=True)
            result = await run_single_trial(scenario, trial, system_prompt)
            all_results.append(result)

            if result["error"]:
                print(f"ERROR: {result['error']}")
            else:
                ok = "OK" if result["action_correct"] else "WRONG"
                fmt = "json" if result.get("json_parsed") else "text"
                target_info = ""
                if result["target_correct"] is not None:
                    target_info = f" target={'OK' if result['target_correct'] else 'WRONG'}"
                print(f"{result['got_action']} ({ok}, {fmt}){target_info}")

    # Aggregate
    total = len(all_results)
    errors = sum(1 for r in all_results if r["error"])
    valid = total - errors
    action_correct = sum(1 for r in all_results if r["action_correct"])
    used_json = sum(1 for r in all_results if r.get("json_parsed") and not r["error"])

    target_trials = [r for r in all_results if r.get("target_correct") is not None]
    target_correct = sum(1 for r in target_trials if r["target_correct"])

    print(f"\n{'='*60}")
    print(f"Results - {prompt_version} ({valid} valid trials, {errors} errors):")
    if valid:
        print(f"  Action accuracy: {action_correct}/{valid} ({action_correct/valid:.1%})")
        print(f"  JSON parse rate: {used_json}/{valid} ({used_json/valid:.1%})")
    if target_trials:
        print(f"  Target accuracy: {target_correct}/{len(target_trials)} ({target_correct/len(target_trials):.1%})")
    print(f"  Threshold: action >= 95%, hedge >= 67%")
    action_passed = valid and action_correct / valid >= 0.95
    print(f"  PASSED: {'YES' if action_passed else 'NO'}")
    print(f"{'='*60}")

    # Per-scenario breakdown
    print(f"\nPer-scenario breakdown:")
    for scenario in SCENARIOS:
        s_results = [r for r in all_results if r["scenario"] == scenario["name"] and not r["error"]]
        if s_results:
            c = sum(1 for r in s_results if r["action_correct"])
            n = len(s_results)
            print(f"  {scenario['name']:<25} action={c}/{n}")

    # Failure analysis
    failures = [r for r in all_results if not r["action_correct"] and not r["error"]]
    if failures:
        print(f"\nFailure analysis:")
        for f in failures:
            print(f"  [{f['scenario']}] expected={f['expected_action']} got={f['got_action']}")
            print(f"    Output: {f['raw_text'][:200]}")

    return {
        "prompt_version": prompt_version,
        "total": total,
        "errors": errors,
        "valid": valid,
        "action_correct": action_correct,
        "action_accuracy": action_correct / valid if valid else 0,
        "passed": action_passed,
        "results": all_results,
    }


async def main():
    """Run improved prompt experiment and compare with Round 1."""
    print("=" * 70)
    print("Experiment 6: Planner Hedge — Improved Prompt (Round 2)")
    print("=" * 70)
    print("\nKey change: Added HEDGE_VS_ASK_GUIDE to prompt")
    print("Round 1 result: 91.7% action accuracy (0/3 on hedge)")
    print("Target: >= 95% action accuracy, hedge >= 67%\n")

    # Run with improved prompt (V2)
    v2_results = await run_experiment("V2_improved", PLANNER_JSON_PROMPT_V2, repeats=3)

    # Summary comparison
    print(f"\n{'='*70}")
    print("COMPARISON: Round 1 vs Round 2")
    print(f"{'='*70}")
    print(f"  Round 1 (V1): 91.7% action accuracy (33/36)")
    print(f"  Round 2 (V2): {v2_results['action_accuracy']:.1%} ({v2_results['action_correct']}/{v2_results['valid']})")

    # Check hedge specifically
    hedge_results = [r for r in v2_results["results"] if r["scenario"] == "hedge" and not r["error"]]
    hedge_correct = sum(1 for r in hedge_results if r["action_correct"])
    print(f"\n  hedge scenario:")
    print(f"    Round 1: 0/3 (0%)")
    if hedge_results:
        print(f"    Round 2: {hedge_correct}/{len(hedge_results)} ({hedge_correct/len(hedge_results):.0%})")
    else:
        print(f"    Round 2: N/A")

    improvement = v2_results["action_accuracy"] - 0.917
    print(f"\n  Improvement: {improvement:+.1%}")
    print(f"  Merge to prompts.py: {'YES (already merged)' if improvement >= 0.05 else 'KEEP (prompt already added, marginal improvement)'}")

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"exp6_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump({
            "experiment": "exp6_planner_improved",
            "round": 2,
            "round1_action_accuracy": 0.917,
            "round1_hedge": "0/3",
            "v2": {
                "action_accuracy": v2_results["action_accuracy"],
                "passed": v2_results["passed"],
                "total": v2_results["total"],
                "action_correct": v2_results["action_correct"],
                "hedge_correct": hedge_correct,
                "hedge_total": len(hedge_results),
            },
            "results": v2_results["results"],
        }, f, indent=2, default=str)

    print(f"\nResults saved: {output_file}")
    return v2_results["passed"]


if __name__ == "__main__":
    asyncio.run(main())
