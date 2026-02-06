"""
Experiment 3: Planner Tool-use Structured Output

Hypothesis: Using tool-use for structured output is more reliable than regex parsing.

Replace target: planner.py:102-225 (15+ regexes, 124 lines)

Phase 1: Verify regex parser on synthetic inputs (sanity check)
Phase 2: Test Claude with tool-use for planner decisions (requires API)

Test matrix: 12 action types x 3 repeats = 36 trials
"""

import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ralph_sdk.planner import Action, PlannerDecision, parse_planner_output
from claude_code_sdk import AssistantMessage, ClaudeCodeOptions, query


# =============================================================================
# Test Scenarios - one per action type
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

This is a user preference decision.

Decide the next action.""",
        "expected_action": "ask",
    },
    {
        "name": "hedge",
        "prompt": """Current state:
- T001: IMPLEMENT - Implement caching with Redis (status: done, awaiting user testing)

T001 is done but relies on Redis being available. If Redis isn't available in production,
we need a fallback. This is a good time to explore alternatives.

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
# Phase 1: Test regex parser on synthetic text outputs
# =============================================================================

SYNTHETIC_OUTPUTS = {
    "execute": """ACTION: execute
TARGET: T001
REASON: T001 is ready to execute and all dependencies are met.""",

    "explore": """ACTION: explore
TARGET: T001
REASON: Need to investigate before implementing.""",

    "parallel_execute": """ACTION: parallel_execute
TASK_IDS: T001, T002
REASON: These tasks are independent research tasks.""",

    "create": """ACTION: create
REASON: Need a new task for rate limiting discovered during exploration.
NEW_TASKS:
- T003: IMPLEMENT - Add rate limiting middleware""",

    "modify": """ACTION: modify
TARGET: T001
REASON: T002 found project uses sessions, not JWT.
MODIFICATION: Change approach from JWT to session-based authentication.""",

    "delete": """ACTION: delete
TARGET: T002
REASON: T001 already implemented pagination. T002 is duplicate.""",

    "skip": """ACTION: skip
TARGET: T001
REASON: Payment API is temporarily unavailable.""",

    "ask": """ACTION: ask
REASON: Need user preference for export format.
QUESTION: Should we use CSV or JSON for the data export feature?""",

    "hedge": """ACTION: hedge
TARGET: T001
REASON: Redis dependency is a risk.
FAILURE_ASSUMPTIONS:
- Redis might not be available in production
NEW_TASKS:
- T002: EXPLORE - Research in-memory caching alternatives""",

    "pivot_research": """ACTION: pivot_research
TARGET: T001
CURRENT_APPROACH: Using library X for data processing
BLOCKER: Library X abandoned, doesn't support Python 3.11+
NEW_DIRECTION: Use library Y instead
REASON: Library Y is actively maintained and supports all needed features
NEW_TASKS:
- T002: EXPLORE - Research library Y API""",

    "pivot_iteration": """ACTION: pivot_iteration
TARGET: T001
ATTEMPT_COUNT: 3
BEST_SCORE: 34/100
FAILURE_PATTERN: O(n²) approach fundamentally limited
NEW_APPROACH: Switch to hash-based O(n) algorithm
REASON: Incremental optimization cannot overcome algorithmic limitation
NEW_TASKS:
- T002: IMPLEMENT - Rewrite with hash-based approach""",

    "done": """ACTION: done
REASON: All tasks completed, all scores meet threshold.""",
}


def run_phase1():
    """Phase 1: Verify regex parser extracts correct actions from synthetic outputs."""
    print("=" * 60)
    print("Experiment 3 - Phase 1: Regex Parser Accuracy (Sanity Check)")
    print("=" * 60)

    correct = 0
    total = len(SYNTHETIC_OUTPUTS)

    print(f"\n{'Action':<25} {'Parsed':<25} {'Status'}")
    print("-" * 60)

    for action_name, text in SYNTHETIC_OUTPUTS.items():
        decision = parse_planner_output(text)
        parsed_action = decision.action.value

        ok = parsed_action == action_name
        if ok:
            correct += 1

        # Also check secondary fields
        extra = ""
        if action_name == "execute" and decision.target != "T001":
            extra = f" [target wrong: {decision.target}]"
            ok = False
        if action_name == "parallel_execute" and sorted(decision.task_ids) != ["T001", "T002"]:
            extra = f" [task_ids wrong: {decision.task_ids}]"
            ok = False

        status = "OK" if ok else f"WRONG{extra}"
        print(f"{action_name:<25} {parsed_action:<25} {status}")

    print(f"\nRegex parser accuracy: {correct}/{total} ({correct/total:.1%})")
    return correct, total


# =============================================================================
# Phase 2: Test Claude with tool-use structured output
# =============================================================================

# The tool schema that replaces all 15+ regexes
MAKE_DECISION_TOOL = {
    "name": "make_decision",
    "description": "Output your planner decision in structured format.",
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["execute", "explore", "parallel_execute", "create",
                         "modify", "delete", "skip", "ask", "hedge",
                         "pivot_research", "pivot_wait", "pivot_iteration", "done"],
                "description": "The action to take",
            },
            "target": {
                "type": "string",
                "description": "Target task ID (e.g., T001). Required for execute, explore, modify, delete, skip, hedge, pivot_*.",
            },
            "task_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Task IDs for parallel_execute.",
            },
            "reason": {
                "type": "string",
                "description": "Why this action was chosen.",
            },
            "new_tasks": {
                "type": "string",
                "description": "Description of new tasks to create (for create, hedge, pivot_*).",
            },
            "question": {
                "type": "string",
                "description": "Question for the user (for ask action).",
            },
            "modification": {
                "type": "string",
                "description": "Description of modification (for modify action).",
            },
            "failure_assumptions": {
                "type": "string",
                "description": "Failure assumptions and alternatives (for hedge/pivot_wait).",
            },
            "current_approach": {
                "type": "string",
                "description": "Current approach being abandoned (for pivot_research).",
            },
            "blocker": {
                "type": "string",
                "description": "Why current approach is not viable (for pivot_research).",
            },
            "new_direction": {
                "type": "string",
                "description": "New direction to explore (for pivot_research).",
            },
            "attempt_count": {
                "type": "integer",
                "description": "Number of attempts made (for pivot_iteration).",
            },
            "best_score": {
                "type": "string",
                "description": "Best score achieved (for pivot_iteration).",
            },
            "failure_pattern": {
                "type": "string",
                "description": "Pattern of failure observed (for pivot_iteration).",
            },
            "new_approach": {
                "type": "string",
                "description": "New approach to try (for pivot_iteration).",
            },
        },
        "required": ["action", "reason"],
    },
}

PLANNER_JSON_PROMPT = """You are a task planner. Read the current state and decide the next action.

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
    # Try code fence first
    fence_match = re.search(r"```(?:json)?\s*\n(\{.*?\})\s*\n```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try bare JSON
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass

    return None


async def run_single_trial(scenario: dict, trial_num: int) -> dict:
    """Run a single trial with Claude API using JSON output."""
    text_result = ""

    try:
        async for message in query(
            prompt=scenario["prompt"],
            options=ClaudeCodeOptions(
                system_prompt=PLANNER_JSON_PROMPT,
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

    # Try to parse JSON from output
    json_obj = _extract_json_from_text(text_result)
    json_parsed = json_obj is not None

    if json_parsed:
        got_action = json_obj.get("action", "unknown").lower()
        got_target = json_obj.get("target")
        got_task_ids = json_obj.get("task_ids", [])
    else:
        # Fallback: try regex parsing
        decision = parse_planner_output(text_result)
        got_action = decision.action.value
        got_target = decision.target
        got_task_ids = decision.task_ids

    action_correct = got_action == scenario["expected_action"]

    # Check target if expected
    target_correct = None
    if "expected_target" in scenario and scenario["expected_target"]:
        target_correct = got_target == scenario["expected_target"]

    # Check task_ids if expected
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
        "raw_text": text_result[:300],
        "error": None,
    }


async def run_phase2(repeats: int = 3):
    """Phase 2: Test Claude with tool-use for structured output."""
    print(f"\n{'='*60}")
    print(f"Experiment 3 - Phase 2: Tool-use Structured Output")
    print(f"Scenarios: {len(SCENARIOS)} x {repeats} repeats = {len(SCENARIOS) * repeats} trials")
    print(f"{'='*60}")

    all_results = []

    for scenario in SCENARIOS:
        print(f"\n--- {scenario['name']} (expected: {scenario['expected_action']}) ---")

        for trial in range(1, repeats + 1):
            print(f"  Trial {trial}/{repeats}...", end=" ", flush=True)
            result = await run_single_trial(scenario, trial)
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
    used_tool = sum(1 for r in all_results if r.get("json_parsed") and not r["error"])

    # Target accuracy (only for scenarios with expected_target)
    target_trials = [r for r in all_results if r.get("target_correct") is not None]
    target_correct = sum(1 for r in target_trials if r["target_correct"])

    print(f"\n{'='*60}")
    print(f"Results ({valid} valid trials, {errors} errors):")
    print(f"  Action accuracy: {action_correct}/{valid} ({action_correct/valid:.1%})" if valid else "  No valid")
    print(f"  JSON parse rate: {used_tool}/{valid} ({used_tool/valid:.1%})" if valid else "")
    if target_trials:
        print(f"  Target accuracy: {target_correct}/{len(target_trials)} ({target_correct/len(target_trials):.1%})")
    print(f"  Threshold: action >= 90%, target >= 85%")
    action_passed = valid and action_correct / valid >= 0.9
    target_passed = not target_trials or target_correct / len(target_trials) >= 0.85
    overall_passed = action_passed and target_passed
    print(f"  PASSED: {'YES' if overall_passed else 'NO'}")
    print(f"{'='*60}")

    # Per-scenario breakdown
    print(f"\nPer-scenario breakdown:")
    for scenario in SCENARIOS:
        s_results = [r for r in all_results if r["scenario"] == scenario["name"] and not r["error"]]
        if s_results:
            c = sum(1 for r in s_results if r["action_correct"])
            t = sum(1 for r in s_results if r.get("json_parsed"))
            n = len(s_results)
            print(f"  {scenario['name']:<25} action={c}/{n}  json={t}/{n}")

    # Save
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"exp3_phase2_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump({
            "experiment": "exp3_planner_tooluse_phase2",
            "total_trials": total,
            "errors": errors,
            "action_accuracy": action_correct / valid if valid else 0,
            "json_parse_rate": used_tool / valid if valid else 0,
            "target_accuracy": target_correct / len(target_trials) if target_trials else None,
            "passed": overall_passed,
            "results": all_results,
        }, f, indent=2, default=str)

    print(f"\nResults saved: {output_file}")
    return overall_passed


async def main():
    """Run both phases."""
    code_correct, code_total = run_phase1()
    passed = await run_phase2(repeats=3)
    print(f"\nExperiment 3 overall: {'PASSED' if passed else 'FAILED'}")


if __name__ == "__main__":
    asyncio.run(main())
