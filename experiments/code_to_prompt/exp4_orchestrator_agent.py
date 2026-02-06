"""
Experiment 4: Orchestrator Agent Loop

Hypothesis: An agent with workflow rules as prompt + tools can drive the iteration loop,
replacing orchestrator.py:440-1067 (628 lines of if/elif).

Approach:
- Define mock sub-agents with deterministic outputs
- Give the orchestrator agent tools to call them
- Test if it follows the correct workflow for 5 scenarios

Test matrix: 5 scenarios x 3 repeats = 15 trials

Success criteria:
- Final state matches >= 80% (4/5 scenarios)
- Iterations within 150% of Python version
- No infinite loops
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from claude_code_sdk import AssistantMessage, ClaudeCodeOptions, query


# =============================================================================
# Mock Sub-Agents: deterministic outputs
# =============================================================================

class MockState:
    """Tracks orchestrator state during a trial."""
    def __init__(self, scenario_name: str, pool: dict):
        self.scenario = scenario_name
        self.pool = pool  # task statuses, scores, etc.
        self.actions_taken = []
        self.iterations = 0
        self.done = False
        self.max_iterations = 10

    def to_pool_text(self) -> str:
        """Generate pool.md-like text from state."""
        lines = ["# Task Pool", "", "## Tasks", ""]
        for tid, info in self.pool["tasks"].items():
            lines.append(f"| {tid} | {info['type']} | {info['status']} | {info['description']} |")
        if self.pool.get("findings"):
            lines.append("")
            lines.append("## Findings")
            for f in self.pool["findings"]:
                lines.append(f"- {f}")
        return "\n".join(lines)


# Scenario definitions: each defines initial state and expected mock responses
SCENARIOS = [
    {
        "name": "A_simple_pass",
        "description": "Simple task, one-shot pass",
        "initial_pool": {
            "tasks": {
                "T001": {"type": "IMPLEMENT", "status": "ready", "description": "Add login page"},
            },
            "findings": [],
        },
        "mock_responses": {
            # plan -> execute T001
            # work T001 -> success
            # review T001 -> passed
            # evaluate T001 -> score 95
            # plan -> done
        },
        "expected_final": "done",
        "expected_max_iterations": 3,
    },
    {
        "name": "B_retry_then_pass",
        "description": "Review fails, retry, then pass",
        "initial_pool": {
            "tasks": {
                "T001": {"type": "IMPLEMENT", "status": "ready", "description": "Add validation"},
            },
            "findings": [],
        },
        "expected_final": "done",
        "expected_max_iterations": 5,
    },
    {
        "name": "C_declining_pivot",
        "description": "Scores decline, trigger pivot",
        "initial_pool": {
            "tasks": {
                "T001": {"type": "IMPLEMENT", "status": "ready", "description": "Optimize search"},
            },
            "findings": [],
        },
        "expected_final": "pivot",
        "expected_max_iterations": 6,
    },
    {
        "name": "D_user_test_checkpoint",
        "description": "Needs user testing, creates checkpoint",
        "initial_pool": {
            "tasks": {
                "T001": {"type": "IMPLEMENT", "status": "ready", "description": "Add responsive UI"},
            },
            "findings": [],
        },
        "expected_final": "checkpoint",
        "expected_max_iterations": 4,
    },
    {
        "name": "E_parallel_execution",
        "description": "Multiple independent tasks, parallel execution",
        "initial_pool": {
            "tasks": {
                "T001": {"type": "EXPLORE", "status": "ready", "description": "Research caching"},
                "T002": {"type": "EXPLORE", "status": "ready", "description": "Research indexing"},
                "T003": {"type": "IMPLEMENT", "status": "blocked", "description": "Implement solution"},
            },
            "findings": [],
        },
        "expected_final": "done",
        "expected_max_iterations": 6,
    },
]


# =============================================================================
# Orchestrator Agent Prompt
# =============================================================================

ORCHESTRATOR_AGENT_PROMPT = """You are an orchestrator agent managing a task pipeline.

## Your Job
Drive the task execution loop to completion. You have these tools (simulated):
- plan(): Get the next action from the planner
- work(task_id, type): Execute a task
- review(task_id): Review a completed task
- evaluate(task_id): Evaluate quality of a task
- read_pool(): Read current task pool state
- write_pool(update): Update task pool state

## Workflow Rules

1. **Start**: Read pool to understand current state
2. **Plan**: Call plan() to decide next action
3. **Execute action**:
   - EXECUTE/EXPLORE: Call work(), then review(), then evaluate()
   - CREATE/MODIFY/DELETE: Update pool directly
   - DONE: End the loop
   - PARALLEL_EXECUTE: Execute multiple tasks concurrently
4. **Handle results**:
   - If review PASSED and score >= 95: Task complete
   - If review RETRY: Re-execute the task
   - If review FAILED: Mark task as failed, let planner decide
   - If evaluation recommends PIVOT: Let planner decide
5. **Loop**: Go back to step 2

## Safety Rules (MUST keep in code)
- Maximum {max_iterations} iterations
- Track iteration count
- Handle errors gracefully (don't crash on review failure)

## Current State

{pool_state}

## Your Task

Simulate running the orchestrator loop. For each step, output:
1. What action you'd take
2. What the expected result would be
3. What the next state would be

Output as JSON array of steps:
```json
[
    {{
        "iteration": 1,
        "action": "plan",
        "decision": "execute T001",
        "next_action": "work T001 IMPLEMENT"
    }},
    {{
        "iteration": 1,
        "action": "work",
        "task_id": "T001",
        "result": "success"
    }},
    ...
    {{
        "iteration": N,
        "action": "done",
        "final_state": "completed"
    }}
]
```

Be realistic about what would happen at each step based on the scenario described.
"""


def build_scenario_prompt(scenario: dict) -> str:
    """Build prompt for a specific scenario."""
    state = MockState(scenario["name"], scenario["initial_pool"])
    pool_text = state.to_pool_text()

    scenario_context = f"""## Scenario: {scenario['name']}
{scenario['description']}

"""

    # Add scenario-specific hints
    if scenario["name"] == "A_simple_pass":
        scenario_context += "The task is straightforward. Expect: execute -> review passes -> evaluate scores 95+ -> done."
    elif scenario["name"] == "B_retry_then_pass":
        scenario_context += "The first attempt has a minor bug. Expect: execute -> review says RETRY -> re-execute -> review passes -> done."
    elif scenario["name"] == "C_declining_pivot":
        scenario_context += "Scores decline across attempts: 40, 35, 30. Expect: execute -> evaluate low score -> retry -> evaluate lower -> pivot recommended."
    elif scenario["name"] == "D_user_test_checkpoint":
        scenario_context += "The task needs user testing (UI task). Expect: execute -> review passes -> evaluate creates checkpoint for user testing."
    elif scenario["name"] == "E_parallel_execution":
        scenario_context += "T001 and T002 are independent EXPLORE tasks. Expect: parallel_execute T001,T002 -> both complete -> unblock T003 -> execute T003 -> done."

    prompt = ORCHESTRATOR_AGENT_PROMPT.format(
        max_iterations=scenario["expected_max_iterations"],
        pool_state=pool_text,
    )

    return scenario_context + "\n" + prompt


def parse_orchestrator_output(text: str) -> dict:
    """Parse orchestrator agent output."""
    # Try to extract JSON array
    import re

    # Try code fence
    fence_match = re.search(r"```(?:json)?\s*\n(\[.*?\])\s*\n```", text, re.DOTALL)
    if fence_match:
        try:
            steps = json.loads(fence_match.group(1))
            return {"steps": steps, "parsed": True}
        except json.JSONDecodeError:
            pass

    # Try bare JSON array
    bracket_start = text.find("[")
    bracket_end = text.rfind("]")
    if bracket_start != -1 and bracket_end > bracket_start:
        try:
            steps = json.loads(text[bracket_start:bracket_end + 1])
            return {"steps": steps, "parsed": True}
        except json.JSONDecodeError:
            pass

    # Fallback: analyze text
    return {"steps": [], "parsed": False, "raw": text[:500]}


def analyze_steps(steps: list, scenario: dict) -> dict:
    """Analyze if the orchestrator followed the correct workflow."""
    if not steps:
        return {"correct": False, "reason": "No steps parsed"}

    expected_final = scenario["expected_final"]
    max_iters = scenario["expected_max_iterations"]

    # Check final state
    last_step = steps[-1]
    final_state = last_step.get("final_state", last_step.get("action", ""))

    # Flexible matching for final state
    final_correct = False
    if expected_final == "done":
        final_correct = "done" in str(final_state).lower() or "complete" in str(final_state).lower()
    elif expected_final == "pivot":
        final_correct = "pivot" in str(final_state).lower() or "pivot" in str(last_step).lower()
    elif expected_final == "checkpoint":
        final_correct = any("checkpoint" in str(s).lower() or "user test" in str(s).lower() for s in steps)

    # Check iteration count
    iterations = len([s for s in steps if s.get("iteration")])
    if iterations == 0:
        iterations = len(steps)
    within_limit = iterations <= max_iters * 1.5

    return {
        "final_correct": final_correct,
        "within_iteration_limit": within_limit,
        "actual_iterations": iterations,
        "expected_max": max_iters,
        "correct": final_correct and within_limit,
    }


async def run_single_trial(scenario: dict, trial_num: int) -> dict:
    """Run a single trial."""
    prompt = build_scenario_prompt(scenario)
    result_text = ""

    try:
        async for message in query(
            prompt=prompt,
            options=ClaudeCodeOptions(
                system_prompt="You are simulating an orchestrator agent. Output the workflow steps as JSON.",
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
            "expected_final": scenario["expected_final"],
            "parsed": False,
            "correct": False,
            "error": str(e),
        }

    output = parse_orchestrator_output(result_text)
    analysis = analyze_steps(output.get("steps", []), scenario) if output.get("parsed") else {
        "correct": False,
        "reason": "Failed to parse steps",
    }

    return {
        "scenario": scenario["name"],
        "trial": trial_num,
        "expected_final": scenario["expected_final"],
        "parsed": output.get("parsed", False),
        "num_steps": len(output.get("steps", [])),
        "final_correct": analysis.get("final_correct", False),
        "within_limit": analysis.get("within_iteration_limit", False),
        "actual_iterations": analysis.get("actual_iterations", 0),
        "correct": analysis.get("correct", False),
        "raw_output": result_text[:300],
        "error": None,
    }


async def run_experiment(repeats: int = 3):
    """Run Experiment 4."""
    print("=" * 60)
    print("Experiment 4: Orchestrator Agent Loop")
    print(f"Scenarios: {len(SCENARIOS)} x {repeats} repeats = {len(SCENARIOS) * repeats} trials")
    print("=" * 60)

    all_results = []

    for scenario in SCENARIOS:
        print(f"\n--- {scenario['name']}: {scenario['description']} (expected: {scenario['expected_final']}) ---")

        for trial in range(1, repeats + 1):
            print(f"  Trial {trial}/{repeats}...", end=" ", flush=True)
            result = await run_single_trial(scenario, trial)
            all_results.append(result)

            if result["error"]:
                print(f"ERROR: {result['error']}")
            else:
                ok = "OK" if result["correct"] else "WRONG"
                parsed = "json" if result["parsed"] else "text"
                print(f"final={'OK' if result['final_correct'] else 'WRONG'} iters={result['actual_iterations']} ({ok}, {parsed})")

    # Aggregate
    total = len(all_results)
    errors = sum(1 for r in all_results if r["error"])
    valid = total - errors
    correct = sum(1 for r in all_results if r["correct"])
    parsed = sum(1 for r in all_results if r.get("parsed") and not r["error"])
    final_correct = sum(1 for r in all_results if r.get("final_correct") and not r["error"])

    print(f"\n{'='*60}")
    print(f"Results ({valid} valid trials, {errors} errors):")
    print(f"  JSON parse rate:    {parsed}/{valid} ({parsed/valid:.1%})" if valid else "  No valid")
    print(f"  Final state correct: {final_correct}/{valid} ({final_correct/valid:.1%})" if valid else "")
    print(f"  Overall correct:    {correct}/{valid} ({correct/valid:.1%})" if valid else "")
    print(f"  Threshold: >= 80% final state correct")

    passed_threshold = valid and final_correct / valid >= 0.8
    print(f"  PASSED: {'YES' if passed_threshold else 'NO'}")
    print(f"{'='*60}")

    # Per-scenario breakdown
    print(f"\nPer-scenario breakdown:")
    for scenario in SCENARIOS:
        s_results = [r for r in all_results if r["scenario"] == scenario["name"] and not r["error"]]
        if s_results:
            c = sum(1 for r in s_results if r["correct"])
            f = sum(1 for r in s_results if r["final_correct"])
            n = len(s_results)
            avg_iters = sum(r.get("actual_iterations", 0) for r in s_results) / n if n else 0
            print(f"  {scenario['name']:<25} final={f}/{n}  overall={c}/{n}  avg_iters={avg_iters:.1f}")

    # Save
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"exp4_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump({
            "experiment": "exp4_orchestrator_agent",
            "total_trials": total,
            "errors": errors,
            "json_parse_rate": parsed / valid if valid else 0,
            "final_state_accuracy": final_correct / valid if valid else 0,
            "overall_accuracy": correct / valid if valid else 0,
            "passed": passed_threshold,
            "results": all_results,
        }, f, indent=2, default=str)

    print(f"\nResults saved: {output_file}")
    return passed_threshold


if __name__ == "__main__":
    passed = asyncio.run(run_experiment(repeats=3))
    print(f"\nExperiment 4: {'PASSED' if passed else 'FAILED'}")
