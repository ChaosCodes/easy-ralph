#!/usr/bin/env python3
"""
Experiment 4: Autonomous Judgment vs Wait for User

Hypothesis: Agent self-judgment + notification is more efficient than waiting for user.

A: Wait for user confirmation each iteration
B: Agent judges autonomously, notifies but continues
"""

import asyncio
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ralph_sdk.evaluator import evaluate, get_attempt_history, Metric, MetricType
from ralph_sdk.pool import init_ralph_dir, write_goal, write_task, read_task


# A task requiring multiple iterations
TEST_TASK = {
    "id": "T001",
    "prompt": "优化排序算法的性能",
    "goal": """# Goal

## Original Request
优化排序算法的性能

## Success Metrics
### Hard Constraints
- [ ] **runs_without_error** [auto]: pass

### Performance Targets
| Metric | Target | Automation |
|--------|--------|------------|
| performance | >= 90 | auto |
| code_quality | >= 70 | auto |
""",
}

# Simulated task content with iteration history
TASK_TEMPLATE = """# T001: Optimize Sorting Performance

## Type
IMPLEMENT

## Status
in_progress

## Description
Optimize the sorting algorithm for better performance.

## Acceptance Criteria
- [ ] Sorting is correct
- [ ] Performance score >= 90
- [ ] Code is readable

## Execution Log

{execution_log}

## Quality Evaluation History

{eval_history}
"""

# Simulated scores across iterations (designed to need pivot)
SIMULATED_SCORES = [45, 52, 55, 54, 58, 60]  # Slow improvement, then plateau

TEST_METRICS = [
    Metric("runs_without_error", MetricType.HARD, "Code runs without errors"),
    Metric("performance", MetricType.SOFT, "Algorithm performance", target=">= 90"),
    Metric("code_quality", MetricType.SUBJECTIVE, "Code readability"),
]


async def simulate_iteration(
    work_dir: str,
    iteration: int,
    score: float,
    wait_for_user: bool = False,
) -> dict:
    """Simulate one iteration of optimization."""
    result = {
        "iteration": iteration,
        "score": score,
        "wait_for_user": wait_for_user,
    }

    # Build execution log
    exec_log = f"### Iteration {iteration}\n"
    exec_log += f"- Tried optimization approach {iteration}\n"
    exec_log += f"- Score achieved: {score}\n\n"

    # Build eval history
    eval_history = ""
    for i in range(iteration):
        eval_history += f"## Quality Evaluation (Iteration {i+1})\n"
        eval_history += f"**Score**: {SIMULATED_SCORES[i]}/100\n\n"

    # Write task file
    task_content = TASK_TEMPLATE.format(
        execution_log=exec_log,
        eval_history=eval_history,
    )
    write_task("T001", task_content, work_dir)

    # Run evaluation with attempt tracking
    previous_scores = SIMULATED_SCORES[:iteration]

    eval_result = await evaluate(
        "T001",
        work_dir,
        metrics=TEST_METRICS,
        previous_scores=previous_scores,
        attempt_number=iteration + 1,
    )

    result["eval_score"] = eval_result.overall_score
    result["should_pivot"] = eval_result.should_pivot
    result["pivot_reason"] = eval_result.pivot_reason

    if wait_for_user:
        # Simulate waiting for user
        result["user_wait_time"] = 5.0  # Simulated 5 second wait
    else:
        result["user_wait_time"] = 0.0

    return result


async def run_with_user_wait(work_dir: str) -> dict:
    """Run with user confirmation required each iteration."""
    print(f"\n{'='*60}")
    print("Running WITH user wait (blocking)")
    print(f"{'='*60}\n")

    init_ralph_dir(work_dir)
    write_goal(TEST_TASK["goal"], work_dir)

    start_time = time.time()
    iterations = []
    total_user_wait = 0.0

    for i, score in enumerate(SIMULATED_SCORES):
        iter_result = await simulate_iteration(work_dir, i, score, wait_for_user=True)
        iterations.append(iter_result)
        total_user_wait += iter_result["user_wait_time"]

        print(f"Iteration {i+1}: score={score}, pivot_recommended={iter_result['should_pivot']}")

        # Simulate user deciding to continue or stop
        if iter_result["should_pivot"]:
            print("  → Agent recommends pivot, but waiting for user...")
            # In real scenario, might wait longer for user decision

        if i >= 3 and iter_result["should_pivot"]:
            # User eventually agrees to pivot after seeing plateau
            break

    return {
        "mode": "user_wait",
        "total_time": time.time() - start_time,
        "total_user_wait": total_user_wait,
        "iterations": len(iterations),
        "final_score": iterations[-1]["eval_score"] if iterations else 0,
        "pivot_at_iteration": len(iterations) if iterations[-1]["should_pivot"] else None,
        "iteration_details": iterations,
    }


async def run_autonomous(work_dir: str) -> dict:
    """Run with autonomous judgment (no blocking wait)."""
    print(f"\n{'='*60}")
    print("Running AUTONOMOUS (non-blocking)")
    print(f"{'='*60}\n")

    init_ralph_dir(work_dir)
    write_goal(TEST_TASK["goal"], work_dir)

    start_time = time.time()
    iterations = []

    for i, score in enumerate(SIMULATED_SCORES):
        iter_result = await simulate_iteration(work_dir, i, score, wait_for_user=False)
        iterations.append(iter_result)

        print(f"Iteration {i+1}: score={score}, pivot_recommended={iter_result['should_pivot']}")

        # Agent decides autonomously
        if iter_result["should_pivot"]:
            print(f"  → Agent autonomously pivots: {iter_result['pivot_reason']}")
            # Agent notifies user but doesn't wait
            break

    return {
        "mode": "autonomous",
        "total_time": time.time() - start_time,
        "total_user_wait": 0.0,
        "iterations": len(iterations),
        "final_score": iterations[-1]["eval_score"] if iterations else 0,
        "pivot_at_iteration": len(iterations) if iterations[-1]["should_pivot"] else None,
        "iteration_details": iterations,
    }


async def main():
    """Run experiment 4."""
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    base_work_dir = Path(__file__).parent / "work"

    # Run with user wait
    work_dir_a = base_work_dir / "autonomous_test_wait"
    if work_dir_a.exists():
        shutil.rmtree(work_dir_a)
    work_dir_a.mkdir(parents=True)

    result_a = await run_with_user_wait(str(work_dir_a))

    # Run autonomous
    work_dir_b = base_work_dir / "autonomous_test_auto"
    if work_dir_b.exists():
        shutil.rmtree(work_dir_b)
    work_dir_b.mkdir(parents=True)

    result_b = await run_autonomous(str(work_dir_b))

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"experiment4_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump({
            "experiment": "autonomous_judgment_vs_wait",
            "timestamp": timestamp,
            "simulated_scores": SIMULATED_SCORES,
            "results": [result_a, result_b],
        }, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*60}")

    # Print summary
    print("\n## Summary\n")
    print(f"With User Wait:")
    print(f"  - Total time: {result_a['total_time']:.1f}s")
    print(f"  - User wait time: {result_a['total_user_wait']:.1f}s")
    print(f"  - Iterations: {result_a['iterations']}")
    print(f"  - Pivot at: iteration {result_a['pivot_at_iteration']}")

    print(f"\nAutonomous:")
    print(f"  - Total time: {result_b['total_time']:.1f}s")
    print(f"  - User wait time: {result_b['total_user_wait']:.1f}s")
    print(f"  - Iterations: {result_b['iterations']}")
    print(f"  - Pivot at: iteration {result_b['pivot_at_iteration']}")

    time_saved = result_a['total_user_wait']
    iterations_saved = result_a['iterations'] - result_b['iterations']
    print(f"\n## Efficiency Gain")
    print(f"  - User wait time saved: {time_saved:.1f}s")
    print(f"  - Iterations saved: {iterations_saved}")


if __name__ == "__main__":
    asyncio.run(main())
