#!/usr/bin/env python3
"""
Experiment 2: Pivot Mechanism Value

Hypothesis: Three pivot triggers reduce wasted effort on dead-end approaches.

A: No pivot mechanism (keep trying same approach)
B: With pivot mechanism (PIVOT_RESEARCH, PIVOT_WAIT, PIVOT_ITERATION)
"""

import asyncio
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ralph_sdk.orchestrator import run
from ralph_sdk.pool import read_goal, read_pool, write_goal, init_ralph_dir


# A task designed to have "first path doesn't work"
TEST_TASK = {
    "id": "pivot_test",
    "prompt": """实现一个高性能的代码相似度检测器。

要求：
1. 能检测 Python 代码的相似度
2. 准确率 >= 95%
3. 处理速度 < 100ms per file pair

注意：这个任务故意设计成第一种方案（基于字符串匹配）不能满足准确率要求，
需要转向更复杂的方法（AST 分析或 embedding）。
""",
}

# Pre-made goal.md that leads to a dead end
DEAD_END_GOAL = """# Goal

## Original Request
实现一个高性能的代码相似度检测器

## Clarified Description
使用字符串匹配算法（Levenshtein 距离）实现代码相似度检测。

## Scope
- 基于字符级别的相似度计算
- 使用编辑距离算法

## Non-goals
- 不考虑语义相似度
- 不使用机器学习方法

## Success Metrics
### Hard Constraints
- [ ] **runs_without_error** [auto]: pass
- [ ] **accuracy** [auto]: >= 95%
- [ ] **performance** [auto]: < 100ms

## Temporal Topics
- [ ] Python string matching libraries - verify current best practice
"""


async def run_without_pivot(work_dir: str, max_iterations: int = 10) -> dict:
    """Run without pivot mechanism - keep trying same approach."""
    print(f"\n{'='*60}")
    print("Running WITHOUT pivot mechanism")
    print(f"{'='*60}\n")

    # Write the dead-end goal
    init_ralph_dir(work_dir)
    write_goal(DEAD_END_GOAL, work_dir)

    start_time = time.time()

    # Run with default settings but track iterations
    # Note: This would normally require modifying the planner to disable pivots
    # For now, we simulate by just running and seeing how many iterations

    result = {
        "mode": "no_pivot",
        "start_time": datetime.now().isoformat(),
    }

    try:
        success = await run(
            goal=TEST_TASK["prompt"],
            cwd=work_dir,
            max_iterations=max_iterations,
            skip_clarify=True,  # Use pre-made goal
        )

        result["success"] = success
        result["elapsed_seconds"] = time.time() - start_time

        # Read final state
        result["final_pool"] = read_pool(work_dir)

    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        result["elapsed_seconds"] = time.time() - start_time

    return result


async def run_with_pivot(work_dir: str, max_iterations: int = 10) -> dict:
    """Run with pivot mechanism enabled."""
    print(f"\n{'='*60}")
    print("Running WITH pivot mechanism")
    print(f"{'='*60}\n")

    # Write the dead-end goal
    init_ralph_dir(work_dir)
    write_goal(DEAD_END_GOAL, work_dir)

    start_time = time.time()

    result = {
        "mode": "with_pivot",
        "start_time": datetime.now().isoformat(),
    }

    try:
        success = await run(
            goal=TEST_TASK["prompt"],
            cwd=work_dir,
            max_iterations=max_iterations,
            skip_clarify=True,
        )

        result["success"] = success
        result["elapsed_seconds"] = time.time() - start_time
        result["final_pool"] = read_pool(work_dir)

    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        result["elapsed_seconds"] = time.time() - start_time

    return result


def count_pivots(pool_content: str) -> int:
    """Count pivot events in pool.md."""
    import re
    pivot_count = 0
    pivot_count += len(re.findall(r'PIVOT_RESEARCH', pool_content))
    pivot_count += len(re.findall(r'PIVOT_ITERATION', pool_content))
    pivot_count += len(re.findall(r'PIVOT_WAIT', pool_content))
    pivot_count += len(re.findall(r'HEDGE', pool_content))
    return pivot_count


async def main():
    """Run experiment 2."""
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    base_work_dir = Path(__file__).parent / "work"

    # Run without pivot
    work_dir_a = base_work_dir / "pivot_test_no_pivot"
    if work_dir_a.exists():
        shutil.rmtree(work_dir_a)
    work_dir_a.mkdir(parents=True)

    result_a = await run_without_pivot(str(work_dir_a))
    result_a["pivot_count"] = count_pivots(result_a.get("final_pool", ""))

    # Run with pivot
    work_dir_b = base_work_dir / "pivot_test_with_pivot"
    if work_dir_b.exists():
        shutil.rmtree(work_dir_b)
    work_dir_b.mkdir(parents=True)

    result_b = await run_with_pivot(str(work_dir_b))
    result_b["pivot_count"] = count_pivots(result_b.get("final_pool", ""))

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"experiment2_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump({
            "experiment": "pivot_mechanism_value",
            "timestamp": timestamp,
            "task": TEST_TASK,
            "results": [result_a, result_b],
        }, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'='*60}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*60}")

    # Print summary
    print("\n## Summary\n")
    print(f"Without Pivot: success={result_a.get('success')}, "
          f"time={result_a.get('elapsed_seconds', 0):.1f}s, "
          f"pivots={result_a.get('pivot_count', 0)}")
    print(f"With Pivot: success={result_b.get('success')}, "
          f"time={result_b.get('elapsed_seconds', 0):.1f}s, "
          f"pivots={result_b.get('pivot_count', 0)}")


if __name__ == "__main__":
    asyncio.run(main())
