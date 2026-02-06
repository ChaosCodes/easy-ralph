#!/usr/bin/env python3
"""
Experiment 1: Clarifier Mode Comparison

Hypothesis: Explore+propose mode helps users clarify vague requirements better than ask mode.

A: Clarifier v1 (ask mode)
B: Clarifier v2 (explore+propose mode)
"""

import asyncio
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ralph_sdk.clarifier import clarify_requirements, explore_and_propose
from ralph_sdk.pool import init_ralph_dir, read_goal


TASKS = [
    {
        "id": "task3_research",
        "prompt": "老师说：探索如何用 LLM 自动评估代码质量，不依赖测试",
    },
    {
        "id": "task4_idea",
        "prompt": "导师说：研究一下多 Agent 协作的任务分配策略",
    },
]


async def run_clarifier_v1(task: dict, work_dir: str) -> dict:
    """Run Clarifier v1 (ask mode)."""
    print(f"\n{'='*60}")
    print(f"Running Clarifier v1 (ask mode) for: {task['id']}")
    print(f"{'='*60}\n")

    start_time = time.time()

    try:
        goal_content = await clarify_requirements(task["prompt"], work_dir)
        elapsed = time.time() - start_time

        return {
            "mode": "ask",
            "task_id": task["id"],
            "success": True,
            "elapsed_seconds": elapsed,
            "goal_length": len(goal_content),
            "goal_content": goal_content,
        }
    except Exception as e:
        return {
            "mode": "ask",
            "task_id": task["id"],
            "success": False,
            "error": str(e),
            "elapsed_seconds": time.time() - start_time,
        }


async def run_clarifier_v2(task: dict, work_dir: str) -> dict:
    """Run Clarifier v2 (explore+propose mode)."""
    print(f"\n{'='*60}")
    print(f"Running Clarifier v2 (explore+propose mode) for: {task['id']}")
    print(f"{'='*60}\n")

    start_time = time.time()

    try:
        goal_content = await explore_and_propose(task["prompt"], work_dir)
        elapsed = time.time() - start_time

        return {
            "mode": "explore",
            "task_id": task["id"],
            "success": True,
            "elapsed_seconds": elapsed,
            "goal_length": len(goal_content),
            "goal_content": goal_content,
        }
    except Exception as e:
        return {
            "mode": "explore",
            "task_id": task["id"],
            "success": False,
            "error": str(e),
            "elapsed_seconds": time.time() - start_time,
        }


async def evaluate_goal_clarity(goal_content: str) -> dict:
    """Use AI to evaluate goal clarity (0-100)."""
    from claude_code_sdk import AssistantMessage, ClaudeCodeOptions, query

    prompt = f"""评估以下 goal.md 的清晰度和可执行性。

Goal 内容：
---
{goal_content[:3000]}  # Truncate if too long
---

评估维度：
1. 需求清晰度 (0-100): 是否清楚要做什么
2. 技术方案清晰度 (0-100): 是否清楚怎么做
3. 范围边界清晰度 (0-100): 是否清楚不做什么
4. 可执行性 (0-100): 是否能直接开始实现

输出格式：
```json
{{
  "requirement_clarity": <score>,
  "technical_clarity": <score>,
  "scope_clarity": <score>,
  "actionability": <score>,
  "overall": <average>,
  "comments": "<简短评价>"
}}
```

只输出 JSON。
"""

    result_text = ""
    async for message in query(
        prompt=prompt,
        options=ClaudeCodeOptions(
            system_prompt="你是一个评估文档质量的助手。请严格按照要求输出 JSON。",
            allowed_tools=[],
            max_turns=1,
        ),
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    result_text += block.text

    # Parse JSON from response
    import re
    json_match = re.search(r'\{[^}]+\}', result_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return {"overall": 50, "comments": "Failed to parse evaluation"}


async def main():
    """Run experiment 1."""
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    all_results = []

    for task in TASKS:
        print(f"\n{'#'*60}")
        print(f"# Task: {task['id']}")
        print(f"{'#'*60}")

        # Create temporary work directories
        base_work_dir = Path(__file__).parent / "work"

        # Run V1 (ask mode)
        work_dir_v1 = base_work_dir / f"{task['id']}_v1"
        if work_dir_v1.exists():
            shutil.rmtree(work_dir_v1)
        work_dir_v1.mkdir(parents=True)

        result_v1 = await run_clarifier_v1(task, str(work_dir_v1))

        if result_v1["success"]:
            result_v1["clarity_eval"] = await evaluate_goal_clarity(result_v1["goal_content"])

        all_results.append(result_v1)

        # Run V2 (explore+propose mode)
        work_dir_v2 = base_work_dir / f"{task['id']}_v2"
        if work_dir_v2.exists():
            shutil.rmtree(work_dir_v2)
        work_dir_v2.mkdir(parents=True)

        result_v2 = await run_clarifier_v2(task, str(work_dir_v2))

        if result_v2["success"]:
            result_v2["clarity_eval"] = await evaluate_goal_clarity(result_v2["goal_content"])

        all_results.append(result_v2)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"experiment1_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump({
            "experiment": "clarifier_mode_comparison",
            "timestamp": timestamp,
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*60}")

    # Print summary
    print("\n## Summary\n")
    for result in all_results:
        mode_name = "Ask" if result["mode"] == "ask" else "Explore"
        print(f"- {result['task_id']} ({mode_name}): ", end="")
        if result["success"]:
            clarity = result.get("clarity_eval", {}).get("overall", "N/A")
            print(f"Clarity: {clarity}, Time: {result['elapsed_seconds']:.1f}s")
        else:
            print(f"Failed: {result.get('error', 'Unknown')}")


if __name__ == "__main__":
    asyncio.run(main())
