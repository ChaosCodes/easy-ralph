"""
Test: Real Complex Task - Add Parallel Execution to Ralph SDK

This task is designed to be hard for single-shot Claude Code because:
1. Requires understanding the full architecture first
2. Needs multi-file changes with consistency
3. Involves concurrency which needs careful design
4. Needs iterative development and testing

We'll test this with:
- Ralph SDK pipeline (multiple iterations allowed)
- Single Claude Code session (one-shot attempt)
"""

import asyncio
import shutil
import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel

console = Console()

# The real task prompt
TASK_PROMPT = """
给 ralph-sdk 添加并行任务执行支持。

## 背景
当前 ralph-sdk 的 orchestrator 是串行执行任务的（一次执行一个 EXECUTE/EXPLORE）。
但很多任务之间是独立的，可以并行执行提高效率。

## 需求
1. Planner 可以一次决定多个独立任务同时执行
2. 添加新的 Action: PARALLEL_EXECUTE
3. 修改 orchestrator 支持并行执行
4. 确保 pool.md 状态更新是原子性的（避免并发写入冲突）
5. 支持配置最大并行度

## 约束
- 保持向后兼容（现有单任务执行仍然工作）
- 只有互不依赖的任务才能并行
- 如果一个并行任务失败，其他任务应该继续

## 验收标准
- [ ] 新增 PARALLEL_EXECUTE action
- [ ] Planner prompt 更新，说明何时使用并行
- [ ] orchestrator 能同时运行多个 Worker
- [ ] pool.md 写入有锁保护
- [ ] 有配置项控制最大并行度
"""

# Simpler task to compare
SIMPLE_TASK_PROMPT = """
给 ralph-sdk 添加一个简单功能：在 pool.md 的 Progress Log 中记录每次操作的耗时。

需求：
1. 修改 append_to_progress_log 函数，添加耗时参数
2. 在 orchestrator 中计算每个操作的耗时并传入
3. 格式：`### 2024-01-01 12:00 (耗时: 5.2s)`
"""


async def test_with_ralph_sdk(task_prompt: str, task_name: str, max_iterations: int = 20) -> dict:
    """Test task using Ralph SDK pipeline."""
    from ralph_sdk.orchestrator import run
    from ralph_sdk.pool import read_pool, read_goal

    # Create isolated test directory
    test_dir = Path(f"/tmp/ralph_test_{task_name}_{datetime.now().strftime('%H%M%S')}")
    test_dir.mkdir(parents=True, exist_ok=True)

    # Copy ralph-sdk source to test directory (so it can modify itself)
    src_dir = Path(__file__).parent.parent / "ralph_sdk"
    dst_dir = test_dir / "ralph_sdk"
    shutil.copytree(src_dir, dst_dir)

    # Create a minimal pyproject.toml
    (test_dir / "pyproject.toml").write_text('''[project]
name = "ralph-sdk-test"
version = "0.1.0"
''')

    console.print(f"\n[cyan]Testing with Ralph SDK: {task_name}[/cyan]")
    console.print(f"[dim]Directory: {test_dir}[/dim]\n")

    start_time = datetime.now()

    try:
        success = await run(
            goal=task_prompt,
            cwd=str(test_dir),
            max_iterations=max_iterations,
            skip_clarify=True,
            verbose=False,
        )
        error = None
    except Exception as e:
        success = False
        error = str(e)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Collect results
    pool_content = read_pool(str(test_dir))
    iterations = pool_content.count("### 20")  # Count timestamps

    return {
        "method": "ralph_sdk",
        "task": task_name,
        "success": success,
        "duration": duration,
        "iterations": iterations,
        "error": error,
        "test_dir": str(test_dir),
    }


async def test_with_claude_code(task_prompt: str, task_name: str) -> dict:
    """Test task using single Claude Code session (simulating one-shot)."""
    from claude_code_sdk import AssistantMessage, ClaudeCodeOptions, query

    # Create isolated test directory
    test_dir = Path(f"/tmp/claude_test_{task_name}_{datetime.now().strftime('%H%M%S')}")
    test_dir.mkdir(parents=True, exist_ok=True)

    # Copy ralph-sdk source
    src_dir = Path(__file__).parent.parent / "ralph_sdk"
    dst_dir = test_dir / "ralph_sdk"
    shutil.copytree(src_dir, dst_dir)

    (test_dir / "pyproject.toml").write_text('''[project]
name = "ralph-sdk-test"
version = "0.1.0"
''')

    console.print(f"\n[cyan]Testing with Claude Code (one-shot): {task_name}[/cyan]")
    console.print(f"[dim]Directory: {test_dir}[/dim]\n")

    start_time = datetime.now()

    tool_count = 0
    try:
        async for message in query(
            prompt=f"""你是一个高级开发者，请完成以下任务：

{task_prompt}

代码在 ralph_sdk/ 目录下。请直接修改代码完成任务。
完成后，列出你修改了哪些文件，以及关键改动。
""",
            options=ClaudeCodeOptions(
                system_prompt="You are an expert developer. Complete the task thoroughly.",
                allowed_tools=["Read", "Write", "Edit", "Glob", "Grep", "Bash"],
                permission_mode="acceptEdits",
                max_turns=50,  # Give it enough turns
                cwd=str(test_dir),
            ),
        ):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if hasattr(block, "name"):
                        tool_count += 1

        success = True
        error = None
    except Exception as e:
        success = False
        error = str(e)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    return {
        "method": "claude_code",
        "task": task_name,
        "success": success,
        "duration": duration,
        "tool_calls": tool_count,
        "error": error,
        "test_dir": str(test_dir),
    }


def evaluate_parallel_execution(test_dir: str) -> dict:
    """Check if parallel execution was properly implemented."""
    test_path = Path(test_dir)
    scores = {}

    # Check 1: PARALLEL_EXECUTE in planner.py
    planner = (test_path / "ralph_sdk" / "planner.py").read_text()
    scores["parallel_action"] = "PARALLEL" in planner.upper()

    # Check 2: Planner prompt updated
    prompts = (test_path / "ralph_sdk" / "prompts.py").read_text()
    scores["prompt_updated"] = "parallel" in prompts.lower() or "PARALLEL" in prompts

    # Check 3: Orchestrator handles parallel
    orchestrator = (test_path / "ralph_sdk" / "orchestrator.py").read_text()
    scores["orchestrator_parallel"] = (
        "gather" in orchestrator or
        "create_task" in orchestrator or
        "parallel" in orchestrator.lower()
    )

    # Check 4: Lock for pool.md
    pool = (test_path / "ralph_sdk" / "pool.py").read_text()
    scores["pool_lock"] = (
        "Lock" in pool or
        "lock" in pool or
        "asyncio.Lock" in pool or
        "threading.Lock" in pool
    )

    # Check 5: Max parallelism config
    scores["max_parallel_config"] = (
        "max_parallel" in orchestrator.lower() or
        "parallelism" in orchestrator.lower() or
        "concurrent" in orchestrator.lower()
    )

    total = sum(scores.values())
    return {
        "scores": scores,
        "total": total,
        "max": len(scores),
        "percentage": total / len(scores) * 100,
    }


async def run_comparison():
    """Run comparison between Ralph SDK and Claude Code."""
    console.print(Panel(
        "[bold]Real Task Test: Parallel Execution Support[/bold]\n\n"
        "Testing if Ralph SDK pipeline can handle complex tasks\n"
        "better than single-shot Claude Code.",
        title="Experiment",
    ))

    # Test with Ralph SDK
    console.print("\n[bold yellow]Phase 1: Testing with Ralph SDK[/bold yellow]")
    ralph_result = await test_with_ralph_sdk(TASK_PROMPT, "parallel", max_iterations=15)

    # Test with Claude Code
    console.print("\n[bold yellow]Phase 2: Testing with Claude Code (one-shot)[/bold yellow]")
    claude_result = await test_with_claude_code(TASK_PROMPT, "parallel")

    # Evaluate both
    console.print("\n[bold yellow]Phase 3: Evaluation[/bold yellow]")

    ralph_eval = evaluate_parallel_execution(ralph_result["test_dir"])
    claude_eval = evaluate_parallel_execution(claude_result["test_dir"])

    # Results
    console.print("\n" + "="*60)
    console.print("[bold]RESULTS[/bold]")
    console.print("="*60)

    console.print("\n[cyan]Ralph SDK:[/cyan]")
    console.print(f"  Duration: {ralph_result['duration']:.1f}s")
    console.print(f"  Iterations: {ralph_result.get('iterations', 'N/A')}")
    console.print(f"  Score: {ralph_eval['total']}/{ralph_eval['max']} ({ralph_eval['percentage']:.0f}%)")
    for check, passed in ralph_eval["scores"].items():
        status = "[green]✓[/green]" if passed else "[red]✗[/red]"
        console.print(f"    {status} {check}")

    console.print("\n[cyan]Claude Code (one-shot):[/cyan]")
    console.print(f"  Duration: {claude_result['duration']:.1f}s")
    console.print(f"  Tool calls: {claude_result.get('tool_calls', 'N/A')}")
    console.print(f"  Score: {claude_eval['total']}/{claude_eval['max']} ({claude_eval['percentage']:.0f}%)")
    for check, passed in claude_eval["scores"].items():
        status = "[green]✓[/green]" if passed else "[red]✗[/red]"
        console.print(f"    {status} {check}")

    # Analysis
    console.print("\n[bold]Analysis:[/bold]")
    if ralph_eval["percentage"] > claude_eval["percentage"]:
        console.print("[green]Ralph SDK performed better - pipeline helps with complex tasks![/green]")
    elif ralph_eval["percentage"] < claude_eval["percentage"]:
        console.print("[yellow]Claude Code performed better - single-shot was sufficient[/yellow]")
    else:
        console.print("[blue]Similar performance[/blue]")


if __name__ == "__main__":
    asyncio.run(run_comparison())
