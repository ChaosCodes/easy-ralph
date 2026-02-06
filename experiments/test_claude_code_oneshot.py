"""Quick one-shot test with Claude Code."""
import asyncio
import shutil
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from claude_code_sdk import AssistantMessage, ClaudeCodeOptions, query
from rich.console import Console

console = Console()

TASK_PROMPT = """
给 ralph-sdk 添加并行任务执行支持。

需求:
1. Planner 可以一次决定多个独立任务同时执行
2. 添加新的 Action: PARALLEL_EXECUTE
3. 修改 orchestrator 支持并行执行
4. 确保 pool.md 状态更新是原子性的（避免并发写入冲突）
5. 支持配置最大并行度

代码在 ralph_sdk/ 目录下。请直接修改代码完成任务。
"""


async def main():
    # Setup
    test_dir = Path(f"/tmp/claude_oneshot_{datetime.now().strftime('%H%M%S')}")
    test_dir.mkdir(parents=True, exist_ok=True)

    src_dir = Path(__file__).parent.parent / "ralph_sdk"
    dst_dir = test_dir / "ralph_sdk"
    shutil.copytree(src_dir, dst_dir)

    console.print(f"\n[bold cyan]Claude Code One-Shot Test[/bold cyan]")
    console.print(f"[dim]Directory: {test_dir}[/dim]\n")

    start = datetime.now()
    tool_count = 0

    async for message in query(
        prompt=TASK_PROMPT,
        options=ClaudeCodeOptions(
            system_prompt="You are an expert developer. Be thorough.",
            allowed_tools=["Read", "Write", "Edit", "Glob", "Grep", "Bash"],
            permission_mode="acceptEdits",
            max_turns=50,
            cwd=str(test_dir),
        ),
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "name"):
                    tool_count += 1
                    if tool_count % 10 == 0:
                        console.print(f"[dim]... {tool_count} tool calls[/dim]")

    duration = (datetime.now() - start).total_seconds()
    console.print(f"\n[green]Done! {tool_count} tool calls in {duration:.0f}s[/green]")

    # Check results
    console.print("\n[bold]Results:[/bold]")

    planner = (test_dir / "ralph_sdk" / "planner.py").read_text()
    orch = (test_dir / "ralph_sdk" / "orchestrator.py").read_text()
    pool = (test_dir / "ralph_sdk" / "pool.py").read_text()

    checks = {
        "PARALLEL_EXECUTE in planner": "PARALLEL" in planner.upper(),
        "task_ids field": "task_ids" in planner,
        "asyncio.gather": "gather" in orch,
        "File locking": "lock" in pool.lower() or "Lock" in pool,
        "max_parallel config": "max_parallel" in orch.lower() or "parallelism" in orch.lower(),
    }

    for name, passed in checks.items():
        status = "[green]✓[/green]" if passed else "[red]✗[/red]"
        console.print(f"  {status} {name}")

    score = sum(checks.values())
    console.print(f"\n[bold]Score: {score}/{len(checks)}[/bold]")
    console.print(f"[dim]Test dir: {test_dir}[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
