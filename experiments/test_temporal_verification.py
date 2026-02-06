"""Test temporal verification mechanism in Ralph SDK."""
import asyncio
import shutil
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from ralph_sdk.orchestrator import run
from rich.console import Console

console = Console()

# Task that requires temporal verification
# - Uses external library (transformers) that changes frequently
# - Asks for "current best practice"
# - Involves version-sensitive API

TASK_PROMPT = """
帮我写一个 Python 脚本，使用 Hugging Face transformers 库加载一个预训练的文本生成模型，
并生成一段文本。

要求：
1. 使用目前推荐的方式加载模型（不要用废弃的 API）
2. 选择一个适合文本生成的小型模型（方便测试）
3. 代码要能直接运行

请确保使用的是最新的 API 用法，不要用过时的方法。
"""


async def main():
    test_dir = Path(f"/tmp/ralph_temporal_{datetime.now().strftime('%H%M%S')}")
    test_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold cyan]═══ Temporal Verification Test ═══[/bold cyan]")
    console.print(f"[dim]Directory: {test_dir}[/dim]")
    console.print(f"\n[yellow]Task: Write transformers code using current best practices[/yellow]\n")

    start = datetime.now()

    try:
        success = await run(
            goal=TASK_PROMPT,
            cwd=str(test_dir),
            max_iterations=8,
            skip_clarify=True,  # Skip interactive questions for automated test
            verbose=True,
        )

        duration = (datetime.now() - start).total_seconds()

        console.print(f"\n[bold]═══ Results ═══[/bold]")
        console.print(f"Duration: {duration:.0f}s")
        console.print(f"Success: {success}")

        # Check for temporal verification markers
        console.print(f"\n[bold]Temporal Verification Checks:[/bold]")

        pool_path = test_dir / ".ralph" / "pool.md"
        goal_path = test_dir / ".ralph" / "goal.md"

        checks = {
            "Goal has Temporal Topics section": False,
            "Pool has Verified Information": False,
            "WebSearch was used": False,
            "Verification annotations present": False,
        }

        if goal_path.exists():
            goal_content = goal_path.read_text()
            checks["Goal has Temporal Topics section"] = "Temporal Topics" in goal_content or "时效性" in goal_content

        if pool_path.exists():
            pool_content = pool_path.read_text()
            checks["Pool has Verified Information"] = "[Verified" in pool_content
            checks["Verification annotations present"] = "已搜索验证" in pool_content or "[Verified" in pool_content

        # Check task files for WebSearch evidence
        tasks_dir = test_dir / ".ralph" / "tasks"
        if tasks_dir.exists():
            for task_file in tasks_dir.glob("*.md"):
                content = task_file.read_text()
                if "WebSearch" in content or "搜索" in content or "verified" in content.lower():
                    checks["WebSearch was used"] = True
                    break

        for name, passed in checks.items():
            status = "[green]✓[/green]" if passed else "[red]✗[/red]"
            console.print(f"  {status} {name}")

        score = sum(checks.values())
        console.print(f"\n[bold]Score: {score}/{len(checks)}[/bold]")

        # Show generated code if exists
        console.print(f"\n[bold]Generated Files:[/bold]")
        for py_file in test_dir.glob("*.py"):
            console.print(f"  - {py_file.name}")
            content = py_file.read_text()
            # Check for modern API patterns
            if "pipeline" in content or "AutoModel" in content:
                console.print(f"    [green]Uses modern API pattern[/green]")
            if "from_pretrained" in content:
                console.print(f"    [dim]Has from_pretrained call[/dim]")

        console.print(f"\n[dim]Test dir: {test_dir}[/dim]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
