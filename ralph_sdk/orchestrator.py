"""Main orchestration module."""

import asyncio
import os

from rich.console import Console
from rich.panel import Panel

from ralph_sdk.clarifier import clarify_requirements, quick_clarify
from ralph_sdk.executor import execute_story
from ralph_sdk.models import ExecutionContext, PRD
from ralph_sdk.prd_generator import generate_prd, load_prd, save_prd
from ralph_sdk.progress import init_progress_file, ProgressTracker

console = Console()


def _show_startup_panel(title: str, **kwargs: str) -> None:
    """Display a startup panel with the given title and key-value pairs."""
    lines = ["[bold]Ralph SDK[/bold]", ""]
    lines.extend(f"{k}: {v}" for k, v in kwargs.items())
    console.print(Panel("\n".join(lines), title=title, border_style="blue"))


async def run_ralph(
    initial_prompt: str,
    cwd: str = ".",
    max_iterations: int = 10,
    skip_clarify: bool = False,
    prd_file: str = "prd.json",
    progress_file: str = "progress.txt",
) -> bool:
    """
    Run the complete Ralph workflow.

    Args:
        initial_prompt: User's feature request
        cwd: Working directory
        max_iterations: Maximum number of story executions
        skip_clarify: Skip the clarification phase
        prd_file: Path to save/load PRD
        progress_file: Path to progress log

    Returns:
        True if all stories completed successfully
    """
    cwd = os.path.abspath(cwd)
    prd_path = os.path.join(cwd, prd_file)
    progress_path = os.path.join(cwd, progress_file)

    _show_startup_panel(
        "Starting",
        **{"Working directory": cwd, "Max iterations": str(max_iterations)},
    )

    # Phase 1: Clarify requirements
    console.print("\n[bold cyan]Phase 1: Requirements Clarification[/bold cyan]\n")

    if skip_clarify:
        requirements = await quick_clarify(initial_prompt)
    else:
        requirements = await clarify_requirements(initial_prompt, cwd=cwd)

    # Phase 2: Generate PRD
    console.print("\n[bold cyan]Phase 2: PRD Generation[/bold cyan]\n")

    prd = await generate_prd(requirements, cwd=cwd)
    save_prd(prd, prd_path)

    # Phase 3: Execute stories
    console.print("\n[bold cyan]Phase 3: Story Execution[/bold cyan]\n")

    success = await execute_prd(
        prd=prd,
        cwd=cwd,
        max_iterations=max_iterations,
        prd_path=prd_path,
        progress_path=progress_path,
    )

    return success


async def execute_prd(
    prd: PRD,
    cwd: str,
    max_iterations: int = 10,
    prd_path: str = "prd.json",
    progress_path: str = "progress.txt",
) -> bool:
    """
    Execute stories from an existing PRD.

    Args:
        prd: PRD object with user stories
        cwd: Working directory
        max_iterations: Maximum iterations
        prd_path: Path to update PRD
        progress_path: Path to progress log

    Returns:
        True if all stories completed
    """
    # Initialize progress tracking
    init_progress_file(progress_path)
    progress = ProgressTracker(progress_path)

    # Create execution context
    context = ExecutionContext(
        cwd=cwd,
        branch_name=prd.branch_name,
        progress_file=progress_path,
        prd_file=prd_path,
    )

    # Setup git branch
    await setup_branch(context)

    for iteration in range(1, max_iterations + 1):
        story = prd.get_next_story()
        if story is None:
            console.print("\n[bold green]All stories complete![/bold green]")
            return True

        console.print(f"\n[bold]=== Iteration {iteration}/{max_iterations} ===[/bold]\n")
        console.print(f"Progress: {prd.progress_summary()}")

        result = await execute_story(
            story=story,
            context=context,
            progress_context=progress.get_context(),
        )

        # Update progress tracking
        progress.append_log(story.id, result)
        progress.consolidate_patterns(result.learnings)
        progress.save()

        if result.success:
            prd.mark_complete(story.id, notes="\n".join(result.learnings))
            save_prd(prd, prd_path)
        else:
            console.print(f"\n[yellow]Story {story.id} failed. Continuing...[/yellow]")

        await asyncio.sleep(1)

    # Check final status
    if prd.is_complete():
        console.print("\n[bold green]All stories complete![/bold green]")
        return True

    console.print(
        f"\n[yellow]Reached max iterations ({max_iterations}). "
        f"{prd.progress_summary()}[/yellow]"
    )
    return False


async def setup_branch(context: ExecutionContext) -> None:
    """Setup git branch for the feature."""
    from claude_code_sdk import query, ClaudeCodeOptions

    branch_prompt = f"""Check if we're on the correct git branch: {context.branch_name}

If not:
1. Check if the branch exists
2. If it exists, check it out
3. If it doesn't exist, create it from main/master

Output the current branch name when done.
"""

    console.print(f"[dim]Setting up branch: {context.branch_name}[/dim]")

    async for message in query(
        prompt=branch_prompt,
        options=ClaudeCodeOptions(
            allowed_tools=["Bash"],
            permission_mode="acceptEdits",
            max_turns=5,
            cwd=context.cwd,
        ),
    ):
        pass  # Just let it run


async def run_from_prd_file(
    prd_path: str,
    max_iterations: int = 10,
    progress_file: str = "progress.txt",
) -> bool:
    """
    Run Ralph from an existing PRD file.

    Args:
        prd_path: Path to prd.json
        max_iterations: Maximum iterations
        progress_file: Path to progress log

    Returns:
        True if all stories completed
    """
    prd_path = os.path.abspath(prd_path)
    cwd = os.path.dirname(prd_path)
    progress_path = os.path.join(cwd, progress_file)

    _show_startup_panel(
        "Resuming",
        PRD=prd_path,
        **{"Max iterations": str(max_iterations)},
    )

    prd = load_prd(prd_path)

    console.print(f"\nLoaded PRD: {prd.project}")
    console.print(f"Progress: {prd.progress_summary()}\n")

    return await execute_prd(
        prd=prd,
        cwd=cwd,
        max_iterations=max_iterations,
        prd_path=prd_path,
        progress_path=progress_path,
    )
