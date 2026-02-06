"""Command-line interface for Ralph SDK."""

import asyncio
from pathlib import Path

import typer
from rich.console import Console

from .logger import archive_session, get_session_summary, list_archived_sessions
from .orchestrator import resume, run
from .pool import (
    goal_exists,
    pool_exists,
    read_goal,
    read_pool,
    is_waiting_for_user,
    parse_feedback,
    read_checkpoint_manifest,
    clear_feedback,
)

app = typer.Typer(
    name="ralph-sdk",
    help="Ralph autonomous agent with dynamic task pool",
    add_completion=False,
)
console = Console()


@app.command("run")
def run_cmd(
    prompt: str = typer.Argument(..., help="Goal or feature description"),
    cwd: str = typer.Option(".", "--cwd", "-C", help="Working directory"),
    max_iterations: int = typer.Option(30, "--max", "-n", help="Maximum iterations"),
    skip_clarify: bool = typer.Option(
        False, "--quick", "-q", help="Skip clarification phase"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed tool calls and thinking"
    ),
    target_score: int = typer.Option(
        None, "--target", "-t", help="Target quality score (0-100), default 95"
    ),
    explore: bool = typer.Option(
        False, "--explore", "-e", help="Explore mode: don't DONE until user says stop"
    ),
) -> None:
    """
    Run the complete Ralph workflow: clarify → init pool → execute loop.

    Example:
        ralph-sdk run "Add user authentication with JWT"
        ralph-sdk run "Implement dark mode" --quick
        ralph-sdk run "Build API" --target 90
    """
    try:
        success = asyncio.run(
            run(
                goal=prompt,
                cwd=cwd,
                max_iterations=max_iterations,
                skip_clarify=skip_clarify,
                verbose=verbose,
                target_score=target_score,
                explore_mode=explore,
            )
        )
        raise typer.Exit(0 if success else 1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled by user[/yellow]")
        raise typer.Exit(130)


@app.command("resume")
def resume_cmd(
    cwd: str = typer.Option(".", "--cwd", "-C", help="Working directory"),
    max_iterations: int = typer.Option(30, "--max", "-n", help="Maximum iterations"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed tool calls and thinking"
    ),
    target_score: int = typer.Option(
        None, "--target", "-t", help="Target quality score (0-100), default 95"
    ),
    explore: bool = typer.Option(
        False, "--explore", "-e", help="Explore mode: don't DONE until user says stop"
    ),
) -> None:
    """
    Resume an existing session from .ralph/ directory.

    If the system is waiting for user feedback, this will process
    feedback.md and continue with the results.

    Example:
        ralph-sdk resume
        ralph-sdk resume --cwd /path/to/project
    """
    if not goal_exists(cwd) or not pool_exists(cwd):
        console.print("[red]Error: No existing session found in .ralph/[/red]")
        console.print("Run [cyan]ralph-sdk run \"your goal\"[/cyan] to start a new session")
        raise typer.Exit(1)

    # Check if waiting for user feedback
    if is_waiting_for_user(cwd):
        feedback = parse_feedback(cwd)
        manifest = read_checkpoint_manifest(cwd)

        if not feedback:
            console.print("[yellow]等待用户测试反馈[/yellow]")
            console.print("\n请编辑 [cyan].ralph/feedback.md[/cyan] 填写测试结果")
            console.print("填写完成后再次运行 [cyan]ralph-sdk resume[/cyan]")

            if manifest and manifest.get("checkpoints"):
                console.print("\n[bold]待测试的 checkpoints:[/bold]")
                for cp in manifest["checkpoints"]:
                    proxy_info = ""
                    if cp.get("proxy_scores"):
                        scores = ", ".join(
                            f"{k}={v}" for k, v in cp["proxy_scores"].items()
                        )
                        proxy_info = f" (代理分数: {scores})"
                    console.print(f"  - {cp['id']}{proxy_info}")

            raise typer.Exit(0)

        # Process feedback
        console.print("[green]✓ 检测到用户反馈[/green]\n")

        console.print("[bold]Checkpoint 测试结果:[/bold]")
        for cp_id, results in feedback.get("checkpoint_results", {}).items():
            console.print(f"  [cyan]{cp_id}[/cyan]:")
            for k, v in results.items():
                console.print(f"    {k}: {v}")

        if feedback.get("overall_feedback"):
            console.print(f"\n[bold]总体反馈:[/bold] {feedback['overall_feedback']}")

        console.print(f"\n[bold]下一步:[/bold] {feedback.get('next_step', 'continue')}")

        # Confirm before continuing
        if not typer.confirm("\n确认这些反馈，继续迭代？"):
            console.print("[yellow]已取消[/yellow]")
            raise typer.Exit(0)

        console.print("\n[green]继续执行...[/green]\n")

    try:
        success = asyncio.run(resume(cwd=cwd, max_iterations=max_iterations, verbose=verbose, target_score=target_score, explore_mode=explore))
        raise typer.Exit(0 if success else 1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled by user[/yellow]")
        raise typer.Exit(130)


@app.command("status")
def status_cmd(
    cwd: str = typer.Option(".", "--cwd", "-C", help="Working directory"),
) -> None:
    """
    Show current session status.

    Example:
        ralph-sdk status
    """
    ralph_dir = Path(cwd) / ".ralph"

    if not ralph_dir.exists():
        console.print("[yellow]No Ralph session found in current directory[/yellow]")
        console.print("Run [cyan]ralph-sdk run \"your goal\"[/cyan] to start")
        raise typer.Exit(0)

    # Check if waiting for feedback
    if is_waiting_for_user(cwd):
        console.print("[bold yellow]Status: WAITING FOR USER FEEDBACK[/bold yellow]\n")
        manifest = read_checkpoint_manifest(cwd)
        if manifest:
            console.print(f"[bold]待测试 checkpoints:[/bold] {len(manifest.get('checkpoints', []))}")
            for cp in manifest.get("checkpoints", []):
                console.print(f"  - {cp['id']}")
            console.print()
            console.print("请编辑 [cyan].ralph/feedback.md[/cyan] 填写测试结果")
            console.print("然后运行 [cyan]ralph-sdk resume[/cyan] 继续")
        console.print()

    # Show goal
    if goal_exists(cwd):
        console.print("[bold]Goal:[/bold]")
        goal = read_goal(cwd)
        # Show first few lines
        lines = goal.split("\n")[:10]
        for line in lines:
            console.print(f"  {line}")
        if len(goal.split("\n")) > 10:
            console.print("  ...")
        console.print()

    # Show pool
    if pool_exists(cwd):
        console.print("[bold]Task Pool:[/bold]")
        pool = read_pool(cwd)
        console.print(pool)
    else:
        console.print("[yellow]Pool not initialized yet[/yellow]")

    # List task files
    tasks_dir = ralph_dir / "tasks"
    if tasks_dir.exists():
        task_files = list(tasks_dir.glob("T*.md"))
        if task_files:
            console.print(f"\n[bold]Task files:[/bold] {len(task_files)}")
            for f in sorted(task_files):
                console.print(f"  - {f.name}")


@app.command("clean")
def clean_cmd(
    cwd: str = typer.Option(".", "--cwd", "-C", help="Working directory"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
    archive: bool = typer.Option(False, "--archive", "-a", help="Archive session before cleaning"),
) -> None:
    """
    Remove .ralph/ directory to start fresh.

    Example:
        ralph-sdk clean
        ralph-sdk clean --force
        ralph-sdk clean --archive  # Save to history before cleaning
    """
    ralph_dir = Path(cwd) / ".ralph"

    if not ralph_dir.exists():
        console.print("[yellow]No .ralph/ directory found[/yellow]")
        raise typer.Exit(0)

    if not force:
        msg = "Archive and clean" if archive else "Delete"
        confirm = typer.confirm(f"{msg} .ralph/ directory and all task data?")
        if not confirm:
            console.print("[yellow]Cancelled[/yellow]")
            raise typer.Exit(0)

    if archive:
        archive_path = archive_session(cwd, keep_current=False)
        if archive_path:
            console.print(f"[green]✓ Archived to {archive_path}[/green]")

    import shutil

    # Clean remaining files (archive_session moves most, but clean any remnants)
    if ralph_dir.exists():
        # Only remove non-history items
        for item in ralph_dir.iterdir():
            if item.name != "history":
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        # Remove .ralph if empty (except history)
        remaining = [f for f in ralph_dir.iterdir() if f.name != "history"]
        if not remaining:
            # Keep history dir, just remove the rest
            pass
        console.print("[green]✓ Cleaned .ralph/ directory[/green]")
    else:
        console.print("[green]✓ Cleaned .ralph/ directory[/green]")


@app.command("logs")
def logs_cmd(
    cwd: str = typer.Option(".", "--cwd", "-C", help="Working directory"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed event log"),
) -> None:
    """
    Show session logs and metrics.

    Example:
        ralph-sdk logs
        ralph-sdk logs --detailed
    """
    summary = get_session_summary(cwd)

    if not summary:
        console.print("[yellow]No session logs found[/yellow]")
        console.print("Run [cyan]ralph-sdk run \"your goal\"[/cyan] to start a session")
        raise typer.Exit(0)

    console.print("[bold]Session Metrics:[/bold]\n")
    console.print(f"  Session ID:    {summary.get('session_id', 'N/A')}")
    console.print(f"  Status:        {summary.get('status', 'N/A')}")
    console.print(f"  Started:       {summary.get('started_at', 'N/A')}")
    if summary.get('ended_at'):
        console.print(f"  Ended:         {summary['ended_at']}")
    console.print(f"  Duration:      {summary.get('duration_seconds', 0):.1f}s")
    console.print()
    console.print(f"  Iterations:    {summary.get('total_iterations', 0)}")
    console.print(f"  Tool Calls:    {summary.get('total_tool_calls', 0)}")
    console.print(f"  Tasks Created: {summary.get('tasks_created', 0)}")
    console.print(f"  Tasks Done:    {summary.get('tasks_completed', 0)}")

    actions = summary.get('actions', {})
    if actions:
        console.print("\n[bold]Actions:[/bold]")
        for action, count in sorted(actions.items()):
            console.print(f"  {action}: {count}")

    if detailed:
        import json

        logs_dir = Path(cwd) / ".ralph" / "logs"
        log_files = list(logs_dir.glob("session_*.jsonl"))
        if log_files:
            latest_log = sorted(log_files)[-1]
            console.print(f"\n[bold]Event Log ({latest_log.name}):[/bold]\n")
            with open(latest_log) as f:
                for line in f:
                    event = json.loads(line)
                    ts = event.get('ts', '')[:19]  # Truncate to seconds
                    evt = event.get('event', '')
                    if evt == 'iteration_start':
                        console.print(f"  [{ts}] [cyan]Iteration {event.get('iteration')}[/cyan]")
                    elif evt == 'planner_decision':
                        console.print(f"  [{ts}] [yellow]Planner: {event.get('action')} {event.get('target', '')}[/yellow]")
                    elif evt == 'worker_complete':
                        status = "[green]✓[/green]" if event.get('success') else "[red]✗[/red]"
                        console.print(f"  [{ts}] {status} Worker: {event.get('task_id')} ({event.get('task_type')})")
                    elif evt == 'reviewer_verdict':
                        console.print(f"  [{ts}] [blue]Review: {event.get('verdict')}[/blue]")
                    elif evt == 'session_end':
                        status = "[green]completed[/green]" if event.get('success') else "[yellow]interrupted[/yellow]"
                        console.print(f"  [{ts}] Session {status}")


@app.command("history")
def history_cmd(
    cwd: str = typer.Option(".", "--cwd", "-C", help="Working directory"),
) -> None:
    """
    List archived sessions.

    Example:
        ralph-sdk history
    """
    sessions = list_archived_sessions(cwd)

    if not sessions:
        console.print("[yellow]No archived sessions found[/yellow]")
        console.print("Use [cyan]ralph-sdk clean --archive[/cyan] to archive sessions")
        raise typer.Exit(0)

    console.print(f"[bold]Archived Sessions ({len(sessions)}):[/bold]\n")

    for s in sessions:
        console.print(f"  [cyan]{s.get('name', 'unknown')}[/cyan]")
        if s.get('goal'):
            goal_preview = s['goal'][:60] + "..." if len(s.get('goal', '')) > 60 else s.get('goal', '')
            console.print(f"    Goal: {goal_preview}")
        if s.get('status'):
            console.print(f"    Status: {s['status']}")
        if s.get('total_iterations'):
            console.print(f"    Iterations: {s['total_iterations']}, Tool calls: {s.get('total_tool_calls', 0)}")
        console.print()


def main() -> None:
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
