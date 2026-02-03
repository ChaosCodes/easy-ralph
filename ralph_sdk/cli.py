"""Command-line interface for Ralph SDK."""

import asyncio
from pathlib import Path

import typer
from rich.console import Console

from ralph_sdk.clarifier import clarify_requirements, quick_clarify
from ralph_sdk.orchestrator import run_from_prd_file, run_ralph
from ralph_sdk.prd_generator import display_prd, generate_prd, load_prd, save_prd

app = typer.Typer(
    name="ralph-sdk",
    help="Ralph autonomous agent loop using Claude Agent SDK",
    add_completion=False,
)
console = Console()


@app.command()
def run(
    prompt: str = typer.Argument(..., help="Feature description or requirement"),
    cwd: str = typer.Option(".", "--cwd", "-C", help="Working directory"),
    max_iterations: int = typer.Option(10, "--max", "-n", help="Maximum iterations"),
    skip_clarify: bool = typer.Option(
        False, "--quick", "-q", help="Skip clarification phase"
    ),
    prd_file: str = typer.Option("prd.json", "--prd", help="PRD file path"),
    progress_file: str = typer.Option(
        "progress.txt", "--progress", help="Progress file path"
    ),
) -> None:
    """
    Run the complete Ralph workflow: clarify → generate PRD → execute.

    Example:
        ralph-sdk run "Add user authentication with JWT"
    """
    try:
        success = asyncio.run(
            run_ralph(
                initial_prompt=prompt,
                cwd=cwd,
                max_iterations=max_iterations,
                skip_clarify=skip_clarify,
                prd_file=prd_file,
                progress_file=progress_file,
            )
        )
        raise typer.Exit(0 if success else 1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled by user[/yellow]")
        raise typer.Exit(130)


@app.command()
def plan(
    prompt: str = typer.Argument(..., help="Feature description"),
    cwd: str = typer.Option(".", "--cwd", "-C", help="Working directory"),
    output: str = typer.Option("prd.json", "--output", "-o", help="Output file"),
    skip_clarify: bool = typer.Option(
        False, "--quick", "-q", help="Skip clarification phase"
    ),
) -> None:
    """
    Generate a PRD without executing (planning only).

    Example:
        ralph-sdk plan "Add task priority feature" -o tasks/prd-priority.json
    """
    try:
        prd = asyncio.run(_generate_prd(prompt, cwd, output, skip_clarify))
        console.print(f"\n[green]PRD saved to {output}[/green]")
        console.print(f"Run with: [cyan]ralph-sdk execute {output}[/cyan]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled by user[/yellow]")
        raise typer.Exit(130)


async def _generate_prd(prompt: str, cwd: str, output: str, skip_clarify: bool):
    """Helper to generate and save a PRD."""
    if skip_clarify:
        requirements = await quick_clarify(prompt)
    else:
        requirements = await clarify_requirements(prompt)

    prd = await generate_prd(requirements, cwd=cwd)
    save_prd(prd, output)
    return prd


@app.command()
def execute(
    prd_path: str = typer.Argument(..., help="Path to prd.json"),
    max_iterations: int = typer.Option(10, "--max", "-n", help="Maximum iterations"),
    progress_file: str = typer.Option(
        "progress.txt", "--progress", help="Progress file path"
    ),
) -> None:
    """
    Execute stories from an existing PRD file.

    Example:
        ralph-sdk execute prd.json --max 20
    """
    if not Path(prd_path).exists():
        console.print(f"[red]Error: PRD file not found: {prd_path}[/red]")
        raise typer.Exit(1)

    try:
        success = asyncio.run(
            run_from_prd_file(
                prd_path=prd_path,
                max_iterations=max_iterations,
                progress_file=progress_file,
            )
        )
        raise typer.Exit(0 if success else 1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled by user[/yellow]")
        raise typer.Exit(130)


@app.command()
def status(
    prd_path: str = typer.Option("prd.json", "--prd", help="PRD file path"),
) -> None:
    """
    Show current progress status.

    Example:
        ralph-sdk status
    """
    if not Path(prd_path).exists():
        console.print(f"[yellow]No PRD found at {prd_path}[/yellow]")
        raise typer.Exit(0)

    prd = load_prd(prd_path)
    display_prd(prd)

    console.print(f"\n[bold]Progress:[/bold] {prd.progress_summary()}")

    pending = prd.get_pending_stories()
    if pending:
        console.print(f"\n[bold]Next story:[/bold] {pending[0].id} - {pending[0].title}")
    else:
        console.print("\n[bold green]All stories complete![/bold green]")


@app.command()
def reset(
    prd_path: str = typer.Option("prd.json", "--prd", help="PRD file path"),
    story_id: str = typer.Option(None, "--story", "-s", help="Reset specific story"),
    all_stories: bool = typer.Option(False, "--all", "-a", help="Reset all stories"),
) -> None:
    """
    Reset story status to re-run them.

    Example:
        ralph-sdk reset --story US-003
        ralph-sdk reset --all
    """
    if not Path(prd_path).exists():
        console.print(f"[red]Error: PRD file not found: {prd_path}[/red]")
        raise typer.Exit(1)

    prd = load_prd(prd_path)

    if all_stories:
        for story in prd.user_stories:
            story.passes = False
            story.notes = ""
        console.print("[green]All stories reset[/green]")
    elif story_id:
        for story in prd.user_stories:
            if story.id == story_id:
                story.passes = False
                story.notes = ""
                console.print(f"[green]Story {story_id} reset[/green]")
                break
        else:
            console.print(f"[red]Story {story_id} not found[/red]")
            raise typer.Exit(1)
    else:
        console.print("[yellow]Specify --story or --all[/yellow]")
        raise typer.Exit(1)

    save_prd(prd, prd_path)


def main() -> None:
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
