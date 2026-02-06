"""
Worker agent: executes EXPLORE and IMPLEMENT tasks.

- EXPLORE: research, investigate, gather information
- IMPLEMENT: write code, make changes
"""

import os
import re
from dataclasses import dataclass, field
from typing import Literal, Optional

from claude_code_sdk import AssistantMessage, ClaudeCodeOptions, query
from rich.console import Console
from rich.panel import Panel

from .pool import (
    add_verified_info,
    get_verified_info,
    is_topic_verified,
    read_goal,
    read_pool,
    read_task,
)
from .prompts import (
    WORKER_EXPLORE_PROMPT,
    WORKER_IMPLEMENT_PROMPT,
    build_worker_prompt,
)

console = Console()


@dataclass
class WorkerResult:
    """Result of worker execution.

    Note: Detailed findings, files_changed, follow_up_tasks are written
    to task files (tasks/*.md) and pool.md - not returned here.
    This follows the design of using filesystem as cross-worker memory.
    """
    success: bool
    task_id: str
    task_type: str
    confidence: Optional[str] = None  # high / medium / low (for EXPLORE)
    error: Optional[str] = None


# --- Output formatting (from old executor.py) ---

def shorten_path(path: str, cwd: str = "") -> str:
    """Shorten file path for display."""
    if cwd and path.startswith(cwd):
        path = path[len(cwd):].lstrip("/")
    home = os.path.expanduser("~")
    if path.startswith(home):
        path = "~" + path[len(home):]
    return path


def format_tool_line(tool_name: str, tool_input: dict, cwd: str = "") -> str:
    """Format a tool use as a single compact line with rich highlighting."""
    if tool_name == "Read":
        path = shorten_path(tool_input.get("file_path", "?"), cwd)
        return f"[bold cyan]Read[/bold cyan] {path}"

    if tool_name == "Write":
        path = shorten_path(tool_input.get("file_path", "?"), cwd)
        lines = len(tool_input.get("content", "").split("\n"))
        return f"[bold green]Write[/bold green] {path} [dim]({lines} lines)[/dim]"

    if tool_name == "Edit":
        path = shorten_path(tool_input.get("file_path", "?"), cwd)
        old_lines = len(tool_input.get("old_string", "").split("\n"))
        new_lines = len(tool_input.get("new_string", "").split("\n"))
        return f"[bold yellow]Edit[/bold yellow] {path} [red]-{old_lines}[/red] [green]+{new_lines}[/green]"

    if tool_name == "Bash":
        cmd = tool_input.get("command", "?")
        if len(cmd) > 60:
            cmd = cmd[:57] + "..."
        return f"[bold magenta]Bash[/bold magenta] {cmd}"

    if tool_name == "Glob":
        pattern = tool_input.get("pattern", "?")
        path = tool_input.get("path", "")
        suffix = f" [dim]in {shorten_path(path, cwd)}[/dim]" if path else ""
        return f"[bold blue]Glob[/bold blue] {pattern}{suffix}"

    if tool_name == "Grep":
        pattern = tool_input.get("pattern", "?")
        path = shorten_path(tool_input.get("path", "."), cwd)
        return f"[bold blue]Grep[/bold blue] '{pattern}' [dim]in {path}[/dim]"

    if tool_name == "Task":
        desc = tool_input.get("description", "?")
        subagent = tool_input.get("subagent_type", "")
        return f"[bold cyan]Task[/bold cyan] {desc} [dim]({subagent})[/dim]"

    if tool_name == "LSP":
        op = tool_input.get("operation", "?")
        path = shorten_path(tool_input.get("filePath", "?"), cwd)
        line = tool_input.get("line", "?")
        return f"[bold green]LSP[/bold green] {op} {path}:{line}"

    if tool_name == "WebFetch":
        url = tool_input.get("url", "?")
        if len(url) > 50:
            url = url[:47] + "..."
        return f"[bold blue]WebFetch[/bold blue] {url}"

    if tool_name == "WebSearch":
        q = tool_input.get("query", "?")
        return f"[bold blue]WebSearch[/bold blue] '{q}'"

    if tool_name == "NotebookEdit":
        path = shorten_path(tool_input.get("notebook_path", "?"), cwd)
        edit_mode = tool_input.get("edit_mode", "replace")
        return f"[bold purple]NotebookEdit[/bold purple] {path} [dim]({edit_mode})[/dim]"

    return f"[bold]{tool_name}[/bold]"


def parse_worker_output(text: str, task_type: str) -> dict:
    """Parse worker output to extract key information.

    Note: Worker always returns success=True. The Reviewer is responsible
    for determining if the task actually succeeded. This avoids fragile
    heuristics based on text pattern matching.
    """
    result = {"success": True}  # Always succeed, let Reviewer decide

    # Extract confidence (for EXPLORE)
    confidence_match = re.search(r"Confidence:\s*(high|medium|low)", text, re.IGNORECASE)
    if confidence_match:
        result["confidence"] = confidence_match.group(1).lower()

    return result


async def work(
    task_id: str,
    task_type: Literal["EXPLORE", "IMPLEMENT"],
    cwd: str = ".",
    verbose: bool = False,
) -> WorkerResult:
    """
    Execute a task (EXPLORE or IMPLEMENT).

    Args:
        task_id: The task ID (e.g., "T001")
        task_type: Either "EXPLORE" or "IMPLEMENT"
        cwd: Working directory

    Returns:
        WorkerResult with execution outcome
    """
    # Read context
    goal = read_goal(cwd)
    pool = read_pool(cwd)
    task_detail = read_task(task_id, cwd)

    if not task_detail:
        return WorkerResult(
            success=False,
            task_id=task_id,
            task_type=task_type,
            error=f"Task file not found: tasks/{task_id}.md",
        )

    # Display task panel
    task_title = task_detail.split('\n')[0].replace('# ', '')
    console.print(
        Panel(
            f"[bold bright_cyan]{task_id}[/bold bright_cyan] [bright_white]{task_title}[/bright_white]\n\n[dim]{task_type} task[/dim]",
            title=f"[bold]{'🔍 Exploring' if task_type == 'EXPLORE' else '🚀 Executing'}[/bold]",
            border_style="bright_blue",
        )
    )
    console.print()

    # Select system prompt based on task type
    system_prompt = WORKER_EXPLORE_PROMPT if task_type == "EXPLORE" else WORKER_IMPLEMENT_PROMPT

    # Build user prompt
    prompt = build_worker_prompt(
        task_id=task_id,
        task_type=task_type,
        goal=goal,
        pool=pool,
        task_detail=task_detail,
    )

    # Configure tools based on task type
    base_tools = [
        "Read", "Glob", "Grep", "LSP",
        "WebFetch", "WebSearch", "Task",
    ]

    if task_type == "IMPLEMENT":
        tools = base_tools + ["Write", "Edit", "Bash"]
        permission_mode = "acceptEdits"
    else:
        # EXPLORE needs Write/Edit to update task files and pool.md
        tools = base_tools + ["Write", "Edit"]
        permission_mode = "acceptEdits"

    # Run worker query with detailed output
    result_text = ""
    turn_count = 0
    tool_count = 0

    async for message in query(
        prompt=prompt,
        options=ClaudeCodeOptions(
            system_prompt=system_prompt,
            allowed_tools=tools,
            permission_mode=permission_mode,
            max_turns=50,
            cwd=cwd,
        ),
    ):
        if isinstance(message, AssistantMessage):
            turn_count += 1

            for block in message.content:
                # Handle text content
                if hasattr(block, "text") and block.text:
                    result_text += block.text
                    if verbose:
                        # Show brief summary of thinking
                        text = block.text.strip()
                        if text and len(text) > 20:
                            lines = text.split('\n')
                            first_line = lines[0][:80]
                            if len(lines[0]) > 80:
                                first_line += "..."
                            console.print(f"     [italic bright_black]💭 {first_line}[/italic bright_black]")

                # Handle tool use
                if hasattr(block, "name") and hasattr(block, "input"):
                    tool_count += 1
                    if verbose:
                        tool_line = format_tool_line(block.name, block.input, cwd)
                        console.print(f"[bright_black][{tool_count:2d}][/bright_black] {tool_line}")

    # Parse result
    parsed = parse_worker_output(result_text, task_type)

    # Display summary
    stats = f"[bright_black]{turn_count} turns, {tool_count} tool calls[/bright_black]"
    if parsed.get("success", True):
        console.print(
            Panel(
                f"[bold bright_green]Success[/bold bright_green]  {stats}",
                title=f"[bold bright_green]{task_id}[/bold bright_green]",
                border_style="bright_green",
            )
        )
    else:
        console.print(
            Panel(
                f"[bold bright_red]Failed[/bold bright_red]  {stats}\n[dim]{parsed.get('error', 'Unknown error')}[/dim]",
                title=f"[bold bright_red]{task_id}[/bold bright_red]",
                border_style="bright_red",
            )
        )

    return WorkerResult(
        success=parsed.get("success", True),
        task_id=task_id,
        task_type=task_type,
        confidence=parsed.get("confidence"),
        error=parsed.get("error"),
    )
