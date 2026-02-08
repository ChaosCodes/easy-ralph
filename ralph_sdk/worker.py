"""
Worker agent: executes EXPLORE and IMPLEMENT tasks.

- EXPLORE: research, investigate, gather information
- IMPLEMENT: write code, make changes
"""

import re
from dataclasses import dataclass, field
from typing import Literal, Optional

from claude_agent_sdk import ClaudeAgentOptions
from rich.console import Console
from rich.panel import Panel

from .logger import format_tool_line, log_tool_call, shorten_path, stream_query
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
    result_stats: Optional[object] = None  # ResultMessage from SDK


def format_duration_ms(ms: int) -> str:
    """Format milliseconds as human-readable duration."""
    from .logger import format_duration
    return format_duration(ms / 1000)


def _format_result_tokens(result_stats) -> str:
    """Format tokens from a ResultMessage."""
    from .logger import format_tokens
    return format_tokens(result_stats.usage)


def extract_worker_metadata(text: str, task_type: str) -> dict:
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

    # Run worker query with unified streaming
    sr = await stream_query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            system_prompt=system_prompt,
            allowed_tools=tools,
            permission_mode=permission_mode,
            max_turns=50,
            cwd=cwd,
        ),
        agent_name="worker",
        emoji="💭",
        cwd=cwd,
        verbose=verbose,
    )
    result_text = sr.text
    tool_count = sr.tool_count
    result_stats = sr.result_stats

    # Parse result
    parsed = extract_worker_metadata(result_text, task_type)

    # Build stats line
    stats_parts = [f"{tool_count} tool calls"]
    if result_stats:
        if result_stats.duration_ms:
            stats_parts.append(format_duration_ms(result_stats.duration_ms))
        tokens_str = _format_result_tokens(result_stats)
        if tokens_str:
            stats_parts.append(tokens_str)
        if result_stats.total_cost_usd:
            stats_parts.append(f"${result_stats.total_cost_usd:.2f}")
    stats = f"[bright_black]{' · '.join(stats_parts)}[/bright_black]"

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
        result_stats=result_stats,
    )
