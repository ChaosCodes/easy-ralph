"""
Worker agent: executes EXPLORE and IMPLEMENT tasks.

- EXPLORE: research, investigate, gather information
- IMPLEMENT: write code, make changes
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from claude_agent_sdk import AgentDefinition, ClaudeAgentOptions, SandboxSettings
from rich.console import Console
from rich.panel import Panel

from .logger import format_tool_line, log_tool_call, shorten_path, stream_query
from .mcp_tools import create_ralph_mcp_server, get_tool_names
from .pool import (
    read_goal,
    read_pool,
    read_task,
)
from .prompts import (
    WORKER_ADVERSARIAL_INVESTIGATION_SECTION,
    WORKER_EXPLORE_PROMPT,
    WORKER_IMPLEMENT_PROMPT,
    build_worker_prompt,
)

console = Console()

# Default domains allowed through sandbox network proxy.
# The sandbox proxy blocks all outbound HTTP unless the domain has a
# WebFetch permission rule. These defaults cover common needs.
DEFAULT_SANDBOX_ALLOWED_DOMAINS = [
    "api.anthropic.com",
    "pypi.org",
    "files.pythonhosted.org",
]

# Subagents available to worker via the Task tool
WORKER_SUBAGENTS = {
    "researcher": AgentDefinition(
        description="Deep research agent for investigating specific technical questions. "
                    "Use when you need thorough investigation of a topic, API, or library.",
        prompt="You are a research specialist. Investigate thoroughly and report findings.",
        tools=["Read", "Glob", "Grep", "WebSearch", "WebFetch"],
        model="sonnet",
    ),
    "test-writer": AgentDefinition(
        description="Test writing specialist. Use to write comprehensive test cases "
                    "for implemented code.",
        prompt="You are a test writing specialist. Write thorough pytest tests.",
        tools=["Read", "Glob", "Grep", "Write", "Edit"],
        model="sonnet",
    ),
}

WORKER_EXPLORE_SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "findings_summary": {"type": "string"},
        },
        "required": ["confidence"],
    },
}


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


def extract_worker_metadata(structured_output: dict | None, text: str, task_type: str) -> dict:
    """Parse worker output to extract key information.

    Prefers structured_output, falls back to regex extraction.
    Worker always returns success=True — the Reviewer determines actual success.
    """
    result = {"success": True}
    if structured_output and "confidence" in structured_output:
        result["confidence"] = structured_output["confidence"]
    else:
        confidence_match = re.search(r"Confidence:\s*(high|medium|low)", text, re.IGNORECASE)
        if confidence_match:
            result["confidence"] = confidence_match.group(1).lower()
    return result


async def work(
    task_id: str,
    task_type: Literal["EXPLORE", "IMPLEMENT"],
    cwd: str = ".",
    verbose: bool = False,
    use_mcp: bool = False,
    thinking_budget: int | None = None,
    adversarial_findings: Optional[str] = None,
    no_sandbox: bool = False,
    sandbox_allowed_domains: list[str] | None = None,
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

    # Inject adversarial findings for IMPLEMENT tasks when available
    if adversarial_findings and task_type == "IMPLEMENT":
        from .pool import AUDITS_DIR
        adversarial_section = WORKER_ADVERSARIAL_INVESTIGATION_SECTION.format(
            audits_dir=str(Path(cwd) / AUDITS_DIR),
            task_id=task_id,
            attempt="current",
            findings_content=adversarial_findings,
        )
        prompt += "\n" + adversarial_section

    # Configure tools based on task type (skip MCP if use_mcp=False)
    core_tools = [
        "Read", "Glob", "Grep", "LSP",
        "WebFetch", "WebSearch", "Task",
    ]
    if use_mcp:
        ralph_mcp = create_ralph_mcp_server(cwd, role="worker")
        ralph_tool_names = get_tool_names("worker")
        base_tools = core_tools + ralph_tool_names
        mcp_servers = {"ralph": ralph_mcp}
    else:
        base_tools = core_tools
        mcp_servers = None

    # Build sandbox network settings: WebFetch rules control the sandbox proxy's
    # domain allowlist. Without these, all outbound HTTP from Bash subprocesses
    # gets 403'd by the sandbox proxy.
    domains = sandbox_allowed_domains if sandbox_allowed_domains is not None else DEFAULT_SANDBOX_ALLOWED_DOMAINS
    sandbox_settings_json = None
    if domains:
        sandbox_settings_json = json.dumps({
            "permissions": {
                "allow": [f"WebFetch(domain:{d})" for d in domains]
            }
        })

    if task_type == "IMPLEMENT":
        tools = base_tools + ["Write", "Edit", "Bash"]
        permission_mode = "acceptEdits"
        if no_sandbox:
            sandbox = None
        else:
            sandbox = SandboxSettings(
                enabled=True,
                autoAllowBashIfSandboxed=True,
                allowUnsandboxedCommands=True,
                # python/pip/git run outside sandbox to avoid Seatbelt restrictions
                excludedCommands=["python3", "python", "pip", "pip3", "git"],
            )
        agents = WORKER_SUBAGENTS
    else:
        # EXPLORE doesn't have Bash, acceptEdits is fine
        tools = base_tools + ["Write", "Edit"]
        permission_mode = "acceptEdits"
        sandbox = None
        agents = None

    # Only EXPLORE gets structured output; IMPLEMENT needs free-form text
    worker_output_format = WORKER_EXPLORE_SCHEMA if task_type == "EXPLORE" else None

    # Configure thinking for worker (default: 0 — mostly tool calls)
    worker_options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        allowed_tools=tools,
        permission_mode=permission_mode,
        sandbox=sandbox,
        settings=sandbox_settings_json,
        enable_file_checkpointing=task_type == "IMPLEMENT",
        agents=agents,
        max_turns=50,
        cwd=cwd,
        output_format=worker_output_format,
        mcp_servers=mcp_servers,
    )
    effective = thinking_budget if thinking_budget is not None else 0
    if effective > 0:
        worker_options.max_thinking_tokens = effective

    # Run worker query with unified streaming
    sr = await stream_query(
        prompt=prompt,
        options=worker_options,
        agent_name="worker",
        emoji="💭",
        cwd=cwd,
        verbose=verbose,
    )
    result_text = sr.text
    tool_count = sr.tool_count
    result_stats = sr.result_stats

    # Parse result
    parsed = extract_worker_metadata(sr.structured_output, result_text, task_type)

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
