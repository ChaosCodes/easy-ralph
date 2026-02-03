"""Story execution module."""

import json
import os
import re
from claude_code_sdk import query, ClaudeCodeOptions, AssistantMessage
from ralph_sdk.models import UserStory, StoryResult, ExecutionContext
from ralph_sdk.prompts import EXECUTOR_SYSTEM_PROMPT
from rich.console import Console
from rich.panel import Panel

console = Console()


def shorten_path(path: str, cwd: str = "") -> str:
    """Shorten file path for display."""
    if cwd and path.startswith(cwd):
        path = path[len(cwd):].lstrip("/")
    # Also try to shorten common prefixes
    home = os.path.expanduser("~")
    if path.startswith(home):
        path = "~" + path[len(home):]
    return path


def format_tool_line(tool_name: str, tool_input: dict, cwd: str = "") -> str:
    """Format a tool use as a single compact line with rich highlighting."""
    icons = {
        "Read": "📖",
        "Write": "✍️ ",
        "Edit": "✏️ ",
        "Bash": "💻",
        "Glob": "🔍",
        "Grep": "🔎",
    }
    icon = icons.get(tool_name, "🔧")

    if tool_name == "Read":
        path = shorten_path(tool_input.get('file_path', '?'), cwd)
        return f"{icon} [bold cyan]Read[/bold cyan] [bright_white]{path}[/bright_white]"

    elif tool_name == "Write":
        path = shorten_path(tool_input.get('file_path', '?'), cwd)
        content = tool_input.get('content', '')
        lines = len(content.split('\n'))
        return f"{icon} [bold green]Write[/bold green] [bright_white]{path}[/bright_white] [dim]({lines} lines)[/dim]"

    elif tool_name == "Edit":
        path = shorten_path(tool_input.get('file_path', '?'), cwd)
        old = tool_input.get('old_string', '')
        new = tool_input.get('new_string', '')
        old_lines = len(old.split('\n'))
        new_lines = len(new.split('\n'))
        return f"{icon} [bold yellow]Edit[/bold yellow] [bright_white]{path}[/bright_white] [red]-{old_lines}[/red] [green]+{new_lines}[/green]"

    elif tool_name == "Bash":
        cmd = tool_input.get('command', '?')
        # Truncate long commands
        if len(cmd) > 60:
            cmd = cmd[:57] + "..."
        return f"{icon} [bold magenta]Bash[/bold magenta] [bright_black on bright_white] {cmd} [/bright_black on bright_white]"

    elif tool_name == "Glob":
        pattern = tool_input.get('pattern', '?')
        path = tool_input.get('path', '')
        if path:
            return f"{icon} [bold blue]Glob[/bold blue] [bright_yellow]{pattern}[/bright_yellow] [dim]in {shorten_path(path, cwd)}[/dim]"
        return f"{icon} [bold blue]Glob[/bold blue] [bright_yellow]{pattern}[/bright_yellow]"

    elif tool_name == "Grep":
        pattern = tool_input.get('pattern', '?')
        path = shorten_path(tool_input.get('path', '.'), cwd)
        return f"{icon} [bold blue]Grep[/bold blue] [bright_yellow]'{pattern}'[/bright_yellow] [dim]in {path}[/dim]"

    else:
        return f"{icon} [bold]{tool_name}[/bold]"


async def execute_story(
    story: UserStory,
    context: ExecutionContext,
    progress_context: str = "",
) -> StoryResult:
    """
    Execute a single user story.

    Args:
        story: The user story to implement
        context: Execution context (cwd, branch, etc.)
        progress_context: Learnings from previous iterations

    Returns:
        StoryResult with success status and learnings
    """
    console.print(
        Panel(
            f"[bold bright_cyan]{story.id}[/bold bright_cyan] [bright_white]{story.title}[/bright_white]\n\n[dim]{story.description}[/dim]",
            title="[bold]🚀 Executing Story[/bold]",
            border_style="bright_blue",
        )
    )

    # Format the system prompt with story details
    system_prompt = EXECUTOR_SYSTEM_PROMPT.format(
        story=story.to_prompt(),
        story_id=story.id,
        story_title=story.title,
    )

    # Build the execution prompt
    execution_prompt = f"""## Context

Branch: {context.branch_name}
Working directory: {context.cwd}

## Previous Learnings
{progress_context if progress_context else "No previous learnings."}

## Task

Implement user story {story.id}: {story.title}

{story.to_prompt()}

## Instructions

1. First, explore the codebase to understand relevant patterns
2. Implement the changes
3. Run quality checks (typecheck, lint, tests as appropriate)
4. If all checks pass, commit the changes
5. Output your result as JSON

Begin implementation.
"""

    result_text = ""
    turn_count = 0
    tool_count = 0

    console.print()

    async for message in query(
        prompt=execution_prompt,
        options=ClaudeCodeOptions(
            system_prompt=system_prompt,
            allowed_tools=[
                "Read",
                "Write",
                "Edit",
                "Bash",
                "Glob",
                "Grep",
            ],
            permission_mode="acceptEdits",
            max_turns=50,
            cwd=context.cwd,
        ),
    ):
        if isinstance(message, AssistantMessage):
            turn_count += 1

            for block in message.content:
                # Handle text content
                if hasattr(block, "text") and block.text:
                    result_text += block.text
                    # Only show meaningful text (skip short fragments)
                    text = block.text.strip()
                    if text and len(text) > 20:
                        # Show a brief summary of what Claude is thinking
                        lines = text.split('\n')
                        first_line = lines[0][:80]
                        if len(lines[0]) > 80:
                            first_line += "..."
                        console.print(f"     [italic bright_black]💭 {first_line}[/italic bright_black]")

                # Handle tool use
                if hasattr(block, "name") and hasattr(block, "input"):
                    tool_count += 1
                    tool_line = format_tool_line(block.name, block.input, context.cwd)
                    console.print(f"[bright_black][{tool_count:2d}][/bright_black] {tool_line}")

    # Parse the result
    result = parse_execution_result(story.id, result_text)

    # Display result summary
    console.print()
    stats = f"[bright_black]{turn_count} turns, {tool_count} tool calls[/bright_black]"

    if result.success:
        files_str = ", ".join(f"[bright_white]{shorten_path(f, context.cwd)}[/bright_white]" for f in result.files_changed) if result.files_changed else "[dim]None[/dim]"
        content = f"[bold bright_green]✓ Success[/bold bright_green]  {stats}\n"
        content += f"[bright_black]Files:[/bright_black] {files_str}\n"
        if result.commit_hash:
            content += f"[bright_black]Commit:[/bright_black] [bright_cyan]{result.commit_hash[:8]}[/bright_cyan]\n"
        if result.learnings:
            content += f"[bright_black]Learnings:[/bright_black]\n"
            for l in result.learnings[:3]:  # Show max 3 learnings
                content += f"  [bright_yellow]•[/bright_yellow] {l}\n"
        console.print(Panel(content.strip(), title=f"[bold bright_green]{story.id}[/bold bright_green]", border_style="bright_green"))
    else:
        content = f"[bold bright_red]✗ Failed[/bold bright_red]  {stats}\n"
        content += f"[bright_black]Error:[/bright_black] [bright_red]{result.error or 'Unknown'}[/bright_red]\n"
        if result.learnings:
            content += f"[bright_black]Learnings:[/bright_black]\n"
            for l in result.learnings[:3]:
                content += f"  [bright_yellow]•[/bright_yellow] {l}\n"
        console.print(Panel(content.strip(), title=f"[bold bright_red]{story.id}[/bold bright_red]", border_style="bright_red"))

    return result


def parse_execution_result(story_id: str, text: str) -> StoryResult:
    """Parse execution result from Claude's response."""
    # Try to find JSON in the response
    json_match = re.search(r"\{[\s\S]*?\}", text)

    if json_match:
        try:
            data = json.loads(json_match.group(0))
            return StoryResult(
                story_id=story_id,
                success=data.get("success", False),
                error=data.get("error"),
                learnings=data.get("learnings", []),
                files_changed=data.get("files_changed", []),
                commit_hash=data.get("commit_hash"),
            )
        except json.JSONDecodeError:
            pass

    # Fallback: analyze the text for success indicators
    success_indicators = [
        "successfully",
        "commit",
        "all checks pass",
        "typecheck passes",
        "tests pass",
    ]
    failure_indicators = ["error", "failed", "cannot", "unable"]

    text_lower = text.lower()
    success = any(ind in text_lower for ind in success_indicators) and not any(
        ind in text_lower for ind in failure_indicators
    )

    return StoryResult(
        story_id=story_id,
        success=success,
        error=None if success else "Execution may have failed - check output",
        learnings=[],
        files_changed=[],
    )


async def verify_story(story: UserStory, context: ExecutionContext) -> bool:
    """
    Verify that a story's acceptance criteria are met.

    This is a separate verification step that can be run after execution.
    """
    verification_prompt = f"""Verify that the following acceptance criteria are met:

{story.to_prompt()}

Check each criterion and report:
1. Which criteria pass
2. Which criteria fail
3. Overall verdict: PASS or FAIL

Output format:
```json
{{
  "criteria_results": [
    {{"criterion": "...", "passes": true/false, "reason": "..."}}
  ],
  "overall_pass": true/false
}}
```
"""

    result_text = ""
    async for message in query(
        prompt=verification_prompt,
        options=ClaudeCodeOptions(
            allowed_tools=["Read", "Bash", "Glob", "Grep"],
            max_turns=10,
            cwd=context.cwd,
        ),
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    result_text += block.text

    # Parse result
    json_match = re.search(r"\{[\s\S]*?\}", result_text)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            return data.get("overall_pass", False)
        except json.JSONDecodeError:
            pass

    return "pass" in result_text.lower() and "fail" not in result_text.lower()
