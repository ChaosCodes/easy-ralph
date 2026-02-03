"""Story execution module."""

import json
import os
import re

from claude_code_sdk import AssistantMessage, ClaudeCodeOptions, query
from rich.console import Console
from rich.panel import Panel

from ralph_sdk.models import ExecutionContext, StoryResult, UserStory
from ralph_sdk.prompts import EXECUTOR_SYSTEM_PROMPT

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
                # File operations
                "Read",
                "Write",
                "Edit",
                "Glob",
                "Grep",
                # Execution
                "Bash",
                # Code intelligence
                "LSP",          # Go to definition, find references, etc.
                "Task",         # Spawn sub-agents for complex exploration
                # Web access
                "WebFetch",     # Fetch external documentation
                "WebSearch",    # Search for solutions
                # Notebook support
                "NotebookEdit", # Edit Jupyter notebooks if needed
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

    # Parse and display the result
    result = parse_execution_result(story.id, result_text)
    console.print()
    _display_result(result, story.id, turn_count, tool_count, context.cwd)

    return result


def _display_result(
    result: StoryResult, story_id: str, turn_count: int, tool_count: int, cwd: str
) -> None:
    """Display the execution result summary."""
    stats = f"[bright_black]{turn_count} turns, {tool_count} tool calls[/bright_black]"

    if result.success:
        files_display = _format_files_list(result.files_changed, cwd)
        content = f"[bold bright_green]Success[/bold bright_green]  {stats}\n"
        content += f"[bright_black]Files:[/bright_black] {files_display}\n"
        if result.commit_hash:
            content += f"[bright_black]Commit:[/bright_black] [bright_cyan]{result.commit_hash[:8]}[/bright_cyan]\n"
        content += _format_learnings(result.learnings)
        console.print(Panel(content.strip(), title=f"[bold bright_green]{story_id}[/bold bright_green]", border_style="bright_green"))
    else:
        content = f"[bold bright_red]Failed[/bold bright_red]  {stats}\n"
        content += f"[bright_black]Error:[/bright_black] [bright_red]{result.error or 'Unknown'}[/bright_red]\n"
        content += _format_learnings(result.learnings)
        console.print(Panel(content.strip(), title=f"[bold bright_red]{story_id}[/bold bright_red]", border_style="bright_red"))


def _format_files_list(files: list[str], cwd: str) -> str:
    """Format a list of files for display."""
    if not files:
        return "[dim]None[/dim]"
    return ", ".join(f"[bright_white]{shorten_path(f, cwd)}[/bright_white]" for f in files)


def _format_learnings(learnings: list[str]) -> str:
    """Format learnings for display."""
    if not learnings:
        return ""
    lines = ["[bright_black]Learnings:[/bright_black]"]
    for learning in learnings[:3]:
        lines.append(f"  [bright_yellow]-[/bright_yellow] {learning}")
    return "\n".join(lines) + "\n"


_SUCCESS_INDICATORS = ["successfully", "commit", "all checks pass", "typecheck passes", "tests pass"]
_FAILURE_INDICATORS = ["error", "failed", "cannot", "unable"]


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
    text_lower = text.lower()
    has_success = any(ind in text_lower for ind in _SUCCESS_INDICATORS)
    has_failure = any(ind in text_lower for ind in _FAILURE_INDICATORS)
    success = has_success and not has_failure

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
