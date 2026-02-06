"""
Reviewer agent: verifies IMPLEMENT task results.

Possible verdicts:
- PASSED: task complete
- RETRY: non-fundamental issue, can retry
- FAILED: fundamental issue, need different approach
"""

import json
import re
from dataclasses import dataclass
from enum import Enum

from claude_code_sdk import AssistantMessage, ClaudeCodeOptions, query
from rich.console import Console

from .pool import read_goal, read_task
from .prompts import REVIEWER_SYSTEM_PROMPT, build_reviewer_prompt
from .worker import format_tool_line

console = Console()


class Verdict(Enum):
    PASSED = "passed"
    RETRY = "retry"
    FAILED = "failed"


@dataclass
class ReviewResult:
    verdict: Verdict
    reason: str
    suggestions: str = ""


def _extract_json(text: str) -> dict | None:
    """Extract a JSON object from text, handling common LLM output patterns."""
    # Pattern 1: JSON in markdown code fence
    fence_match = re.search(r"```(?:json)?\s*\n(\{.*?\})\s*\n```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Pattern 2: Bare JSON object
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass

    return None


def parse_reviewer_output(text: str) -> ReviewResult:
    """Parse reviewer output into a ReviewResult.

    Tries JSON first (more reliable), falls back to regex for backwards compatibility.
    """
    # Try JSON parsing first
    json_obj = _extract_json(text)
    if json_obj and "verdict" in json_obj:
        verdict_str = json_obj["verdict"].lower()
        try:
            verdict = Verdict(verdict_str)
        except ValueError:
            verdict = Verdict.PASSED
        return ReviewResult(
            verdict=verdict,
            reason=json_obj.get("reason", ""),
            suggestions=json_obj.get("suggestions", ""),
        )

    # Fallback: regex parsing
    verdict_match = re.search(r"VERDICT:\s*(\w+)", text, re.IGNORECASE)
    verdict_str = verdict_match.group(1).lower() if verdict_match else "passed"

    try:
        verdict = Verdict(verdict_str)
    except ValueError:
        verdict = Verdict.PASSED

    reason_match = re.search(r"REASON:\s*(.+?)(?=\n(?:SUGGESTIONS:|$))", text, re.IGNORECASE | re.DOTALL)
    reason = reason_match.group(1).strip() if reason_match else ""

    suggestions_match = re.search(r"SUGGESTIONS:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    suggestions = suggestions_match.group(1).strip() if suggestions_match else ""

    return ReviewResult(
        verdict=verdict,
        reason=reason,
        suggestions=suggestions,
    )


async def review(task_id: str, cwd: str = ".", verbose: bool = False) -> ReviewResult:
    """
    Review an IMPLEMENT task after execution.

    Args:
        task_id: The task ID to review
        cwd: Working directory

    Returns:
        ReviewResult with verdict and explanation
    """
    console.print(f"[dim]Reviewing {task_id}...[/dim]")

    # Read context
    goal = read_goal(cwd)
    task_detail = read_task(task_id, cwd)

    if not task_detail:
        return ReviewResult(
            verdict=Verdict.FAILED,
            reason=f"Task file not found: tasks/{task_id}.md",
        )

    # Build prompt
    prompt = build_reviewer_prompt(
        task_id=task_id,
        goal=goal,
        task_detail=task_detail,
    )

    # Run reviewer query with output
    result_text = ""
    tool_count = 0

    async for message in query(
        prompt=prompt,
        options=ClaudeCodeOptions(
            system_prompt=REVIEWER_SYSTEM_PROMPT,
            allowed_tools=["Read", "Bash", "Glob", "Grep", "LSP", "WebSearch"],
            max_turns=10,
            cwd=cwd,
        ),
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                # Handle text
                if hasattr(block, "text") and block.text:
                    result_text += block.text
                    if verbose:
                        text = block.text.strip()
                        if text and len(text) > 20:
                            lines = text.split('\n')
                            first_line = lines[0][:80]
                            if len(lines[0]) > 80:
                                first_line += "..."
                            console.print(f"     [italic bright_black]📋 {first_line}[/italic bright_black]")

                # Handle tool use
                if hasattr(block, "name") and hasattr(block, "input"):
                    tool_count += 1
                    if verbose:
                        tool_line = format_tool_line(block.name, block.input, cwd)
                        console.print(f"[bright_black][{tool_count:2d}][/bright_black] {tool_line}")

    # Parse and return result
    return parse_reviewer_output(result_text)
