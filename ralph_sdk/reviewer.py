"""
Reviewer agent: verifies IMPLEMENT task results.

Possible verdicts:
- PASSED: task complete
- RETRY: non-fundamental issue, can retry
- FAILED: fundamental issue, need different approach
"""

import re
from dataclasses import dataclass
from enum import Enum

from claude_agent_sdk import ClaudeAgentOptions
from rich.console import Console

from .logger import log_tool_call, stream_query
from .pool import read_goal, read_task
from .prompts import REVIEWER_SYSTEM_PROMPT, build_reviewer_prompt
from .utils import extract_json

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
    result_stats: object = None  # ResultMessage from SDK


def parse_reviewer_output(text: str) -> ReviewResult:
    """Parse reviewer output into a ReviewResult.

    Tries JSON first (more reliable), falls back to regex for backwards compatibility.
    """
    # Try JSON parsing first
    json_obj = extract_json(text)
    if json_obj and "verdict" in json_obj:
        verdict_str = json_obj["verdict"].lower()
        try:
            verdict = Verdict(verdict_str)
        except ValueError:
            verdict = Verdict.RETRY
        return ReviewResult(
            verdict=verdict,
            reason=json_obj.get("reason", ""),
            suggestions=json_obj.get("suggestions", ""),
        )

    # Fallback: regex parsing
    verdict_match = re.search(r"VERDICT:\s*(\w+)", text, re.IGNORECASE)
    verdict_str = verdict_match.group(1).lower() if verdict_match else "retry"

    try:
        verdict = Verdict(verdict_str)
    except ValueError:
        verdict = Verdict.RETRY

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

    # Run reviewer query with unified streaming
    sr = await stream_query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            system_prompt=REVIEWER_SYSTEM_PROMPT,
            allowed_tools=["Read", "Bash", "Glob", "Grep", "LSP", "WebSearch"],
            permission_mode="acceptEdits",
            max_turns=10,
            cwd=cwd,
        ),
        agent_name="reviewer",
        emoji="📋",
        cwd=cwd,
        verbose=verbose,
    )

    # Parse and return result
    review_result = parse_reviewer_output(sr.text)
    review_result.result_stats = sr.result_stats
    return review_result
