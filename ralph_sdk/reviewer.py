"""
Reviewer agent: verifies IMPLEMENT task results.

Possible verdicts:
- PASSED: task complete
- RETRY: non-fundamental issue, can retry
- FAILED: fundamental issue, need different approach
"""

import json
from dataclasses import dataclass
from enum import Enum

from claude_agent_sdk import ClaudeAgentOptions, SandboxSettings
from rich.console import Console

from .logger import log_tool_call, stream_query
from .pool import read_goal, read_task
from .prompts import REVIEWER_SYSTEM_PROMPT, build_reviewer_prompt
from .utils import extract_json

console = Console()

REVIEWER_OUTPUT_SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["passed", "retry", "failed"]},
            "reason": {"type": "string"},
            "suggestions": {"type": "string"},
        },
        "required": ["verdict", "reason"],
    },
}


class Verdict(Enum):
    PASSED = "passed"
    RETRY = "retry"
    FAILED = "failed"


@dataclass
class ReviewResult:
    verdict: Verdict
    reason: str
    suggestions: str = ""
    tool_count: int = 0
    result_stats: object = None  # ResultMessage from SDK


def parse_reviewer_output(structured_output: dict | None, text: str) -> ReviewResult:
    """Parse reviewer output. Prefers structured_output, falls back to JSON extraction."""
    obj = structured_output
    if not obj:
        obj = extract_json(text)
    if obj and "verdict" in obj:
        verdict_str = obj["verdict"].lower()
        try:
            verdict = Verdict(verdict_str)
        except ValueError:
            verdict = Verdict.RETRY
        return ReviewResult(
            verdict=verdict,
            reason=obj.get("reason", ""),
            suggestions=obj.get("suggestions", ""),
        )
    return ReviewResult(verdict=Verdict.RETRY, reason="Failed to parse reviewer output")


async def review(task_id: str, cwd: str = ".", verbose: bool = False, thinking_budget: int | None = None, sandbox_allowed_domains: list[str] | None = None) -> ReviewResult:
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
    reviewer_sandbox = SandboxSettings(
        enabled=True,
        autoAllowBashIfSandboxed=True,
        allowUnsandboxedCommands=True,
        # python/pip/git run outside sandbox to avoid Seatbelt restrictions
        excludedCommands=["python3", "python", "pip", "pip3", "git"],
    )

    # WebFetch rules control the sandbox proxy's domain allowlist
    from .worker import DEFAULT_SANDBOX_ALLOWED_DOMAINS
    domains = sandbox_allowed_domains if sandbox_allowed_domains is not None else DEFAULT_SANDBOX_ALLOWED_DOMAINS
    sandbox_settings_json = None
    if domains:
        sandbox_settings_json = json.dumps({
            "permissions": {
                "allow": [f"WebFetch(domain:{d})" for d in domains]
            }
        })

    # Configure thinking for reviewer (default: 0 — mostly runs tests and checks)
    reviewer_options = ClaudeAgentOptions(
        system_prompt=REVIEWER_SYSTEM_PROMPT,
        allowed_tools=["Read", "Bash", "Glob", "Grep", "LSP", "WebSearch"],
        permission_mode="acceptEdits",
        sandbox=reviewer_sandbox,
        settings=sandbox_settings_json,
        max_turns=25,
        cwd=cwd,
        output_format=REVIEWER_OUTPUT_SCHEMA,
    )
    effective = thinking_budget if thinking_budget is not None else 0
    if effective > 0:
        reviewer_options.max_thinking_tokens = effective

    sr = await stream_query(
        prompt=prompt,
        options=reviewer_options,
        agent_name="reviewer",
        emoji="📋",
        cwd=cwd,
        verbose=verbose,
        include_partial=verbose,
    )

    # Parse and return result
    review_result = parse_reviewer_output(sr.structured_output, sr.text)
    review_result.tool_count = sr.tool_count
    review_result.result_stats = sr.result_stats

    # Layer 2: Detect max_turns exhausted without verdict, resume for follow-up
    REVIEWER_MAX_TURNS = 25
    turns_exhausted = (
        sr.result_stats
        and sr.result_stats.num_turns >= REVIEWER_MAX_TURNS
        and review_result.reason == ""
    )

    if turns_exhausted:
        console.print("[yellow]⚠ Reviewer exhausted turns without verdict, requesting follow-up...[/yellow]")

        # Resume same session to preserve full investigation context
        followup_sr = await stream_query(
            prompt="你已经用完了所有调查轮次。基于你已经看到的所有内容，立即输出 verdict JSON。",
            options=ClaudeAgentOptions(
                system_prompt=REVIEWER_SYSTEM_PROMPT,
                resume=sr.result_stats.session_id,
                allowed_tools=[],
                max_turns=3,
                cwd=cwd,
                output_format=REVIEWER_OUTPUT_SCHEMA,
            ),
            agent_name="reviewer",
            emoji="📋",
            cwd=cwd,
            verbose=verbose,
            include_partial=verbose,
        )

        if followup_sr.text.strip():
            review_result = parse_reviewer_output(followup_sr.structured_output, followup_sr.text)
            review_result.tool_count = sr.tool_count  # keep original tool count
            review_result.result_stats = sr.result_stats

        # If follow-up also failed, provide a meaningful reason
        if review_result.reason == "":
            review_result = ReviewResult(
                verdict=Verdict.RETRY,
                reason=f"Reviewer exhausted {REVIEWER_MAX_TURNS} turns without producing a verdict (follow-up also failed).",
                tool_count=sr.tool_count,
                result_stats=sr.result_stats,
            )

    return review_result
