"""
Planner agent: decides the next action based on current state.

Reads goal.md and pool.md, outputs one of these actions:
- EXECUTE, EXPLORE: Execute tasks
- CREATE, MODIFY, DELETE: Manage tasks
- SKIP: Skip blocked task temporarily
- ASK: Ask user for decision
- DONE: Goal completed
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from claude_agent_sdk import ClaudeAgentOptions
from rich.console import Console

from .logger import log_tool_call, stream_query
from .mcp_tools import create_ralph_mcp_server, get_tool_names
from .pool import clear_handoff_note, read_goal, read_handoff_note, read_pool
from .prompts import EXPLORE_MODE_ADDENDUM, PLANNER_SYSTEM_PROMPT, build_planner_prompt
from .utils import extract_json

console = Console()

PLANNER_OUTPUT_SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string"},
            "target": {"type": ["string", "null"]},
            "task_ids": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string"},
            "new_tasks": {"type": "string"},
            "question": {"type": "string"},
            "modification": {"type": "string"},
            "failure_assumptions": {"type": "string"},
            "current_approach": {"type": "string"},
            "blocker": {"type": "string"},
            "new_direction": {"type": "string"},
            "attempt_count": {"type": "integer"},
            "best_score": {"type": "string"},
            "failure_pattern": {"type": "string"},
            "new_approach": {"type": "string"},
            "fork_approaches": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["action", "reason"],
    },
}


class Action(Enum):
    # Task execution
    EXECUTE = "execute"
    EXPLORE = "explore"
    PARALLEL_EXECUTE = "parallel_execute"  # Execute multiple independent tasks concurrently

    # Task management
    CREATE = "create"
    MODIFY = "modify"      # Modify existing task description/scope
    DELETE = "delete"

    # Control flow
    SKIP = "skip"          # Skip blocked task, continue with others
    ASK = "ask"            # Ask user for decision
    HEDGE = "hedge"        # Explore alternatives (pessimistic preparation)
    DONE = "done"

    # Pivot actions (agent-initiated direction changes)
    PIVOT_RESEARCH = "pivot_research"    # Abandon after research confirms not viable
    PIVOT_WAIT = "pivot_wait"            # Explore alternatives while waiting (alias for HEDGE)
    PIVOT_ITERATION = "pivot_iteration"  # Change direction after multiple failed attempts

    # Session forking
    FORK = "fork"                        # Try multiple approaches via session forking

    # Legacy (kept for compatibility, may be removed)
    DECOMPOSE = "decompose"


@dataclass
class PlannerDecision:
    action: Action
    target: Optional[str] = None  # task_id for most actions
    task_ids: list[str] = None    # for PARALLEL_EXECUTE: list of task IDs
    reason: str = ""
    new_tasks: str = ""           # for CREATE/DECOMPOSE/HEDGE/PIVOT
    question: str = ""            # for ASK
    modification: str = ""        # for MODIFY
    hedge_for: Optional[str] = None        # for HEDGE: which task to hedge
    failure_assumptions: str = ""          # for HEDGE/PIVOT_WAIT: failure assumptions

    # PIVOT_RESEARCH fields
    current_approach: str = ""    # What approach is being abandoned
    blocker: str = ""             # Why it's not viable
    new_direction: str = ""       # What new direction to try

    # PIVOT_ITERATION fields
    attempt_count: int = 0        # How many attempts were made
    best_score: str = ""          # Highest score achieved
    failure_pattern: str = ""     # Pattern of failure observed
    new_approach: str = ""        # New approach to try

    # FORK fields
    fork_approaches: list[str] = None  # List of approach descriptions for FORK

    # Execution stats (set after query completes)
    result_stats: Optional[object] = None  # ResultMessage from SDK

    def __post_init__(self):
        if self.task_ids is None:
            self.task_ids = []
        if self.fork_approaches is None:
            self.fork_approaches = []

    @property
    def is_pivot(self) -> bool:
        """Check if this is a pivot action."""
        return self.action in (
            Action.PIVOT_RESEARCH,
            Action.PIVOT_WAIT,
            Action.PIVOT_ITERATION,
            Action.HEDGE,
        )

    @property
    def pivot_type(self) -> Optional[str]:
        """Get the type of pivot if this is a pivot action."""
        if self.action == Action.PIVOT_RESEARCH:
            return "research"
        elif self.action in (Action.PIVOT_WAIT, Action.HEDGE):
            return "wait"
        elif self.action == Action.PIVOT_ITERATION:
            return "iteration"
        return None


def _parse_planner_json(obj: dict) -> PlannerDecision:
    """Parse a JSON object into a PlannerDecision."""
    action_str = obj.get("action", "skip").lower()
    try:
        action = Action(action_str)
    except ValueError:
        action = Action.SKIP

    target = obj.get("target")
    task_ids = obj.get("task_ids", [])
    if isinstance(task_ids, str):
        task_ids = re.findall(r'T\d+', task_ids)

    hedge_for = target if action in (Action.HEDGE, Action.PIVOT_WAIT) else None

    fork_approaches = obj.get("fork_approaches", [])
    if isinstance(fork_approaches, str):
        fork_approaches = [fork_approaches]

    return PlannerDecision(
        action=action,
        target=target,
        task_ids=task_ids,
        reason=obj.get("reason", ""),
        new_tasks=obj.get("new_tasks", ""),
        question=obj.get("question", ""),
        modification=obj.get("modification", ""),
        hedge_for=hedge_for,
        failure_assumptions=obj.get("failure_assumptions", ""),
        current_approach=obj.get("current_approach", ""),
        blocker=obj.get("blocker", ""),
        new_direction=obj.get("new_direction", ""),
        attempt_count=int(obj.get("attempt_count", 0)),
        best_score=str(obj.get("best_score", "")),
        failure_pattern=obj.get("failure_pattern", ""),
        new_approach=obj.get("new_approach", ""),
        fork_approaches=fork_approaches,
    )


def parse_planner_output(structured_output: dict | None, text: str) -> PlannerDecision:
    """Parse planner output. Prefers structured_output, falls back to JSON extraction."""
    obj = structured_output
    if not obj:
        obj = extract_json(text)
    if obj and "action" in obj:
        return _parse_planner_json(obj)
    return PlannerDecision(action=Action.SKIP, reason="Failed to parse planner output")


async def plan(cwd: str = ".", verbose: bool = False, explore_mode: bool = False, use_mcp: bool = False, thinking_budget: int | None = None) -> PlannerDecision:
    """
    Run the planner to decide the next action.

    Args:
        cwd: Working directory containing .ralph/
        verbose: Show detailed output
        explore_mode: If True, apply explore mode rules (no DONE without user confirmation)

    Returns:
        PlannerDecision with the chosen action
    """
    # Read current state
    goal = read_goal(cwd)
    pool = read_pool(cwd)

    if not goal:
        raise ValueError("goal.md not found. Run clarifier first.")
    if not pool:
        raise ValueError("pool.md not found. Run initializer first.")

    # Read handoff note if available (from previous session resume)
    handoff = read_handoff_note(cwd)

    # Build prompt
    prompt = build_planner_prompt(goal, pool, handoff=handoff)

    # Clear handoff after first use (it's now in the prompt)
    if handoff:
        clear_handoff_note(cwd)

    # Add explore mode context to prompt if enabled
    if explore_mode:
        prompt = f"**[探索模式 ACTIVE]**\n\n{prompt}\n\n---\n{EXPLORE_MODE_ADDENDUM}"

    # Build system prompt with explore mode addendum if needed
    system_prompt = PLANNER_SYSTEM_PROMPT
    if explore_mode:
        system_prompt = f"{PLANNER_SYSTEM_PROMPT}\n\n{EXPLORE_MODE_ADDENDUM}"

    # Create MCP server for planner tools (skip if use_mcp=False)
    base_tools = [
        "Read", "Glob", "Grep", "Write", "Edit",
        "LSP",
        "WebFetch", "WebSearch",
    ]
    if use_mcp:
        ralph_mcp = create_ralph_mcp_server(cwd, role="planner")
        ralph_tool_names = get_tool_names("planner")
        allowed_tools = base_tools + ralph_tool_names
        mcp_servers = {"ralph": ralph_mcp}
    else:
        allowed_tools = base_tools
        mcp_servers = None

    # Configure thinking for planner (default: 10000 — decision-making benefits most)
    planner_options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        allowed_tools=allowed_tools,
        permission_mode="acceptEdits",
        max_turns=15,
        cwd=cwd,
        output_format=PLANNER_OUTPUT_SCHEMA,
        mcp_servers=mcp_servers,
    )
    effective = thinking_budget if thinking_budget is not None else 10_000
    if effective > 0:
        planner_options.max_thinking_tokens = effective

    # Run planner query with unified streaming
    sr = await stream_query(
        prompt=prompt,
        options=planner_options,
        agent_name="planner",
        emoji="🧠",
        cwd=cwd,
        verbose=verbose,
    )

    # Parse and return decision
    decision = parse_planner_output(sr.structured_output, sr.text)
    decision.result_stats = sr.result_stats
    return decision
