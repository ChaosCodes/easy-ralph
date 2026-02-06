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

from claude_code_sdk import AssistantMessage, ClaudeCodeOptions, query
from rich.console import Console

from .pool import read_goal, read_pool
from .prompts import EXPLORE_MODE_ADDENDUM, PLANNER_SYSTEM_PROMPT, build_planner_prompt
from .worker import format_tool_line

console = Console()


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

    def __post_init__(self):
        if self.task_ids is None:
            self.task_ids = []

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


def parse_planner_output(text: str) -> PlannerDecision:
    """Parse planner output into a PlannerDecision."""
    # Extract ACTION
    action_match = re.search(r"ACTION:\s*(\w+)", text, re.IGNORECASE)
    # Default to SKIP if parsing fails - safer than DONE which would terminate
    action_str = action_match.group(1).lower() if action_match else "skip"

    try:
        action = Action(action_str)
    except ValueError:
        action = Action.SKIP  # Default to skip if unknown action

    # Extract TARGET (single task ID)
    target_match = re.search(r"TARGET:\s*(T\d+)", text, re.IGNORECASE)
    target = target_match.group(1) if target_match else None

    # Extract TASK_IDS (multiple task IDs for PARALLEL_EXECUTE)
    # Format: TASK_IDS: T001, T002, T003 or TASK_IDS: T001 T002 T003
    task_ids = []
    task_ids_match = re.search(r"TASK_IDS:\s*(.+?)(?=\n|$)", text, re.IGNORECASE)
    if task_ids_match:
        task_ids_str = task_ids_match.group(1)
        # Extract all task IDs from the line
        task_ids = re.findall(r'T\d+', task_ids_str)

    # Extract REASON (up to next field or end)
    reason_match = re.search(
        r"REASON:\s*(.+?)(?=\n(?:NEW_TASKS:|QUESTION:|MODIFICATION:|TASK_IDS:|CURRENT_APPROACH:|ATTEMPT_COUNT:|$))",
        text, re.IGNORECASE | re.DOTALL
    )
    reason = reason_match.group(1).strip() if reason_match else ""

    # Extract NEW_TASKS
    new_tasks_match = re.search(
        r"NEW_TASKS:\s*(.+?)(?=\n(?:QUESTION:|MODIFICATION:|$)|\Z)",
        text, re.IGNORECASE | re.DOTALL
    )
    new_tasks = new_tasks_match.group(1).strip() if new_tasks_match else ""

    # Extract QUESTION (for ASK action)
    question_match = re.search(
        r"QUESTION:\s*(.+?)(?=\n(?:MODIFICATION:|$)|\Z)",
        text, re.IGNORECASE | re.DOTALL
    )
    question = question_match.group(1).strip() if question_match else ""

    # Extract MODIFICATION (for MODIFY action)
    modification_match = re.search(
        r"MODIFICATION:\s*(.+?)(?=\n(?:FAILURE_ASSUMPTIONS:|$)|\Z)",
        text, re.IGNORECASE | re.DOTALL
    )
    modification = modification_match.group(1).strip() if modification_match else ""

    # Extract FAILURE_ASSUMPTIONS (for HEDGE/PIVOT_WAIT action)
    failure_match = re.search(
        r"FAILURE_ASSUMPTIONS:\s*(.+?)(?=\n(?:NEW_TASKS:|$)|\Z)",
        text, re.IGNORECASE | re.DOTALL
    )
    failure_assumptions = failure_match.group(1).strip() if failure_match else ""

    # Extract PIVOT_RESEARCH fields
    current_approach_match = re.search(
        r"CURRENT_APPROACH:\s*(.+?)(?=\n(?:BLOCKER:|$)|\Z)",
        text, re.IGNORECASE | re.DOTALL
    )
    current_approach = current_approach_match.group(1).strip() if current_approach_match else ""

    blocker_match = re.search(
        r"BLOCKER:\s*(.+?)(?=\n(?:NEW_DIRECTION:|$)|\Z)",
        text, re.IGNORECASE | re.DOTALL
    )
    blocker = blocker_match.group(1).strip() if blocker_match else ""

    new_direction_match = re.search(
        r"NEW_DIRECTION:\s*(.+?)(?=\n(?:REASON:|NEW_TASKS:|$)|\Z)",
        text, re.IGNORECASE | re.DOTALL
    )
    new_direction = new_direction_match.group(1).strip() if new_direction_match else ""

    # Extract PIVOT_ITERATION fields
    attempt_count_match = re.search(r"ATTEMPT_COUNT:\s*(\d+)", text, re.IGNORECASE)
    attempt_count = int(attempt_count_match.group(1)) if attempt_count_match else 0

    best_score_match = re.search(
        r"BEST_SCORE:\s*(.+?)(?=\n|$)",
        text, re.IGNORECASE
    )
    best_score = best_score_match.group(1).strip() if best_score_match else ""

    failure_pattern_match = re.search(
        r"FAILURE_PATTERN:\s*(.+?)(?=\n(?:NEW_APPROACH:|$)|\Z)",
        text, re.IGNORECASE | re.DOTALL
    )
    failure_pattern = failure_pattern_match.group(1).strip() if failure_pattern_match else ""

    new_approach_match = re.search(
        r"NEW_APPROACH:\s*(.+?)(?=\n(?:REASON:|NEW_TASKS:|$)|\Z)",
        text, re.IGNORECASE | re.DOTALL
    )
    new_approach = new_approach_match.group(1).strip() if new_approach_match else ""

    # For HEDGE/PIVOT_WAIT action, target is the task being hedged
    hedge_for = target if action in (Action.HEDGE, Action.PIVOT_WAIT) else None

    return PlannerDecision(
        action=action,
        target=target,
        task_ids=task_ids,
        reason=reason,
        new_tasks=new_tasks,
        question=question,
        modification=modification,
        hedge_for=hedge_for,
        failure_assumptions=failure_assumptions,
        # PIVOT_RESEARCH fields
        current_approach=current_approach,
        blocker=blocker,
        new_direction=new_direction,
        # PIVOT_ITERATION fields
        attempt_count=attempt_count,
        best_score=best_score,
        failure_pattern=failure_pattern,
        new_approach=new_approach,
    )


async def plan(cwd: str = ".", verbose: bool = False, explore_mode: bool = False) -> PlannerDecision:
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

    # Build prompt
    prompt = build_planner_prompt(goal, pool)

    # Add explore mode context to prompt if enabled
    if explore_mode:
        prompt = f"**[探索模式 ACTIVE]**\n\n{prompt}\n\n---\n{EXPLORE_MODE_ADDENDUM}"

    # Build system prompt with explore mode addendum if needed
    system_prompt = PLANNER_SYSTEM_PROMPT
    if explore_mode:
        system_prompt = f"{PLANNER_SYSTEM_PROMPT}\n\n{EXPLORE_MODE_ADDENDUM}"

    # Run planner query with output
    result_text = ""
    tool_count = 0

    async for message in query(
        prompt=prompt,
        options=ClaudeCodeOptions(
            system_prompt=system_prompt,
            allowed_tools=[
                "Read", "Glob", "Grep", "Write", "Edit",  # File operations
                "LSP",  # Code intelligence
                "WebFetch", "WebSearch",  # Research best practices, docs
            ],
            permission_mode="acceptEdits",
            max_turns=15,
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
                            console.print(f"     [italic bright_black]🧠 {first_line}[/italic bright_black]")

                # Handle tool use
                if hasattr(block, "name") and hasattr(block, "input"):
                    tool_count += 1
                    if verbose:
                        tool_line = format_tool_line(block.name, block.input, cwd)
                        console.print(f"[bright_black][{tool_count:2d}][/bright_black] {tool_line}")

    # Parse and return decision
    return parse_planner_output(result_text)
