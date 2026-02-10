"""
MCP tools that bridge pool.py functions to Claude agents.

Problem: Agents are Claude instances with only built-in tools (Read, Write, Edit, etc.).
They cannot call Python functions like add_verified_info() or append_to_findings().
This module creates in-process MCP tools that agents CAN call, bridging the gap.

Tier 1 (HIGH value — closing real design gaps):
- ralph_check_verified: Check if a topic has been verified before searching
- ralph_add_verified: Record a verified finding after searching
- ralph_list_verified: List all verified topics
- ralph_append_finding: Atomically append a finding to pool.md
- ralph_create_task: Atomically create task file + update pool.md
- ralph_get_pivot_signals: Get unprocessed pivot recommendations
- ralph_mark_pivot_processed: Mark a pivot recommendation as processed

Tier 2 (MEDIUM value — improving observability):
- ralph_log_progress: Worker self-reports progress
"""

from claude_agent_sdk import ToolAnnotations, create_sdk_mcp_server, tool

# The cwd is set per-server instance via create_ralph_mcp_server(cwd)
_CWD = "."


def _set_cwd(cwd: str) -> None:
    global _CWD
    _CWD = cwd


# -----------------------------------------------------------------------------
# Tier 1: Verified Info Cache (3 tools)
# -----------------------------------------------------------------------------

@tool(
    name="ralph_check_verified",
    description=(
        "Check if a topic has already been verified by a previous search. "
        "MUST call this before WebSearch to avoid duplicate searches. "
        "Returns the cached finding if the topic was already verified, "
        "or 'not_found' if you need to search."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "The topic to check (e.g., 'transformers library API', 'Python 3.12 features')",
            },
        },
        "required": ["topic"],
    },
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False),
)
async def ralph_check_verified(args: dict) -> dict:
    from .pool import get_verified_info
    topic = args["topic"]
    result = get_verified_info(topic, cwd=_CWD)
    if result:
        return {
            "content": [{"type": "text", "text": f"ALREADY VERIFIED: {result}\n\nDo NOT search again. Use this cached result."}]
        }
    return {
        "content": [{"type": "text", "text": "NOT FOUND: This topic has not been verified. You should WebSearch to verify it, then call ralph_add_verified to cache the result."}]
    }


@tool(
    name="ralph_add_verified",
    description=(
        "Record a verified finding after WebSearch. "
        "This caches the result so other workers don't repeat the same search. "
        "Call this after every WebSearch that produces a useful finding."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "The topic that was verified (e.g., 'transformers library API')",
            },
            "finding": {
                "type": "string",
                "description": "What was found (e.g., 'Use AutoModelForCausalLM.from_pretrained() in v4.38+')",
            },
            "source_url": {
                "type": "string",
                "description": "URL of the source that confirmed this finding",
            },
        },
        "required": ["topic", "finding", "source_url"],
    },
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False),
)
async def ralph_add_verified(args: dict) -> dict:
    from .pool import add_verified_info
    add_verified_info(
        topic=args["topic"],
        finding=args["finding"],
        source_url=args["source_url"],
        cwd=_CWD,
    )
    return {
        "content": [{"type": "text", "text": f"Verified info cached: [{args['topic']}] Other workers will reuse this."}]
    }


@tool(
    name="ralph_list_verified",
    description="List all topics that have been verified. Use to see what's already cached before starting work.",
    input_schema={
        "type": "object",
        "properties": {},
    },
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False),
)
async def ralph_list_verified(args: dict) -> dict:
    from .pool import list_verified_topics
    topics = list_verified_topics(cwd=_CWD)
    if not topics:
        return {"content": [{"type": "text", "text": "No topics verified yet."}]}
    topic_list = "\n".join(f"- {t}" for t in topics)
    return {"content": [{"type": "text", "text": f"Verified topics ({len(topics)}):\n{topic_list}"}]}


# -----------------------------------------------------------------------------
# Tier 1: Findings Append (1 tool)
# -----------------------------------------------------------------------------

@tool(
    name="ralph_append_finding",
    description=(
        "Atomically append a finding to pool.md Findings section. "
        "Uses file locking — safe for concurrent access. "
        "Prefer this over manually editing pool.md Findings section."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "finding": {
                "type": "string",
                "description": "The finding to record (e.g., 'Redis v7 supports ACL2 with fine-grained permissions')",
            },
        },
        "required": ["finding"],
    },
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False),
)
async def ralph_append_finding(args: dict) -> dict:
    from .pool import append_to_findings
    append_to_findings(args["finding"], cwd=_CWD)
    return {"content": [{"type": "text", "text": "Finding appended to pool.md (with file lock)."}]}


# -----------------------------------------------------------------------------
# Tier 1: Atomic Task Creation (1 tool)
# -----------------------------------------------------------------------------

@tool(
    name="ralph_create_task",
    description=(
        "Atomically create a new task: creates the task file AND updates pool.md. "
        "This guarantees both operations happen together — no partial updates. "
        "Use this instead of manually writing task files and editing pool.md separately."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "Task ID (e.g., 'T003')",
            },
            "task_type": {
                "type": "string",
                "enum": ["EXPLORE", "IMPLEMENT"],
                "description": "Type of task",
            },
            "title": {
                "type": "string",
                "description": "Short task title",
            },
            "description": {
                "type": "string",
                "description": "Detailed task description",
            },
        },
        "required": ["task_id", "task_type", "title", "description"],
    },
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False),
)
async def ralph_create_task(args: dict) -> dict:
    import re

    from .pool import _read_pool_unlocked, _atomic_write, file_lock, init_task, POOL_LOCK_FILE, POOL_FILE
    from pathlib import Path

    task_id = args["task_id"]
    task_type = args["task_type"]
    title = args["title"]
    description = args["description"]

    base = Path(_CWD)
    lock_path = base / POOL_LOCK_FILE

    # Atomic: create task file + update pool.md under the same lock
    with file_lock(lock_path):
        # 1. Create task file
        init_task(task_id, task_type, title, description, cwd=_CWD)

        # 2. Update pool.md — add to Active Tasks table
        pool_content = _read_pool_unlocked(_CWD)
        if pool_content:
            status = "pending"
            new_row = f"| {task_id} | {task_type} | {status} | {title} |"

            # Find the Active Tasks table and append
            if "## Active Tasks" in pool_content:
                # Find the last table row or the header separator
                lines = pool_content.split("\n")
                insert_idx = None
                in_active = False
                for idx, line in enumerate(lines):
                    if "## Active Tasks" in line:
                        in_active = True
                    elif in_active and line.startswith("## "):
                        # Next section — insert before it
                        insert_idx = idx
                        break
                    elif in_active and line.startswith("|") and not line.startswith("|--") and not line.startswith("| ID"):
                        insert_idx = idx + 1  # After last table row

                if insert_idx is not None:
                    lines.insert(insert_idx, new_row)
                    pool_content = "\n".join(lines)
                    _atomic_write(base / POOL_FILE, pool_content)

    return {
        "content": [{"type": "text", "text": f"Task {task_id} created atomically (task file + pool.md updated)."}]
    }


# -----------------------------------------------------------------------------
# Tier 1: Pivot Signals (2 tools)
# -----------------------------------------------------------------------------

@tool(
    name="ralph_get_pivot_signals",
    description=(
        "Get all unprocessed [PIVOT_RECOMMENDED] signals from pool.md. "
        "Call this at the start of planning to check if the Evaluator "
        "has recommended any direction changes."
    ),
    input_schema={
        "type": "object",
        "properties": {},
    },
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False),
)
async def ralph_get_pivot_signals(args: dict) -> dict:
    import re
    from .pool import read_pool

    pool_content = read_pool(cwd=_CWD)
    if not pool_content or "[PIVOT_RECOMMENDED]" not in pool_content:
        return {"content": [{"type": "text", "text": "No pending pivot signals."}]}

    # Extract all PIVOT_RECOMMENDED lines
    signals = re.findall(
        r'.*\[PIVOT_RECOMMENDED\].*',
        pool_content,
    )

    if not signals:
        return {"content": [{"type": "text", "text": "No pending pivot signals."}]}

    signal_text = "\n".join(f"- {s.strip()}" for s in signals)
    return {
        "content": [{
            "type": "text",
            "text": f"PENDING PIVOT SIGNALS ({len(signals)}):\n{signal_text}\n\n"
                    f"You MUST respond with HEDGE, PIVOT_RESEARCH, or PIVOT_ITERATION for each. "
                    f"After handling, call ralph_mark_pivot_processed to clear the signal."
        }]
    }


@tool(
    name="ralph_mark_pivot_processed",
    description=(
        "Mark a [PIVOT_RECOMMENDED] signal as processed. "
        "Call this after handling a pivot signal with HEDGE/PIVOT_RESEARCH/PIVOT_ITERATION."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The task ID whose pivot signal to mark as processed (e.g., 'T001')",
            },
        },
        "required": ["task_id"],
    },
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False),
)
async def ralph_mark_pivot_processed(args: dict) -> dict:
    from .pool import clear_pivot_recommendation
    clear_pivot_recommendation(args["task_id"], cwd=_CWD)
    return {"content": [{"type": "text", "text": f"Pivot signal for {args['task_id']} marked as [PIVOT_PROCESSED]."}]}


# -----------------------------------------------------------------------------
# Tier 2: Progress Self-Report (1 tool)
# -----------------------------------------------------------------------------

@tool(
    name="ralph_log_progress",
    description=(
        "Log a progress entry to pool.md Progress Log. "
        "Use this to report significant milestones during task execution."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "entry": {
                "type": "string",
                "description": "Progress entry to log (e.g., 'Completed API integration, moving to tests')",
            },
        },
        "required": ["entry"],
    },
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False),
)
async def ralph_log_progress(args: dict) -> dict:
    from .pool import append_to_progress_log
    append_to_progress_log(args["entry"], cwd=_CWD)
    return {"content": [{"type": "text", "text": "Progress logged."}]}


# -----------------------------------------------------------------------------
# Server Factory
# -----------------------------------------------------------------------------

# All tools organized by role
WORKER_TOOLS = [
    ralph_check_verified,
    ralph_add_verified,
    ralph_list_verified,
    ralph_append_finding,
    ralph_log_progress,
]

PLANNER_TOOLS = [
    ralph_create_task,
    ralph_get_pivot_signals,
    ralph_mark_pivot_processed,
    ralph_append_finding,
    ralph_list_verified,
]

ALL_TOOLS = [
    ralph_check_verified,
    ralph_add_verified,
    ralph_list_verified,
    ralph_append_finding,
    ralph_create_task,
    ralph_get_pivot_signals,
    ralph_mark_pivot_processed,
    ralph_log_progress,
]

# Tool name lists for allowed_tools
WORKER_TOOL_NAMES = [t.name for t in WORKER_TOOLS]
PLANNER_TOOL_NAMES = [t.name for t in PLANNER_TOOLS]
ALL_TOOL_NAMES = [t.name for t in ALL_TOOLS]


def create_ralph_mcp_server(cwd: str = ".", role: str = "all"):
    """
    Create an MCP server config for Ralph tools.

    Args:
        cwd: Working directory (where .ralph/ lives)
        role: Which tools to include:
            - "worker": verified info + findings + progress
            - "planner": task creation + pivot signals + findings
            - "all": all tools

    Returns:
        McpSdkServerConfig dict for use in ClaudeAgentOptions.mcp_servers
    """
    _set_cwd(cwd)

    if role == "worker":
        tools = WORKER_TOOLS
    elif role == "planner":
        tools = PLANNER_TOOLS
    else:
        tools = ALL_TOOLS

    return create_sdk_mcp_server(
        name="ralph",
        version="1.0.0",
        tools=tools,
    )


def get_tool_names(role: str = "all") -> list[str]:
    """Get tool names for a given role (for allowed_tools)."""
    if role == "worker":
        return list(WORKER_TOOL_NAMES)
    elif role == "planner":
        return list(PLANNER_TOOL_NAMES)
    return list(ALL_TOOL_NAMES)
