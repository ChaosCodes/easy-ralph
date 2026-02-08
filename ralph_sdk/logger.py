"""
Experiment logging for Ralph SDK.

Provides structured logging for:
- Session events (iterations, decisions, executions)
- Metrics (tool calls, tokens, timing)
- History archiving
"""

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ResultMessage, query
from rich.console import Console

RALPH_DIR = ".ralph"
LOGS_DIR = f"{RALPH_DIR}/logs"
HISTORY_DIR = f"{RALPH_DIR}/history"
TOOL_CALLS_FILE = f"{LOGS_DIR}/tool_calls.jsonl"


def _summarize_tool_input(tool_input: dict) -> dict:
    """Create a loggable summary of tool input, truncating large fields."""
    summary = {}
    for k, v in tool_input.items():
        if isinstance(v, str) and len(v) > 500:
            summary[k] = v[:500] + f"...({len(v)} chars)"
        else:
            summary[k] = v
    return summary


def log_tool_call(cwd: str, agent: str, tool_name: str, tool_input: dict) -> None:
    """Append a tool call record to the tool_calls.jsonl log.

    Standalone function — no SessionLogger instance needed.
    """
    log_path = Path(cwd) / TOOL_CALLS_FILE
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now().isoformat(),
        "agent": agent,
        "tool": tool_name,
        "input": _summarize_tool_input(tool_input),
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def format_duration(seconds: float) -> str:
    """Format duration as human-readable string. e.g., 45s / 2m 15s / 1h 5m"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {secs}s" if secs else f"{minutes}m"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m" if mins else f"{hours}h"


def format_tokens(usage: dict | None) -> str:
    """Format token usage. e.g., 18.5k tokens"""
    if not usage:
        return ""
    total = (usage.get("input_tokens", 0) or 0) + (usage.get("output_tokens", 0) or 0)
    if total == 0:
        return ""
    if total >= 1_000_000:
        return f"{total / 1_000_000:.1f}M tokens"
    if total >= 1000:
        return f"{total / 1000:.1f}k tokens"
    return f"{total} tokens"


def format_stats_line(
    elapsed_seconds: float = 0,
    usage: dict | None = None,
    cost_usd: float | None = None,
    turns: int = 0,
    tool_calls: int = 0,
) -> str:
    """Format a compact stats line. e.g., (2m 15s · 18.5k tokens · 12 turns · $0.12)"""
    parts = []
    if elapsed_seconds > 0:
        parts.append(format_duration(elapsed_seconds))
    tokens_str = format_tokens(usage)
    if tokens_str:
        parts.append(tokens_str)
    if turns > 0:
        parts.append(f"{turns} turns")
    if tool_calls > 0:
        parts.append(f"{tool_calls} tool calls")
    if cost_usd is not None and cost_usd > 0:
        parts.append(f"${cost_usd:.2f}")
    if not parts:
        return ""
    return f"({' · '.join(parts)})"


# --- Output formatting (moved from worker.py) ---

def shorten_path(path: str, cwd: str = "") -> str:
    """Shorten file path for display."""
    if cwd and path.startswith(cwd):
        path = path[len(cwd):].lstrip("/")
    home = os.path.expanduser("~")
    if path.startswith(home):
        path = "~" + path[len(home):]
    return path


def format_tool_line(tool_name: str, tool_input: dict, cwd: str = "") -> str | None:
    """Format a tool use as a single compact line with rich highlighting.

    Returns None for interactive tools (AskUserQuestion) that have their own UI.
    """
    if tool_name == "AskUserQuestion":
        return None

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


# --- Unified streaming query ---

_console = Console()


@dataclass
class StreamResult:
    """Result of a streaming query to Claude."""
    text: str
    tool_count: int
    result_stats: object | None  # ResultMessage


async def stream_query(
    prompt: str,
    options: ClaudeAgentOptions,
    *,
    agent_name: str,
    emoji: str = "💭",
    cwd: str = ".",
    verbose: bool = False,
    show_tools: bool | None = None,
    status_message: str | None = None,
) -> StreamResult:
    """Run a streaming query and handle output uniformly.

    Args:
        prompt: The prompt to send.
        options: ClaudeAgentOptions for the query.
        agent_name: Agent name for log_tool_call (e.g. "worker").
        emoji: Emoji prefix for thinking lines.
        cwd: Working directory.
        verbose: If True, print first line of thinking text.
        show_tools: If True, always show tool lines. If None, follows verbose.
        status_message: If provided, printed as dim text before starting.

    Returns:
        StreamResult with accumulated text, tool count, and result stats.
    """
    if status_message:
        _console.print(f"[dim]{status_message}[/dim]")

    _should_show = show_tools if show_tools is not None else True

    result_text = ""
    tool_count = 0
    result_stats = None

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text") and block.text:
                    result_text += block.text
                    if verbose:
                        text = block.text.strip()
                        if text and len(text) > 20:
                            first_line = text.split('\n')[0]
                            _console.print(f"     [italic bright_black]{emoji} {first_line}[/italic bright_black]")

                if hasattr(block, "name") and hasattr(block, "input"):
                    tool_count += 1
                    log_tool_call(cwd, agent_name, block.name, block.input)
                    if _should_show:
                        tool_line = format_tool_line(block.name, block.input, cwd)
                        if tool_line:
                            _console.print(f"[bright_black][{tool_count:2d}][/bright_black] {tool_line}")
        elif isinstance(message, ResultMessage):
            result_stats = message

    return StreamResult(text=result_text, tool_count=tool_count, result_stats=result_stats)


@dataclass
class SessionMetrics:
    """Aggregated metrics for a session."""
    session_id: str
    started_at: str
    ended_at: Optional[str] = None
    total_iterations: int = 0
    total_tool_calls: int = 0
    tasks_created: int = 0
    tasks_completed: int = 0
    actions: dict = field(default_factory=dict)  # action -> count
    duration_seconds: float = 0.0
    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    goal: str = ""
    status: str = "running"  # running, completed, interrupted


class SessionLogger:
    """Logger for a single Ralph session."""

    def __init__(self, cwd: str = "."):
        self.cwd = Path(cwd)
        self.logs_dir = self.cwd / LOGS_DIR
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Generate session ID
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.logs_dir / f"session_{self.session_id}.jsonl"
        self.metrics_file = self.logs_dir / "metrics.json"

        # Initialize metrics
        self.metrics = SessionMetrics(
            session_id=self.session_id,
            started_at=datetime.now().isoformat(),
        )

        # Track start time for duration
        self._start_time = datetime.now()

    def _write_event(self, event: dict) -> None:
        """Append an event to the session log."""
        event["ts"] = datetime.now().isoformat()
        with open(self.log_file, "a") as f:
            f.write(json.dumps(event) + "\n")

    def _save_metrics(self) -> None:
        """Save current metrics to file."""
        self.metrics.duration_seconds = (datetime.now() - self._start_time).total_seconds()
        with open(self.metrics_file, "w") as f:
            json.dump(asdict(self.metrics), f, indent=2)

    def log_query_stats(self, result_message) -> None:
        """Accumulate stats from a ResultMessage."""
        if result_message is None:
            return
        if result_message.usage:
            self.metrics.total_input_tokens += result_message.usage.get("input_tokens", 0) or 0
            self.metrics.total_output_tokens += result_message.usage.get("output_tokens", 0) or 0
        if result_message.total_cost_usd:
            self.metrics.total_cost_usd += result_message.total_cost_usd
        self._save_metrics()

    def log_session_start(self, goal: str) -> None:
        """Log session start."""
        self.metrics.goal = goal[:200]  # Truncate for storage
        self._write_event({
            "event": "session_start",
            "goal": goal,
            "session_id": self.session_id,
        })
        self._save_metrics()

    def log_iteration_start(self, iteration: int, max_iterations: int) -> None:
        """Log iteration start."""
        self.metrics.total_iterations = iteration
        self._write_event({
            "event": "iteration_start",
            "iteration": iteration,
            "max_iterations": max_iterations,
        })

    def log_iteration_end(self, iteration: int, action: str, success: bool = True) -> None:
        """Log iteration end."""
        self._write_event({
            "event": "iteration_end",
            "iteration": iteration,
            "action": action,
            "success": success,
        })

    def log_planner_decision(
        self,
        action: str,
        target: Optional[str],
        reason: str,
        tool_calls: int = 0,
    ) -> None:
        """Log planner decision."""
        self.metrics.total_tool_calls += tool_calls
        self.metrics.actions[action] = self.metrics.actions.get(action, 0) + 1

        self._write_event({
            "event": "planner_decision",
            "action": action,
            "target": target,
            "reason": reason[:200],  # Truncate
            "tool_calls": tool_calls,
        })
        self._save_metrics()

    def log_worker_complete(
        self,
        task_id: str,
        task_type: str,
        success: bool,
        tool_calls: int = 0,
        error: Optional[str] = None,
    ) -> None:
        """Log worker completion."""
        self.metrics.total_tool_calls += tool_calls

        self._write_event({
            "event": "worker_complete",
            "task_id": task_id,
            "task_type": task_type,
            "success": success,
            "tool_calls": tool_calls,
            "error": error,
        })
        self._save_metrics()

    def log_reviewer_verdict(
        self,
        task_id: str,
        verdict: str,
        reason: str,
        tool_calls: int = 0,
    ) -> None:
        """Log reviewer verdict."""
        self.metrics.total_tool_calls += tool_calls

        if verdict == "passed":
            self.metrics.tasks_completed += 1

        self._write_event({
            "event": "reviewer_verdict",
            "task_id": task_id,
            "verdict": verdict,
            "reason": reason[:200],
            "tool_calls": tool_calls,
        })
        self._save_metrics()

    def log_task_created(self, task_id: str, task_type: str, description: str) -> None:
        """Log task creation."""
        self.metrics.tasks_created += 1

        self._write_event({
            "event": "task_created",
            "task_id": task_id,
            "task_type": task_type,
            "description": description[:200],
        })
        self._save_metrics()

    def log_task_modified(self, task_id: str, modification: str) -> None:
        """Log task modification."""
        self._write_event({
            "event": "task_modified",
            "task_id": task_id,
            "modification": modification[:200],
        })

    def log_task_skipped(self, task_id: Optional[str], reason: str) -> None:
        """Log task skip."""
        self._write_event({
            "event": "task_skipped",
            "task_id": task_id,
            "reason": reason[:200],
        })

    def log_user_question(self, question: str, answer: str) -> None:
        """Log user question and answer."""
        self._write_event({
            "event": "user_question",
            "question": question[:200],
            "answer": answer[:200],
        })

    def log_evaluation(
        self,
        task_id: str,
        passed: bool,
        score: float,
        issues: list[str] | None = None,
        tool_calls: int = 0,
    ) -> None:
        """Log quality evaluation result."""
        self.metrics.total_tool_calls += tool_calls

        self._write_event({
            "event": "evaluation",
            "task_id": task_id,
            "passed": passed,
            "score": score,
            "issues": issues[:5] if issues else [],  # Limit to 5 issues
            "tool_calls": tool_calls,
        })
        self._save_metrics()

    def log_session_end(self, success: bool, reason: str = "") -> None:
        """Log session end."""
        self.metrics.ended_at = datetime.now().isoformat()
        self.metrics.status = "completed" if success else "interrupted"

        self._write_event({
            "event": "session_end",
            "success": success,
            "reason": reason,
            "total_iterations": self.metrics.total_iterations,
            "total_tool_calls": self.metrics.total_tool_calls,
            "duration_seconds": self.metrics.duration_seconds,
            "total_cost_usd": self.metrics.total_cost_usd,
            "total_input_tokens": self.metrics.total_input_tokens,
            "total_output_tokens": self.metrics.total_output_tokens,
        })
        self._save_metrics()

    def log_error(self, error: str, context: Optional[dict] = None) -> None:
        """Log an error."""
        self._write_event({
            "event": "error",
            "error": error,
            "context": context or {},
        })


def archive_session(cwd: str = ".", keep_current: bool = False) -> Optional[str]:
    """
    Archive the current session to history.

    Args:
        cwd: Working directory
        keep_current: If True, copy instead of move

    Returns:
        Archive path if successful, None otherwise
    """
    ralph_dir = Path(cwd) / RALPH_DIR
    if not ralph_dir.exists():
        return None

    # Create history directory
    history_dir = Path(cwd) / HISTORY_DIR
    history_dir.mkdir(parents=True, exist_ok=True)

    # Generate archive name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"session_{timestamp}"
    archive_path = history_dir / archive_name

    # Copy or move
    if keep_current:
        shutil.copytree(ralph_dir, archive_path, ignore=shutil.ignore_patterns("history"))
    else:
        # Move everything except history
        archive_path.mkdir()
        for item in ralph_dir.iterdir():
            if item.name != "history":
                shutil.move(str(item), str(archive_path / item.name))

    return str(archive_path)


def list_archived_sessions(cwd: str = ".") -> list[dict]:
    """List all archived sessions."""
    history_dir = Path(cwd) / HISTORY_DIR
    if not history_dir.exists():
        return []

    sessions = []
    for session_dir in sorted(history_dir.iterdir(), reverse=True):
        if session_dir.is_dir():
            metrics_file = session_dir / "logs" / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)
                sessions.append({
                    "path": str(session_dir),
                    "name": session_dir.name,
                    **metrics,
                })
            else:
                # Fallback for sessions without metrics
                goal_file = session_dir / "goal.md"
                goal = ""
                if goal_file.exists():
                    goal = goal_file.read_text()[:100]
                sessions.append({
                    "path": str(session_dir),
                    "name": session_dir.name,
                    "goal": goal,
                })

    return sessions


def get_session_summary(cwd: str = ".") -> Optional[dict]:
    """Get summary of current session."""
    metrics_file = Path(cwd) / LOGS_DIR / "metrics.json"
    if not metrics_file.exists():
        return None

    with open(metrics_file) as f:
        return json.load(f)
