"""
Experiment logging for Ralph SDK.

Provides structured logging for:
- Session events (iterations, decisions, executions)
- Metrics (tool calls, tokens, timing)
- History archiving
"""

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

RALPH_DIR = ".ralph"
LOGS_DIR = f"{RALPH_DIR}/logs"
HISTORY_DIR = f"{RALPH_DIR}/history"


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
