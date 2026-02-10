"""
File operations for .ralph/ directory.

Structure:
    .ralph/
    ├── goal.md           # Clarifier output
    ├── pool.md           # Task index (lightweight)
    ├── feedback.md       # User feedback template (when waiting)
    ├── checkpoints/      # Model checkpoints for testing
    │   └── manifest.json # Checkpoint metadata
    └── tasks/
        ├── T001.md       # Task details
        └── ...
"""

import fcntl
import json
import os
import re
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

RALPH_DIR = ".ralph"
GOAL_FILE = f"{RALPH_DIR}/goal.md"
POOL_FILE = f"{RALPH_DIR}/pool.md"
TASKS_DIR = f"{RALPH_DIR}/tasks"
AUDITS_DIR = f"{RALPH_DIR}/audits"
FEEDBACK_FILE = f"{RALPH_DIR}/feedback.md"
CHECKPOINTS_DIR = f"{RALPH_DIR}/checkpoints"
MANIFEST_FILE = f"{CHECKPOINTS_DIR}/manifest.json"


# -----------------------------------------------------------------------------
# Checkpoint Data Structure
# -----------------------------------------------------------------------------


@dataclass
class ProxyScore:
    """A proxy metric score for a checkpoint."""
    metric_name: str
    score: float  # 0-100
    target: Optional[str] = None  # e.g., ">= 70%"
    passed: bool = False  # Whether it meets the target
    notes: str = ""


@dataclass
class Checkpoint:
    """
    A checkpoint represents a testable artifact (model, code version, etc.)

    Worker produces checkpoints when implementing tasks that need user testing.
    Evaluator adds proxy scores to checkpoints.
    User fills in final results via feedback.md.
    """
    id: str  # Unique ID, e.g., "T002_v1", "T002_v2"
    task_id: str  # The task that produced this checkpoint
    version: int = 1  # Version number for this task

    # Artifact location
    path: Optional[str] = None  # Path to the artifact (model file, code directory, etc.)
    artifact_type: str = "code"  # code, model, binary, config, etc.

    # Metadata
    created_at: str = ""  # ISO format timestamp
    description: str = ""  # What this checkpoint contains
    changes: list[str] = field(default_factory=list)  # What changed from previous version

    # Proxy scores (auto-evaluated)
    proxy_scores: list[ProxyScore] = field(default_factory=list)
    proxy_overall: float = 0.0  # Overall proxy score (0-100)

    # User testing results (filled after feedback)
    user_score: Optional[float] = None  # 1-5 scale
    user_notes: str = ""
    tested_at: Optional[str] = None

    # Status
    status: str = "pending"  # pending, tested, best, rejected

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.id:
            self.id = f"{self.task_id}_v{self.version}"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Checkpoint":
        """Create from dictionary."""
        # Handle proxy_scores specially
        proxy_scores = [
            ProxyScore(**ps) if isinstance(ps, dict) else ps
            for ps in data.pop("proxy_scores", [])
        ]
        return cls(proxy_scores=proxy_scores, **data)

    def add_proxy_score(self, metric_name: str, score: float, target: str = None, notes: str = "") -> None:
        """Add a proxy score for this checkpoint."""
        passed = False
        if target:
            # Parse target like ">= 70%" or "<= 100ms"
            match = re.match(r"([<>=]+)\s*(\d+(?:\.\d+)?)", target)
            if match:
                op, val = match.groups()
                val = float(val)
                if ">=" in op:
                    passed = score >= val
                elif "<=" in op:
                    passed = score <= val
                elif ">" in op:
                    passed = score > val
                elif "<" in op:
                    passed = score < val

        self.proxy_scores.append(ProxyScore(
            metric_name=metric_name,
            score=score,
            target=target,
            passed=passed,
            notes=notes,
        ))

        # Update overall proxy score (average of all)
        if self.proxy_scores:
            self.proxy_overall = sum(ps.score for ps in self.proxy_scores) / len(self.proxy_scores)

    def set_user_result(self, score: float, notes: str = "") -> None:
        """Set user testing result."""
        self.user_score = score
        self.user_notes = notes
        self.tested_at = datetime.now().isoformat()
        self.status = "tested"

    def mark_as_best(self) -> None:
        """Mark this checkpoint as the best version."""
        self.status = "best"

    def mark_as_rejected(self) -> None:
        """Mark this checkpoint as rejected."""
        self.status = "rejected"


# -----------------------------------------------------------------------------
# File Locking for Concurrent Access
# -----------------------------------------------------------------------------

# Global lock file paths (relative to cwd)
POOL_LOCK_FILE = f"{RALPH_DIR}/.pool.lock"
TASK_LOCK_PREFIX = f"{RALPH_DIR}/.task_"


@contextmanager
def file_lock(lock_path: Path, timeout: float = 30.0):
    """
    Context manager for file-based locking using fcntl.

    Uses advisory locking with exclusive access and timeout.
    Creates lock file if it doesn't exist.

    Args:
        lock_path: Path to the lock file
        timeout: Maximum time to wait for lock in seconds (default 30s)

    Raises:
        TimeoutError: If lock cannot be acquired within timeout
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT)
    start_time = time.time()

    try:
        while True:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break  # Successfully acquired lock
            except BlockingIOError:
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Could not acquire file lock within {timeout}s: {lock_path}")
                time.sleep(0.1)
        yield
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        except OSError:
            pass  # fd may be invalid if something went wrong
        os.close(lock_fd)


def _atomic_write(path: Path, content: str) -> None:
    """
    Write content to file atomically using temp file + rename.

    This prevents corruption if the process is interrupted mid-write.
    """
    # Create temp file in same directory to ensure same filesystem
    dir_path = path.parent
    dir_path.mkdir(parents=True, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(dir=str(dir_path), prefix=".tmp_")
    try:
        os.write(fd, content.encode("utf-8"))
        os.close(fd)
        fd = -1  # Mark as closed
        # Atomic rename
        os.rename(temp_path, path)
    except Exception:
        # Clean up fd if still open
        if fd >= 0:
            os.close(fd)
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def init_ralph_dir(cwd: str = ".") -> None:
    """Create .ralph/ directory structure."""
    base = Path(cwd)
    base.mkdir(parents=True, exist_ok=True)
    (base / RALPH_DIR).mkdir(exist_ok=True)
    (base / TASKS_DIR).mkdir(exist_ok=True)
    (base / AUDITS_DIR).mkdir(exist_ok=True)


def ralph_exists(cwd: str = ".") -> bool:
    """Check if .ralph/ directory exists."""
    return (Path(cwd) / RALPH_DIR).exists()


def goal_exists(cwd: str = ".") -> bool:
    """Check if goal.md exists."""
    return (Path(cwd) / GOAL_FILE).exists()


def pool_exists(cwd: str = ".") -> bool:
    """Check if pool.md exists."""
    return (Path(cwd) / POOL_FILE).exists()


# --- goal.md ---

def read_goal(cwd: str = ".") -> str:
    """Read goal.md content."""
    path = Path(cwd) / GOAL_FILE
    if not path.exists():
        return ""
    return path.read_text()


def write_goal(content: str, cwd: str = ".") -> None:
    """Write goal.md content."""
    init_ralph_dir(cwd)
    path = Path(cwd) / GOAL_FILE
    path.write_text(content)


# --- pool.md ---

def _read_pool_unlocked(cwd: str = ".") -> str:
    """Read pool.md content without locking (for use inside lock context)."""
    path = Path(cwd) / POOL_FILE
    if not path.exists():
        return ""
    return path.read_text()


def read_pool(cwd: str = ".") -> str:
    """Read pool.md content."""
    return _read_pool_unlocked(cwd)


def write_pool(content: str, cwd: str = ".") -> None:
    """
    Write pool.md content with file locking and atomic write.

    Uses:
    1. File lock to prevent concurrent writes
    2. Atomic write (temp file + rename) to prevent corruption
    """
    init_ralph_dir(cwd)
    base = Path(cwd)
    lock_path = base / POOL_LOCK_FILE
    target_path = base / POOL_FILE

    with file_lock(lock_path):
        _atomic_write(target_path, content)


def init_pool(goal_summary: str, initial_tasks: str, cwd: str = ".") -> None:
    """Initialize pool.md with goal and initial tasks."""
    content = f"""# Task Pool

## Goal Summary
{goal_summary}

## Active Tasks

{initial_tasks}

## Completed

(none yet)

## Deleted

(none)

## Findings

(discoveries shared across tasks)

## Verified Information (时效性验证缓存)

<!--
Format: [Verified YYYY-MM-DD] <topic>: <finding> (source: <url>)
Workers should check here before searching to avoid duplicate queries.
-->

(none yet)

## Progress Log

"""
    write_pool(content, cwd)


# --- tasks/*.md ---

def read_task(task_id: str, cwd: str = ".") -> str:
    """Read task detail file."""
    path = Path(cwd) / TASKS_DIR / f"{task_id}.md"
    if not path.exists():
        return ""
    return path.read_text()


def write_task(task_id: str, content: str, cwd: str = ".") -> None:
    """
    Write task detail file with per-task file locking.

    Each task has its own lock to allow concurrent writes to different tasks.
    """
    init_ralph_dir(cwd)
    base = Path(cwd)
    lock_path = base / f"{TASK_LOCK_PREFIX}{task_id}.lock"
    target_path = base / TASKS_DIR / f"{task_id}.md"

    with file_lock(lock_path):
        _atomic_write(target_path, content)


def task_exists(task_id: str, cwd: str = ".") -> bool:
    """Check if task file exists."""
    return (Path(cwd) / TASKS_DIR / f"{task_id}.md").exists()


def list_tasks(cwd: str = ".") -> list[str]:
    """List all task IDs."""
    tasks_path = Path(cwd) / TASKS_DIR
    if not tasks_path.exists():
        return []
    return [f.stem for f in tasks_path.glob("T*.md")]


def init_task(
    task_id: str,
    task_type: str,
    title: str,
    description: str,
    cwd: str = "."
) -> None:
    """Initialize a new task file."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    if task_type == "EXPLORE":
        content = f"""# {task_id}: {title}

## Type
EXPLORE

## Status
pending

## Created
{now}

## Question
{description}

## Exploration Hints
- (add hints here)

## Execution Log

(execution details will be recorded here)

## Findings

(discoveries from this exploration)

## Confidence

(high / medium / low - to be filled after execution)

## Follow-up Tasks

(suggested next tasks)
"""
    else:  # IMPLEMENT
        content = f"""# {task_id}: {title}

## Type
IMPLEMENT

## Status
pending

## Created
{now}

## Description
{description}

## Acceptance Criteria
- [ ] (add criteria here)

## Blocked By
- (none)

## Execution Log

(execution details will be recorded here)

## Files Changed

(list of modified files)

## Notes

(additional notes)
"""

    write_task(task_id, content, cwd)


def append_to_progress_log(entry: str, cwd: str = ".") -> None:
    """Append an entry to the Progress Log section in pool.md.

    Uses lock scope that covers both read and write for atomicity.
    """
    base = Path(cwd)
    lock_path = base / POOL_LOCK_FILE

    with file_lock(lock_path):
        pool_content = _read_pool_unlocked(cwd)

        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        log_entry = f"\n### {now}\n{entry}\n"

        # Append to end of file (Progress Log is last section)
        updated = pool_content.rstrip() + log_entry
        _atomic_write(base / POOL_FILE, updated)


def extract_task_ids_from_pool(cwd: str = ".") -> list[str]:
    """
    Extract all task IDs mentioned in pool.md.

    Looks for patterns like T001, T002, etc. in the task table.
    Returns a list of unique task IDs found.
    """
    pool_content = read_pool(cwd)
    if not pool_content:
        return []

    # Match task IDs like T001, T002, T123, etc.
    task_ids = re.findall(r'\bT\d{3,}\b', pool_content)

    # Return unique IDs preserving order
    seen = set()
    unique_ids = []
    for tid in task_ids:
        if tid not in seen:
            seen.add(tid)
            unique_ids.append(tid)

    return unique_ids


def ensure_task_files_exist(cwd: str = ".") -> list[str]:
    """
    Ensure all tasks in pool.md have corresponding task files.

    Creates missing task files with a basic template.
    Returns list of task IDs that were created.
    """
    task_ids = extract_task_ids_from_pool(cwd)
    created = []

    for task_id in task_ids:
        if not task_exists(task_id, cwd):
            # Create a minimal task file
            init_task(
                task_id=task_id,
                task_type="IMPLEMENT",  # Default to IMPLEMENT
                title=f"Task {task_id}",
                description="(Auto-created - details to be filled by Planner)",
                cwd=cwd,
            )
            created.append(task_id)

    return created


# --- Feedback and Checkpoints ---

def init_checkpoints_dir(cwd: str = ".") -> None:
    """Create checkpoints directory."""
    (Path(cwd) / CHECKPOINTS_DIR).mkdir(parents=True, exist_ok=True)


def create_checkpoint(
    task_id: str,
    path: str = None,
    artifact_type: str = "code",
    description: str = "",
    changes: list[str] = None,
    cwd: str = ".",
) -> Checkpoint:
    """
    Create a new checkpoint for a task.

    Uses UUID-based ID to avoid race conditions with concurrent checkpoint creation.

    Args:
        task_id: The task that produced this checkpoint
        path: Path to the artifact
        artifact_type: Type of artifact (code, model, binary, etc.)
        description: What this checkpoint contains
        changes: What changed from previous version

    Returns:
        New Checkpoint object
    """
    # Use UUID for unique checkpoint ID to avoid race conditions
    checkpoint_id = f"{task_id}_{uuid.uuid4().hex[:8]}"

    return Checkpoint(
        id=checkpoint_id,
        task_id=task_id,
        version=1,  # Version is now informational only
        path=path,
        artifact_type=artifact_type,
        description=description,
        changes=changes or [],
    )


def get_checkpoint(checkpoint_id: str, cwd: str = ".") -> Optional[Checkpoint]:
    """Get a specific checkpoint by ID."""
    manifest = read_checkpoint_manifest(cwd)
    if not manifest:
        return None

    for cp_data in manifest.get("checkpoints", []):
        if cp_data.get("id") == checkpoint_id:
            return Checkpoint.from_dict(cp_data)
    return None


def get_task_checkpoints(task_id: str, cwd: str = ".") -> list[Checkpoint]:
    """Get all checkpoints for a task."""
    manifest = read_checkpoint_manifest(cwd)
    if not manifest:
        return []

    checkpoints = []
    for cp_data in manifest.get("checkpoints", []):
        if cp_data.get("task_id") == task_id:
            checkpoints.append(Checkpoint.from_dict(cp_data))

    return sorted(checkpoints, key=lambda cp: cp.version)


def get_best_checkpoint(task_id: str, cwd: str = ".") -> Optional[Checkpoint]:
    """Get the best checkpoint for a task (highest user score or latest if untested)."""
    checkpoints = get_task_checkpoints(task_id, cwd)
    if not checkpoints:
        return None

    # First, look for explicitly marked best
    for cp in checkpoints:
        if cp.status == "best":
            return cp

    # Then, highest user score
    tested = [cp for cp in checkpoints if cp.user_score is not None]
    if tested:
        return max(tested, key=lambda cp: cp.user_score)

    # Finally, highest proxy score
    with_proxy = [cp for cp in checkpoints if cp.proxy_overall > 0]
    if with_proxy:
        return max(with_proxy, key=lambda cp: cp.proxy_overall)

    # Default to latest
    return checkpoints[-1]


def get_pending_checkpoints(cwd: str = ".") -> list[Checkpoint]:
    """Get all checkpoints pending user testing."""
    manifest = read_checkpoint_manifest(cwd)
    if not manifest:
        return []

    return [
        Checkpoint.from_dict(cp_data)
        for cp_data in manifest.get("checkpoints", [])
        if cp_data.get("status") == "pending"
    ]


def read_eval_config_from_goal(cwd: str = ".") -> dict:
    """
    Parse evaluation configuration from goal.md.

    Returns dict with keys: mode, test_frequency, batch_preference
    """
    goal_content = read_goal(cwd)
    config = {
        "mode": "全自动",
        "test_frequency": None,
        "batch_preference": None,
    }

    # Look for Evaluation Mode section
    eval_match = re.search(
        r"### Evaluation Mode\s*\n(.*?)(?=\n### |\n## |\Z)",
        goal_content,
        re.DOTALL
    )
    if eval_match:
        section = eval_match.group(1)

        mode_match = re.search(r"\*\*测试模式\*\*:\s*(.+)", section)
        if mode_match:
            config["mode"] = mode_match.group(1).strip()

        freq_match = re.search(r"\*\*测试频率\*\*:\s*(.+)", section)
        if freq_match:
            config["test_frequency"] = freq_match.group(1).strip()

        batch_match = re.search(r"\*\*测试安排\*\*:\s*(.+)", section)
        if batch_match:
            config["batch_preference"] = batch_match.group(1).strip()

    return config


def generate_feedback_template(
    checkpoints: list[Checkpoint] | list[dict],
    instructions: str,
    cwd: str = ".",
) -> str:
    """
    Generate feedback.md template for user to fill.

    Args:
        checkpoints: List of Checkpoint objects or dicts
        instructions: Testing instructions for user

    Returns:
        The generated template content
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# 测试反馈",
        "",
        f"生成时间: {now}",
        "",
        "请填写每个 checkpoint 的测试结果，然后运行 `ralph resume`",
        "",
        "## 测试说明",
        "",
        instructions,
        "",
        "## 待测试项",
        "",
    ]

    for cp in checkpoints:
        # Handle both Checkpoint objects and dicts
        if isinstance(cp, Checkpoint):
            cp_id = cp.id
            cp_path = cp.path or "N/A"
            proxy_scores = cp.proxy_scores
            description = cp.description
            changes = cp.changes
        else:
            cp_id = cp.get("id", "unknown")
            cp_path = cp.get("path", "N/A")
            proxy_scores = cp.get("proxy_scores", {})
            description = cp.get("description", "")
            changes = cp.get("changes", [])

        lines.append(f"### {cp_id}")
        lines.append(f"- 路径: `{cp_path}`")

        if description:
            lines.append(f"- 描述: {description}")

        if changes:
            lines.append("- 改动:")
            for change in changes:
                lines.append(f"  - {change}")

        if proxy_scores:
            lines.append("- 代理指标分数:")
            if isinstance(proxy_scores, list):
                # List of ProxyScore objects
                for ps in proxy_scores:
                    if isinstance(ps, ProxyScore):
                        status = "✓" if ps.passed else "✗"
                        lines.append(f"  - {ps.metric_name}: {ps.score:.1f} {status}")
                    else:
                        lines.append(f"  - {ps.get('metric_name', '?')}: {ps.get('score', 0):.1f}")
            elif isinstance(proxy_scores, dict):
                # Dict format (legacy)
                for k, v in proxy_scores.items():
                    lines.append(f"  - {k}: {v}")

        lines.append("")
        lines.append("**你的测试结果:**")
        lines.append("```")
        lines.append("成功率: ")
        lines.append("延迟表现: ")
        lines.append("其他观察: ")
        lines.append("评分 (1-5): ")
        lines.append("```")
        lines.append("")

    lines.extend([
        "## 总体反馈",
        "",
        "哪个版本更好？为什么？",
        "",
        "```",
        "",
        "```",
        "",
        "## 下一步",
        "",
        "- [ ] 基于最好的版本继续迭代",
        "- [ ] 尝试新方向",
        "- [ ] 结束，当前版本够用",
        "",
    ])

    content = "\n".join(lines)

    # Write to file
    path = Path(cwd) / FEEDBACK_FILE
    path.write_text(content)

    return content


def parse_feedback(cwd: str = ".") -> Optional[dict]:
    """
    Parse user's feedback from feedback.md.

    Returns dict with:
        - checkpoint_results: dict mapping checkpoint id to results
        - overall_feedback: str
        - next_step: str (continue/new_direction/finish)

    Returns None if feedback.md doesn't exist or isn't filled.
    """
    path = Path(cwd) / FEEDBACK_FILE
    if not path.exists():
        return None

    content = path.read_text()

    # Check if user has filled anything
    if "成功率: \n" in content:  # Still has empty template
        return None

    result = {
        "checkpoint_results": {},
        "overall_feedback": "",
        "next_step": "continue",
    }

    # Parse checkpoint results
    checkpoint_blocks = re.findall(
        r"### ([\w_]+)\n.*?\*\*你的测试结果:\*\*\s*```\n(.*?)```",
        content,
        re.DOTALL
    )
    for cp_id, results_text in checkpoint_blocks:
        results = {}
        for line in results_text.strip().split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                value = value.strip()
                if value:  # Only include non-empty values
                    results[key.strip()] = value
        if results:
            result["checkpoint_results"][cp_id] = results

    # Parse overall feedback
    overall_match = re.search(
        r"## 总体反馈.*?```\n(.*?)```",
        content,
        re.DOTALL
    )
    if overall_match:
        result["overall_feedback"] = overall_match.group(1).strip()

    # Parse next step
    if "[x] 基于最好的版本继续迭代" in content.lower() or "[X]" in content:
        result["next_step"] = "continue"
    elif "[x] 尝试新方向" in content.lower():
        result["next_step"] = "new_direction"
    elif "[x] 结束" in content.lower():
        result["next_step"] = "finish"

    return result


def save_checkpoint_manifest(
    checkpoints: list[Checkpoint] | list[dict],
    status: str = "waiting_for_user",
    instructions: str = "",
    cwd: str = ".",
) -> None:
    """
    Save checkpoint manifest for tracking pending tests.

    Uses file locking for concurrent safety.

    Args:
        checkpoints: List of Checkpoint objects or dicts
        status: Current status (waiting_for_user, completed, etc.)
        instructions: Testing instructions
    """
    init_checkpoints_dir(cwd)
    base = Path(cwd)
    lock_path = base / f"{CHECKPOINTS_DIR}/.manifest.lock"
    target_path = base / MANIFEST_FILE

    # Convert Checkpoint objects to dicts
    checkpoint_data = []
    for cp in checkpoints:
        if isinstance(cp, Checkpoint):
            checkpoint_data.append(cp.to_dict())
        else:
            checkpoint_data.append(cp)

    manifest = {
        "status": status,
        "created_at": datetime.now().isoformat(),
        "instructions": instructions,
        "checkpoints": checkpoint_data,
    }

    with file_lock(lock_path):
        _atomic_write(target_path, json.dumps(manifest, indent=2, ensure_ascii=False))


def read_checkpoint_manifest(cwd: str = ".") -> Optional[dict]:
    """Read checkpoint manifest if exists."""
    path = Path(cwd) / MANIFEST_FILE
    if not path.exists():
        return None
    return json.loads(path.read_text())


def update_checkpoint_with_results(
    checkpoint_id: str,
    results: dict,
    cwd: str = ".",
) -> None:
    """Update a checkpoint with user's test results."""
    manifest = read_checkpoint_manifest(cwd)
    if not manifest:
        return

    for cp in manifest["checkpoints"]:
        if cp["id"] == checkpoint_id:
            # Update with user results
            cp["user_results"] = results

            # Parse score if provided
            score_str = results.get("评分 (1-5)", "")
            if score_str:
                try:
                    cp["user_score"] = float(score_str)
                except ValueError:
                    pass

            cp["user_notes"] = results.get("其他观察", "")
            cp["tested_at"] = datetime.now().isoformat()
            cp["status"] = "tested"
            break

    save_checkpoint_manifest(
        checkpoints=manifest["checkpoints"],
        status=manifest["status"],
        instructions=manifest["instructions"],
        cwd=cwd,
    )


def add_checkpoint(checkpoint: Checkpoint, cwd: str = ".") -> None:
    """Add a new checkpoint to the manifest."""
    manifest = read_checkpoint_manifest(cwd)

    if manifest:
        checkpoints = manifest.get("checkpoints", [])
    else:
        checkpoints = []
        manifest = {"status": "in_progress", "instructions": ""}

    checkpoints.append(checkpoint.to_dict())

    save_checkpoint_manifest(
        checkpoints=checkpoints,
        status=manifest.get("status", "in_progress"),
        instructions=manifest.get("instructions", ""),
        cwd=cwd,
    )


def update_checkpoint(checkpoint: Checkpoint, cwd: str = ".") -> None:
    """Update an existing checkpoint in the manifest."""
    manifest = read_checkpoint_manifest(cwd)
    if not manifest:
        return

    checkpoints = manifest.get("checkpoints", [])
    for i, cp in enumerate(checkpoints):
        if cp["id"] == checkpoint.id:
            checkpoints[i] = checkpoint.to_dict()
            break

    save_checkpoint_manifest(
        checkpoints=checkpoints,
        status=manifest.get("status", "in_progress"),
        instructions=manifest.get("instructions", ""),
        cwd=cwd,
    )


def clear_feedback(cwd: str = ".") -> None:
    """Remove feedback.md after processing."""
    path = Path(cwd) / FEEDBACK_FILE
    if path.exists():
        path.unlink()


def is_waiting_for_user(cwd: str = ".") -> bool:
    """Check if system is waiting for user feedback."""
    manifest = read_checkpoint_manifest(cwd)
    return manifest is not None and manifest.get("status") == "waiting_for_user"


# -----------------------------------------------------------------------------
# Pessimistic Preparation Support
# -----------------------------------------------------------------------------


def mark_pending_test(task_id: str, cwd: str = ".") -> None:
    """
    Mark a task as pending user testing in pool.md.

    This does NOT pause execution - it just records the status.
    The Planner will decide whether to HEDGE or continue with other tasks.

    Uses lock scope that covers both read and write for atomicity.
    """
    base = Path(cwd)
    lock_path = base / POOL_LOCK_FILE

    with file_lock(lock_path):
        pool_content = _read_pool_unlocked(cwd)

        # Add a note to Findings section
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        note = f"\n**[{now}] {task_id} 待用户测试** - 继续探索替代方案\n"

        if "## Findings" in pool_content:
            pool_content = pool_content.replace(
                "## Findings",
                f"## Findings{note}"
            )
            _atomic_write(base / POOL_FILE, pool_content)


def append_failure_assumptions(
    task_id: str,
    assumptions: str,
    cwd: str = ".",
) -> None:
    """
    Record failure assumptions for a task in pool.md.

    These assumptions help the Planner understand what alternatives
    should be explored while waiting for user feedback.

    Uses lock scope that covers both read and write for atomicity.
    """
    base = Path(cwd)
    lock_path = base / POOL_LOCK_FILE

    with file_lock(lock_path):
        pool_content = _read_pool_unlocked(cwd)

        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Build the failure assumptions block
        assumptions_block = f"""

### Failure Assumptions for {task_id} ({now})
{assumptions}
"""

        # Check if there's already a Failure Assumptions section
        if "## Failure Assumptions" in pool_content:
            # Append to existing section
            pool_content = pool_content.replace(
                "## Failure Assumptions",
                f"## Failure Assumptions{assumptions_block}"
            )
        elif "## Findings" in pool_content:
            # Insert after Findings section header
            pool_content = pool_content.replace(
                "## Findings",
                f"## Findings\n\n## Failure Assumptions{assumptions_block}\n"
            )
        else:
            # Append to end
            pool_content = pool_content.rstrip() + f"\n\n## Failure Assumptions{assumptions_block}"

        _atomic_write(base / POOL_FILE, pool_content)


def get_pending_test_tasks(cwd: str = ".") -> list[str]:
    """
    Get list of task IDs that are pending user testing.

    Looks for tasks marked as "pending_test" in pool.md status.
    """
    pool_content = read_pool(cwd)
    if not pool_content:
        return []

    # Look for tasks with pending_test status in the task table
    # Format: | T001 | IMPLEMENT | pending_test | ...
    pending = re.findall(
        r'\|\s*(T\d+)\s*\|[^|]*\|\s*pending_test\s*\|',
        pool_content
    )

    return pending


def get_hedged_tasks(cwd: str = ".") -> list[str]:
    """
    Get list of task IDs that have been hedged (have alternatives being explored).

    Looks for tasks mentioned in the Failure Assumptions section.
    """
    pool_content = read_pool(cwd)
    if not pool_content:
        return []

    # Look for "Failure Assumptions for TXXX" patterns
    hedged = re.findall(
        r'### Failure Assumptions for (T\d+)',
        pool_content
    )

    return list(set(hedged))


# -----------------------------------------------------------------------------
# Verified Information Management (时效性验证缓存)
# -----------------------------------------------------------------------------


def add_verified_info(
    topic: str,
    finding: str,
    source_url: str,
    cwd: str = ".",
) -> None:
    """
    Add verified information to pool.md for reuse by other tasks.

    Format: [Verified YYYY-MM-DD] <topic>: <finding> (source: <url>)

    This helps avoid duplicate searches and ensures Workers use
    up-to-date information.

    Uses lock scope that covers both read and write for atomicity.
    """
    base = Path(cwd)
    lock_path = base / POOL_LOCK_FILE

    with file_lock(lock_path):
        pool_content = _read_pool_unlocked(cwd)
        if not pool_content:
            return

        today = datetime.now().strftime("%Y-%m-%d")
        entry = f"- [Verified {today}] {topic}: {finding} (source: {source_url})"

        if "## Verified Information" in pool_content:
            # Remove "(none yet)" placeholder if present
            pool_content = pool_content.replace("(none yet)", "", 1) if "(none yet)" in pool_content else pool_content

            # Find the next section after "## Verified Information" to insert before it
            vi_match = re.search(r"(## Verified Information[^\n]*\n)", pool_content)
            if vi_match:
                vi_end = vi_match.end()
                # Find the next "## " section header after the Verified Information header
                next_section = re.search(r"\n## ", pool_content[vi_end:])
                if next_section:
                    insert_pos = vi_end + next_section.start()
                else:
                    insert_pos = len(pool_content.rstrip())

                # Insert the entry before the next section (with trailing newline)
                pool_content = pool_content[:insert_pos].rstrip() + "\n" + entry + "\n\n" + pool_content[insert_pos:].lstrip("\n")
        else:
            # Section doesn't exist, add it before Progress Log
            if "## Progress Log" in pool_content:
                pool_content = pool_content.replace(
                    "## Progress Log",
                    f"## Verified Information (时效性验证缓存)\n\n{entry}\n\n## Progress Log"
                )
            else:
                # Append at end
                pool_content = pool_content.rstrip() + f"\n\n## Verified Information (时效性验证缓存)\n\n{entry}\n"

        _atomic_write(base / POOL_FILE, pool_content)


def get_verified_info(topic: str, cwd: str = ".") -> Optional[str]:
    """
    Check if a topic has already been verified.

    Args:
        topic: The topic to search for (case-insensitive partial match)
        cwd: Working directory

    Returns:
        The verified finding if found, None otherwise
    """
    pool_content = read_pool(cwd)
    if not pool_content:
        return None

    # Look for matching verified entries
    # Format: [Verified YYYY-MM-DD] <topic>: <finding> (source: <url>)
    pattern = rf'\[Verified \d{{4}}-\d{{2}}-\d{{2}}\]\s*{re.escape(topic)}[^:]*:\s*(.+?)\s*\(source:'
    match = re.search(pattern, pool_content, re.IGNORECASE)

    if match:
        return match.group(1).strip()
    return None


def list_verified_topics(cwd: str = ".") -> list[str]:
    """
    List all topics that have been verified.

    Returns:
        List of topic names
    """
    pool_content = read_pool(cwd)
    if not pool_content:
        return []

    # Extract all verified topics
    topics = re.findall(
        r'\[Verified \d{4}-\d{2}-\d{2}\]\s*([^:]+):',
        pool_content
    )

    return [t.strip() for t in topics]


def is_topic_verified(topic: str, cwd: str = ".") -> bool:
    """
    Quick check if a topic has been verified.

    Args:
        topic: The topic to check (case-insensitive partial match)

    Returns:
        True if the topic has been verified
    """
    return get_verified_info(topic, cwd) is not None


# -----------------------------------------------------------------------------
# Pivot Recommendation Management (信息断层修复)
# -----------------------------------------------------------------------------


def append_to_findings(finding: str, cwd: str = ".") -> None:
    """
    Append a finding to pool.md Findings section.

    Used to propagate Evaluator's pivot recommendations to Planner.
    Format: **[PIVOT_RECOMMENDED]** task_id: reason

    Uses lock scope that covers both read and write for atomicity.
    """
    base = Path(cwd)
    lock_path = base / POOL_LOCK_FILE

    with file_lock(lock_path):
        pool_content = _read_pool_unlocked(cwd)
        if not pool_content:
            return

        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = f"\n- [{now}] {finding}\n"

        # Find the Findings section and append
        if "## Findings" in pool_content or "## Shared Findings" in pool_content:
            # Check for either section name
            if "## Findings" in pool_content:
                section_marker = "## Findings"
            else:
                section_marker = "## Shared Findings"

            # Insert after the section header
            pool_content = pool_content.replace(
                section_marker,
                f"{section_marker}{entry}"
            )
        else:
            # Create Findings section before Progress Log
            if "## Progress Log" in pool_content:
                pool_content = pool_content.replace(
                    "## Progress Log",
                    f"## Findings{entry}\n## Progress Log"
                )
            else:
                # Append at end
                pool_content = pool_content.rstrip() + f"\n\n## Findings{entry}"

        _atomic_write(base / POOL_FILE, pool_content)


def clear_pivot_recommendation(task_id: str, cwd: str = ".") -> None:
    """
    Clear or mark as processed a pivot recommendation from pool.md.

    Changes [PIVOT_RECOMMENDED] to [PIVOT_PROCESSED] for the given task.
    Uses regex to tolerate format variants (bold, extra spaces, etc.).

    Uses lock scope that covers both read and write for atomicity.
    """
    base = Path(cwd)
    lock_path = base / POOL_LOCK_FILE

    with file_lock(lock_path):
        pool_content = _read_pool_unlocked(cwd)
        if not pool_content:
            return

        # Use regex to match any line containing [PIVOT_RECOMMENDED] and the task_id.
        # Tolerates: **[PIVOT_RECOMMENDED]**, [PIVOT_RECOMMENDED], extra spaces,
        # bold markers around it, etc.
        pattern = re.compile(
            r'(\*{0,2})\[PIVOT_RECOMMENDED\](\*{0,2})\s*' + re.escape(task_id),
            re.IGNORECASE,
        )
        pool_content = pattern.sub(
            lambda m: f'{m.group(1)}[PIVOT_PROCESSED]{m.group(2)} {task_id}',
            pool_content,
        )

        _atomic_write(base / POOL_FILE, pool_content)


def has_pivot_recommendation(cwd: str = ".") -> bool:
    """
    Check if there are any unprocessed PIVOT recommendations.

    Returns:
        True if [PIVOT_RECOMMENDED] exists in pool.md
    """
    pool_content = read_pool(cwd)
    return "[PIVOT_RECOMMENDED]" in pool_content


# -----------------------------------------------------------------------------
# Pool Status Updates
# -----------------------------------------------------------------------------


def update_pool_status(new_status: str, cwd: str = ".") -> None:
    """
    Update or insert a ## Status line in pool.md.

    Handles both formats:
    - "## Status: VALUE" (colon on same line)
    - "## Status\\nVALUE" (value on next line)

    If no Status section exists, appends one after the header.
    """
    base = Path(cwd)
    lock_path = base / POOL_LOCK_FILE

    with file_lock(lock_path):
        pool_content = _read_pool_unlocked(cwd)
        if not pool_content:
            return

        # Try to replace existing status (colon format or newline format)
        updated, count = re.subn(
            r"## Status[:\s]+\S+",
            f"## Status: {new_status}",
            pool_content,
            count=1,
        )
        if count > 0:
            _atomic_write(base / POOL_FILE, updated)
        else:
            # No Status section found — append after first line
            lines = pool_content.split("\n", 1)
            if len(lines) > 1:
                new_content = lines[0] + f"\n\n## Status: {new_status}\n" + lines[1]
            else:
                new_content = pool_content + f"\n\n## Status: {new_status}\n"
            _atomic_write(base / POOL_FILE, new_content)


def mark_pending_tasks_skipped(reason: str, cwd: str = ".") -> list[str]:
    """
    Mark all pending tasks as skipped/superseded in their task files.

    Called when the planner chooses DONE with remaining pending tasks.

    Returns list of task IDs that were marked skipped.
    """
    task_ids = extract_task_ids_from_pool(cwd)
    skipped = []

    for tid in task_ids:
        task_content = read_task(tid, cwd)
        if not task_content:
            continue

        content_lower = task_content.lower()
        # Skip optional emoji prefix (e.g. "## Status: ✅ completed")
        status_match = re.search(r"## status[:\s]+(?:[^\w\s]\s*)?(\w+)", content_lower)
        if status_match and status_match.group(1) == "pending":
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            skip_note = f"\n\n## Skipped ({now})\n**Reason**: {reason}\n"
            # Update status to skipped
            task_content = re.sub(
                r"(## Status[:\s]+)\S+",
                r"\1skipped",
                task_content,
                count=1,
                flags=re.IGNORECASE,
            )
            write_task(tid, task_content + skip_note, cwd)
            skipped.append(tid)

    return skipped


# -----------------------------------------------------------------------------
# Context Compaction (上下文压缩)
# -----------------------------------------------------------------------------

PROGRESS_ARCHIVE_FILE = f"{RALPH_DIR}/progress_archive.md"
POOL_WORD_COUNT_THRESHOLD = 1500


def get_pool_word_count(cwd: str = ".") -> int:
    """Get word count of pool.md."""
    content = read_pool(cwd)
    if not content:
        return 0
    return len(content.split())


def pool_needs_compaction(cwd: str = ".") -> bool:
    """Check if pool.md exceeds the word count threshold."""
    return get_pool_word_count(cwd) > POOL_WORD_COUNT_THRESHOLD


def extract_pool_sections(pool_content: str) -> dict[str, str]:
    """
    Extract named sections from pool.md.

    Returns dict mapping section name to content (including header).
    """
    sections = {}
    current_name = "_header"
    current_lines = []

    for line in pool_content.split("\n"):
        if line.startswith("## "):
            if current_lines:
                sections[current_name] = "\n".join(current_lines)
            current_name = line.lstrip("# ").strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections[current_name] = "\n".join(current_lines)

    return sections


def archive_progress_log(pool_content: str, keep_recent: int = 5, cwd: str = ".") -> str:
    """
    Archive old progress log entries, keeping only the most recent ones.

    Returns the updated pool_content with trimmed Progress Log.
    Old entries are appended to .ralph/progress_archive.md.
    """
    # Find Progress Log section
    log_match = re.search(
        r"(## Progress Log\s*\n)(.*?)(\Z)",
        pool_content,
        re.DOTALL,
    )
    if not log_match:
        return pool_content

    header = log_match.group(1)
    log_body = log_match.group(2)

    # Split into entries (each starts with ### YYYY-MM-DD)
    entries = re.split(r"(?=\n### \d{4}-\d{2}-\d{2})", log_body)
    entries = [e for e in entries if e.strip()]

    if len(entries) <= keep_recent:
        return pool_content  # Nothing to archive

    # Split into old and recent
    old_entries = entries[:-keep_recent]
    recent_entries = entries[-keep_recent:]

    # Archive old entries
    archive_path = Path(cwd) / PROGRESS_ARCHIVE_FILE
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with open(archive_path, "a") as f:
        for entry in old_entries:
            f.write(entry.rstrip() + "\n")

    # Rebuild pool content
    new_log = header + "\n".join(recent_entries)
    updated = pool_content[:log_match.start()] + new_log
    return updated


async def compact_pool(cwd: str = ".") -> bool:
    """
    Compact pool.md using Claude to summarize verbose sections.

    Steps:
    1. Archive old progress log entries to progress_archive.md
    2. Use Claude to compress Findings and Failure Assumptions
    3. Remove expired Verified Information (>7 days)

    Returns True if compaction was performed.
    """
    from .prompts import COMPACTION_PROMPT

    pool_content = read_pool(cwd)
    if not pool_content:
        return False

    # Step 1: Archive progress log (code-based, no LLM needed)
    pool_content = archive_progress_log(pool_content, keep_recent=5, cwd=cwd)

    # Step 2: Use Claude to compress the rest
    today = datetime.now().strftime("%Y-%m-%d")
    prompt = COMPACTION_PROMPT.format(today=today, pool_content=pool_content)

    from claude_agent_sdk import ClaudeAgentOptions
    from .logger import stream_query

    sr = await stream_query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            system_prompt="You are a concise text compactor. Output only the compressed pool.md content.",
            allowed_tools=[],
            permission_mode="default",
            max_turns=1,
            cwd=cwd,
        ),
        agent_name="compactor",
        emoji="📦",
        cwd=cwd,
        verbose=False,
    )

    # Extract the compacted content from response
    compacted = sr.text.strip()

    # Sanity check: compacted content should have key sections
    if "## Active Tasks" not in compacted and "## Goal Summary" not in compacted:
        # Compaction output looks invalid, skip
        return False

    write_pool(compacted, cwd)
    return True


# -----------------------------------------------------------------------------
# Handoff Notes (Resume 支持)
# -----------------------------------------------------------------------------

HANDOFF_FILE = f"{RALPH_DIR}/handoff.md"


def generate_handoff_note(cwd: str = ".") -> str:
    """
    Generate a handoff note summarizing current session state.

    Reads pool.md and task files to build a structured summary
    for the next session to resume from.

    Returns the handoff content.
    """
    pool_content = read_pool(cwd)
    if not pool_content:
        return ""

    # Parse task statuses from pool
    task_ids = extract_task_ids_from_pool(cwd)

    completed = []
    in_progress = []
    pending = []

    for tid in task_ids:
        task_content = read_task(tid, cwd)
        if not task_content:
            pending.append(tid)
            continue

        content_lower = task_content.lower()
        # Match both "## Status\nvalue" and "## Status: value" formats
        # Skip optional emoji prefix (e.g. "## Status: ✅ completed")
        status_match = re.search(r"## status[:\s]+(?:[^\w\s]\s*)?(\w+)", content_lower)
        if status_match:
            status_val = status_match.group(1)
            if status_val in ("completed", "passed", "done"):
                completed.append(tid)
            elif status_val == "pending":
                pending.append(tid)
            else:
                in_progress.append(tid)
        else:
            in_progress.append(tid)

    # Extract key findings from pool (matches both "## Findings" and "## Shared Findings")
    findings_match = re.search(
        r"## (?:Shared )?Findings\s*\n(.*?)(?=\n## |\Z)",
        pool_content,
        re.DOTALL,
    )
    findings_text = findings_match.group(1).strip() if findings_match else "(none)"

    # Extract failure assumptions
    failure_match = re.search(
        r"## Failure Assumptions\s*\n(.*?)(?=\n## |\Z)",
        pool_content,
        re.DOTALL,
    )
    failure_text = failure_match.group(1).strip() if failure_match else "(none)"

    # Extract recent progress (last 3 entries)
    progress_match = re.search(
        r"## Progress Log\s*\n(.*?)(?=\Z)",
        pool_content,
        re.DOTALL,
    )
    progress_text = ""
    if progress_match:
        entries = re.split(r"(?=### \d{4}-\d{2}-\d{2})", progress_match.group(1))
        entries = [e.strip() for e in entries if e.strip()]
        recent = entries[-3:] if len(entries) > 3 else entries
        progress_text = "\n".join(recent)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    handoff = f"""# Handoff Note
Generated: {now}

## 当前状态
- 已完成: {', '.join(completed) if completed else '(none)'}
- 进行中: {', '.join(in_progress) if in_progress else '(none)'}
- 待处理: {', '.join(pending) if pending else '(none)'}

## 关键发现
{findings_text}

## 失败假设
{failure_text}

## 最近进展
{progress_text}

## 建议下一步
(Planner should analyze the above and decide)
"""

    # Write to file
    path = Path(cwd) / HANDOFF_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(handoff)

    return handoff


def read_handoff_note(cwd: str = ".") -> str:
    """Read handoff note if it exists."""
    path = Path(cwd) / HANDOFF_FILE
    if not path.exists():
        return ""
    return path.read_text()


def clear_handoff_note(cwd: str = ".") -> None:
    """Remove handoff note after it's been consumed."""
    path = Path(cwd) / HANDOFF_FILE
    if path.exists():
        path.unlink()
