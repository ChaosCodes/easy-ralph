"""
Shared utilities for MCP validation experiments.

Provides:
- Temporary .ralph directory setup/teardown
- Result collection and reporting
- Pool.md / task file creation helpers
"""

import json
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class TestCase:
    """A single test case with pass/fail result."""
    name: str
    passed: bool
    details: str = ""
    category: str = "unit"  # "unit" or "integration"


@dataclass
class ExperimentResult:
    """Aggregated result of an experiment."""
    experiment_name: str
    description: str
    test_cases: list[TestCase] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def total(self) -> int:
        return len(self.test_cases)

    @property
    def passed(self) -> int:
        return sum(1 for t in self.test_cases if t.passed)

    @property
    def failed(self) -> int:
        return sum(1 for t in self.test_cases if not t.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0

    def add(self, name: str, passed: bool, details: str = "", category: str = "unit"):
        self.test_cases.append(TestCase(name=name, passed=passed, details=details, category=category))

    def summary(self) -> str:
        lines = [
            f"{'=' * 60}",
            f"  {self.experiment_name}",
            f"  {self.description}",
            f"{'=' * 60}",
            f"  Total: {self.total}  Passed: {self.passed}  Failed: {self.failed}  ({self.pass_rate:.0%})",
            "",
        ]

        # Group by category
        categories = {}
        for tc in self.test_cases:
            categories.setdefault(tc.category, []).append(tc)

        for cat, cases in categories.items():
            cat_passed = sum(1 for c in cases if c.passed)
            lines.append(f"  [{cat}] {cat_passed}/{len(cases)}")
            for tc in cases:
                status = "PASS" if tc.passed else "FAIL"
                icon = "  " if tc.passed else "  "
                lines.append(f"    {icon} [{status}] {tc.name}")
                if tc.details and not tc.passed:
                    for detail_line in tc.details.split("\n"):
                        lines.append(f"           {detail_line}")
            lines.append("")

        return "\n".join(lines)

    def save(self, output_dir: str = None) -> Path:
        if output_dir is None:
            output_dir = Path(__file__).parent / "results"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"{self.experiment_name}_{timestamp}.json"
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
        return path


class RalphTestDir:
    """Context manager for creating temporary .ralph directories for testing."""

    def __init__(self, prefix: str = "ralph_test_"):
        self.prefix = prefix
        self.path: Optional[Path] = None

    def __enter__(self) -> "RalphTestDir":
        self.path = Path(tempfile.mkdtemp(prefix=self.prefix))
        return self

    def __exit__(self, *args):
        if self.path and self.path.exists():
            shutil.rmtree(self.path)

    @property
    def cwd(self) -> str:
        return str(self.path)

    def setup_basic_project(
        self,
        goal: str = "Test goal",
        tasks: list[dict] = None,
        pool_extra: str = "",
    ) -> None:
        """Set up a minimal .ralph directory with goal.md, pool.md, and task files."""
        from ralph_sdk.pool import init_ralph_dir, write_goal, init_pool, init_task

        init_ralph_dir(self.cwd)
        write_goal(goal, self.cwd)

        # Build task table
        if tasks:
            task_table_lines = ["| ID | Type | Status | Title |", "|---|---|---|---|"]
            for t in tasks:
                task_table_lines.append(
                    f"| {t['id']} | {t.get('type', 'EXPLORE')} | {t.get('status', 'pending')} | {t.get('title', 'Task')} |"
                )
                init_task(
                    task_id=t["id"],
                    task_type=t.get("type", "EXPLORE"),
                    title=t.get("title", "Task"),
                    description=t.get("description", "Test task"),
                    cwd=self.cwd,
                )
            task_table = "\n".join(task_table_lines)
        else:
            task_table = "(no tasks)"

        init_pool(
            goal_summary=goal[:100],
            initial_tasks=task_table,
            cwd=self.cwd,
        )

        # Append extra content to pool.md if provided
        if pool_extra:
            from ralph_sdk.pool import read_pool, write_pool
            content = read_pool(self.cwd)
            content += "\n" + pool_extra
            write_pool(content, self.cwd)

    def read_pool(self) -> str:
        from ralph_sdk.pool import read_pool
        return read_pool(self.cwd)

    def read_task(self, task_id: str) -> str:
        from ralph_sdk.pool import read_task
        return read_task(task_id, self.cwd)

    def list_task_files(self) -> list[str]:
        """List actual task files on disk."""
        tasks_dir = self.path / ".ralph" / "tasks"
        if not tasks_dir.exists():
            return []
        return sorted(f.stem for f in tasks_dir.glob("T*.md"))

    def count_tool_calls(self, tool_name: str) -> int:
        """Count tool calls of a specific type in tool_calls.jsonl."""
        log_path = self.path / ".ralph" / "logs" / "tool_calls.jsonl"
        if not log_path.exists():
            return 0
        count = 0
        with open(log_path) as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if record.get("tool") == tool_name:
                        count += 1
                except json.JSONDecodeError:
                    pass
        return count

    def get_tool_calls(self, tool_name: str = None) -> list[dict]:
        """Get all tool calls, optionally filtered by name."""
        log_path = self.path / ".ralph" / "logs" / "tool_calls.jsonl"
        if not log_path.exists():
            return []
        calls = []
        with open(log_path) as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if tool_name is None or record.get("tool") == tool_name:
                        calls.append(record)
                except json.JSONDecodeError:
                    pass
        return calls
