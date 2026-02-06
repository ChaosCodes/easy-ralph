"""
Experiment: Find Hidden Bugs (No Hints)

Code with bugs but NO comments indicating where they are.
This is a realistic scenario.
"""

import asyncio
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from claude_code_sdk import AssistantMessage, ClaudeCodeOptions, query
from rich.console import Console

console = Console()


def setup_project(base_dir: Path, name: str) -> Path:
    project_dir = base_dir / name
    if project_dir.exists():
        shutil.rmtree(project_dir)
    project_dir.mkdir(parents=True)
    (project_dir / "src").mkdir()
    (project_dir / "src" / "__init__.py").write_text("")

    # Cache implementation with hidden bugs
    cache_py = '''"""Simple LRU cache implementation."""
from collections import OrderedDict
from typing import Any, Optional
import threading
import time


class LRUCache:
    """Least Recently Used cache with expiration."""

    def __init__(self, max_size: int = 100, ttl_seconds: float = 300):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None

            value, timestamp = self.cache[key]

            # Check if expired
            if time.time() - timestamp > self.ttl:
                del self.cache[key]
                self.misses += 1
                return None

            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        with self.lock:
            if key in self.cache:
                # Update existing
                self.cache[key] = (value, time.time())
                self.cache.move_to_end(key)
            else:
                # Add new
                if len(self.cache) >= self.max_size:
                    # Remove oldest
                    self.cache.popitem(last=False)
                self.cache[key] = (value, time.time())

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if key in self.cache:
            del self.cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all entries."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
        }


class AsyncCache:
    """Async wrapper for LRU cache."""

    def __init__(self, max_size: int = 100, ttl_seconds: float = 300):
        self.cache = LRUCache(max_size, ttl_seconds)
        self.pending: dict[str, asyncio.Future] = {}

    async def get_or_compute(self, key: str, compute_fn) -> Any:
        """Get from cache or compute if missing."""
        # Check cache first
        value = self.cache.get(key)
        if value is not None:
            return value

        # Check if already computing
        if key in self.pending:
            return await self.pending[key]

        # Compute new value
        future = asyncio.Future()
        self.pending[key] = future

        try:
            value = await compute_fn()
            self.cache.set(key, value)
            future.set_result(value)
            return value
        except Exception as e:
            future.set_exception(e)
            raise
        finally:
            del self.pending[key]
'''
    (project_dir / "src" / "cache.py").write_text(cache_py)

    return project_dir


# The bugs in this code:
# 1. delete() doesn't use self.lock - thread unsafe
# 2. clear() doesn't use self.lock - thread unsafe
# 3. stats() doesn't use self.lock - can return inconsistent values
# 4. AsyncCache.get_or_compute has race condition - between cache.get() and
#    checking self.pending, another coroutine could start computing
# 5. If compute_fn returns None, it will be treated as cache miss on next get
# 6. pending dict operations not protected, could have issues with concurrent access

TASK_PROMPT = """
Review src/cache.py for bugs.

This is an LRU cache with TTL expiration and an async wrapper.
The code is used in production with concurrent access.

Look for:
- Thread safety issues
- Race conditions
- Logic bugs
- Edge cases

List ALL bugs you find with:
1. Location (class/method)
2. Description of the bug
3. How it could cause problems in production

Be thorough - this code handles concurrent access.
"""


async def run_analysis(project_dir: Path) -> str:
    result_text = ""
    async for message in query(
        prompt=TASK_PROMPT,
        options=ClaudeCodeOptions(
            system_prompt="You are reviewing code for a production system. Be thorough.",
            allowed_tools=["Read", "Write"],
            permission_mode="acceptEdits",
            max_turns=10,
            cwd=str(project_dir),
        ),
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    result_text += block.text
    return result_text


def evaluate_analysis(report: str) -> dict:
    """Strict evaluation - check for specific bugs."""
    bugs = {
        "delete_no_lock": False,
        "clear_no_lock": False,
        "stats_no_lock": False,
        "async_race_condition": False,
        "none_value_bug": False,
        "pending_not_protected": False,
    }

    report_lower = report.lower()

    # delete() missing lock
    if "delete" in report_lower and ("lock" in report_lower or "thread" in report_lower):
        if "missing" in report_lower or "without" in report_lower or "doesn't" in report_lower or "not" in report_lower:
            bugs["delete_no_lock"] = True

    # clear() missing lock
    if "clear" in report_lower and ("lock" in report_lower or "thread" in report_lower):
        bugs["clear_no_lock"] = True

    # stats() consistency
    if "stats" in report_lower and ("lock" in report_lower or "inconsistent" in report_lower or "thread" in report_lower):
        bugs["stats_no_lock"] = True

    # AsyncCache race condition
    if "get_or_compute" in report_lower or "async" in report_lower:
        if "race" in report_lower or "concurrent" in report_lower:
            bugs["async_race_condition"] = True

    # None value bug
    if "none" in report_lower and ("return" in report_lower or "value" in report_lower or "cache" in report_lower):
        if "miss" in report_lower or "compute" in report_lower:
            bugs["none_value_bug"] = True

    # pending dict not protected
    if "pending" in report_lower and ("protect" in report_lower or "lock" in report_lower or "concurrent" in report_lower):
        bugs["pending_not_protected"] = True

    found = sum(bugs.values())
    total = len(bugs)

    return {
        "bugs_found": bugs,
        "score": found,
        "max_score": total,
        "percentage": found / total * 100,
    }


async def run_experiment():
    console.print("\n[bold cyan]Experiment: Find Hidden Bugs (No Hints)[/bold cyan]\n")
    console.print("LRU cache with TTL - no bug comments in code.")
    console.print("6 hidden bugs to find.\n")

    base_dir = Path("/tmp/ralph_exp_hidden")
    base_dir.mkdir(exist_ok=True)

    results = []

    for run_idx in range(2):
        console.print(f"[bold]Run {run_idx + 1}/2[/bold]")

        project_dir = setup_project(base_dir, f"run_{run_idx}")
        report = await run_analysis(project_dir)
        evaluation = evaluate_analysis(report)
        results.append(evaluation)

        console.print(f"  Found: {evaluation['score']}/{evaluation['max_score']}")
        for bug, found in evaluation["bugs_found"].items():
            status = "[green]✓[/green]" if found else "[red]✗[/red]"
            console.print(f"    {status} {bug}")
        console.print()

    avg = sum(r["percentage"] for r in results) / len(results)
    console.print(f"\n[bold]Average: {avg:.0f}%[/bold]")

    if avg < 50:
        console.print("\n[yellow]Finding hidden bugs without hints is hard![/yellow]")
    elif avg < 80:
        console.print("\n[blue]Found most but not all bugs[/blue]")
    else:
        console.print("\n[green]Excellent bug detection![/green]")


if __name__ == "__main__":
    asyncio.run(run_experiment())
