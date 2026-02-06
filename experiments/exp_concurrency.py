"""
Experiment: Concurrency Bug Detection

This is designed to be genuinely hard - finding race conditions
and timing bugs that require careful analysis.
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

    # A rate limiter with multiple subtle bugs
    rate_limiter = '''"""Token bucket rate limiter with bugs."""
import asyncio
import time
from dataclasses import dataclass
from typing import Optional

@dataclass
class RateLimiter:
    """Token bucket rate limiter.

    BUG 1: Race condition in acquire() - check and decrement not atomic
    BUG 2: refill() can exceed max_tokens
    BUG 3: wait_for_token() busy-waits instead of using asyncio.sleep properly
    BUG 4: No lock protection for concurrent access
    """
    max_tokens: int
    refill_rate: float  # tokens per second
    tokens: float = 0
    last_refill: float = 0

    def __post_init__(self):
        self.tokens = float(self.max_tokens)
        self.last_refill = time.time()

    def refill(self) -> None:
        """Add tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        # BUG 2: Should cap at max_tokens
        self.tokens += elapsed * self.refill_rate
        self.last_refill = now

    def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens. Returns True if successful."""
        self.refill()
        # BUG 1: Race condition - another coroutine could modify tokens
        # between the check and the decrement
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    async def wait_for_token(self, tokens: int = 1) -> None:
        """Wait until tokens are available, then acquire."""
        # BUG 3: Busy-waiting, inefficient
        while not self.acquire(tokens):
            await asyncio.sleep(0)  # Should calculate actual wait time


class RequestQueue:
    """Queue requests with rate limiting.

    BUG 5: Lost wakeup - if notify() is called before wait(),
           the notification is lost
    BUG 6: No timeout handling
    """
    def __init__(self, limiter: RateLimiter):
        self.limiter = limiter
        self.queue: list = []
        self.event = asyncio.Event()

    async def enqueue(self, request: str) -> None:
        """Add request to queue."""
        self.queue.append(request)
        self.event.set()  # BUG 5: Event might be set but no one waiting yet

    async def process_next(self) -> Optional[str]:
        """Process next request when rate limit allows."""
        if not self.queue:
            await self.event.wait()  # BUG 5: Might miss the set()
            self.event.clear()

        if self.queue:
            await self.limiter.wait_for_token()
            return self.queue.pop(0)
        return None
'''
    (project_dir / "src" / "rate_limiter.py").write_text(rate_limiter)

    return project_dir


TASK_PROMPT = """
Review src/rate_limiter.py for concurrency bugs.

This is a token bucket rate limiter with a request queue.
It's designed for async usage with multiple coroutines.

Find and document ALL concurrency issues:
1. Race conditions
2. Lost wakeups
3. Busy-waiting
4. Missing synchronization
5. Other timing issues

For each bug found:
- Explain the bug
- Show a scenario where it causes problems
- Suggest the fix (but don't implement - just document)

Output your analysis as a markdown report.
"""


async def run_analysis(project_dir: Path) -> str:
    """Run bug analysis and return the report."""
    result_text = ""

    async for message in query(
        prompt=TASK_PROMPT,
        options=ClaudeCodeOptions(
            system_prompt="You are a concurrency expert reviewing async Python code.",
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
    """Check if all bugs were found."""
    bugs = {
        "race_condition_acquire": False,
        "tokens_exceed_max": False,
        "busy_wait": False,
        "no_lock": False,
        "lost_wakeup": False,
        "no_timeout": False,
    }

    report_lower = report.lower()

    # Check for each bug
    if "race" in report_lower and "acquire" in report_lower:
        bugs["race_condition_acquire"] = True
    if ("exceed" in report_lower or "cap" in report_lower or "max" in report_lower) and "token" in report_lower:
        bugs["tokens_exceed_max"] = True
    if "busy" in report_lower or ("sleep(0)" in report_lower) or "spin" in report_lower:
        bugs["busy_wait"] = True
    if "lock" in report_lower or "mutex" in report_lower or "synchron" in report_lower:
        bugs["no_lock"] = True
    if "lost" in report_lower or "miss" in report_lower or "wakeup" in report_lower:
        bugs["lost_wakeup"] = True
    if "timeout" in report_lower:
        bugs["no_timeout"] = True

    found = sum(bugs.values())
    total = len(bugs)

    return {
        "bugs_found": bugs,
        "score": found,
        "max_score": total,
        "percentage": found / total * 100,
    }


async def run_experiment():
    console.print("\n[bold cyan]Experiment: Concurrency Bug Detection[/bold cyan]\n")
    console.print("Testing ability to find subtle concurrency bugs.")
    console.print("This requires understanding async/await, race conditions, etc.\n")

    base_dir = Path("/tmp/ralph_exp_concurrency")
    base_dir.mkdir(exist_ok=True)

    results = []

    for run_idx in range(2):
        console.print(f"[bold]Run {run_idx + 1}/2[/bold]")

        project_dir = setup_project(base_dir, f"run_{run_idx}")
        report = await run_analysis(project_dir)
        evaluation = evaluate_analysis(report)
        results.append(evaluation)

        console.print(f"  Bugs found: {evaluation['score']}/{evaluation['max_score']}")
        for bug, found in evaluation["bugs_found"].items():
            status = "[green]✓[/green]" if found else "[red]✗[/red]"
            console.print(f"    {status} {bug}")
        console.print()

    avg = sum(r["percentage"] for r in results) / len(results)
    console.print(f"\n[bold]Average: {avg:.0f}%[/bold]")

    if avg < 50:
        console.print("[red]Concurrency bugs are hard to spot![/red]")
    elif avg < 80:
        console.print("[yellow]Found some but not all bugs[/yellow]")
    else:
        console.print("[green]Good concurrency understanding![/green]")


if __name__ == "__main__":
    asyncio.run(run_experiment())
