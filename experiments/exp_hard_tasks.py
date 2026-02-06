"""
Experiment: Tasks I Think I'll Fail At

Design tasks that exploit known weaknesses:
1. Multi-file state tracking
2. Implicit constraints
3. Long-range dependencies
"""

import asyncio
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from claude_code_sdk import AssistantMessage, ClaudeCodeOptions, query
from rich.console import Console
from rich.table import Table

console = Console()


def setup_state_machine_project(base_dir: Path, name: str) -> Path:
    """
    Create a project with a state machine that has subtle bugs
    when states are modified in one file but not another.
    """
    project_dir = base_dir / name
    if project_dir.exists():
        shutil.rmtree(project_dir)
    project_dir.mkdir(parents=True)
    (project_dir / "src").mkdir()
    (project_dir / "src" / "__init__.py").write_text("")

    # State definitions - the source of truth
    states_py = '''"""Order states."""
from enum import Enum

class OrderState(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"

# Valid transitions
TRANSITIONS = {
    OrderState.PENDING: [OrderState.CONFIRMED, OrderState.CANCELLED],
    OrderState.CONFIRMED: [OrderState.SHIPPED, OrderState.CANCELLED],
    OrderState.SHIPPED: [OrderState.DELIVERED],
    OrderState.DELIVERED: [],
    OrderState.CANCELLED: [],
}
'''
    (project_dir / "src" / "states.py").write_text(states_py)

    # Order class - uses states
    order_py = '''"""Order model."""
from .states import OrderState, TRANSITIONS

class Order:
    def __init__(self, order_id: str):
        self.order_id = order_id
        self.state = OrderState.PENDING
        self.history = []

    def transition(self, new_state: OrderState) -> bool:
        """Attempt state transition. Returns True if successful."""
        allowed = TRANSITIONS.get(self.state, [])
        if new_state in allowed:
            self.history.append((self.state, new_state))
            self.state = new_state
            return True
        return False

    def can_cancel(self) -> bool:
        """Check if order can be cancelled."""
        # BUG: hardcoded states, doesn't use TRANSITIONS
        return self.state in [OrderState.PENDING, OrderState.CONFIRMED]
'''
    (project_dir / "src" / "order.py").write_text(order_py)

    # Notification service - another file that knows about states
    notifications_py = '''"""Notification service."""
from .states import OrderState

def get_notification_template(state: OrderState) -> str:
    """Get email template for state."""
    # BUG: Missing CANCELLED state, will raise KeyError
    templates = {
        OrderState.PENDING: "Your order is being processed.",
        OrderState.CONFIRMED: "Your order has been confirmed!",
        OrderState.SHIPPED: "Your order is on its way!",
        OrderState.DELIVERED: "Your order has been delivered.",
        # CANCELLED is missing!
    }
    return templates[state]

def should_notify(old_state: OrderState, new_state: OrderState) -> bool:
    """Decide if we should send notification."""
    # BUG: Hardcoded logic, doesn't handle all transitions
    if new_state == OrderState.SHIPPED:
        return True
    if new_state == OrderState.DELIVERED:
        return True
    return False
'''
    (project_dir / "src" / "notifications.py").write_text(notifications_py)

    # API handlers - yet another file with state knowledge
    api_py = '''"""API handlers."""
from .order import Order
from .states import OrderState

def handle_cancel_request(order: Order) -> dict:
    """Handle cancellation request."""
    # BUG: Checks state directly instead of using can_cancel()
    if order.state == OrderState.PENDING:
        order.transition(OrderState.CANCELLED)
        return {"success": True}
    return {"success": False, "error": "Cannot cancel"}

def get_order_status_display(order: Order) -> str:
    """Get human-readable status."""
    # BUG: Incomplete mapping
    display = {
        "pending": "Processing",
        "confirmed": "Confirmed",
        "shipped": "Shipped",
    }
    return display.get(order.state.value, "Unknown")
'''
    (project_dir / "src" / "api.py").write_text(api_py)

    return project_dir


TASK_1_PROMPT = """
There's a bug report: "Orders that are CONFIRMED can be cancelled via the API,
but the cancel button doesn't work. Also, cancelled orders show 'Unknown' status."

Fix these bugs in the codebase. The files are in src/:
- states.py: State definitions and valid transitions
- order.py: Order model
- notifications.py: Email notifications
- api.py: API handlers

Make sure your fix is consistent across all files.
"""


async def run_worker(project_dir: Path, task: str, max_turns: int = 25) -> dict:
    """Run worker and collect results."""
    result_text = ""
    tool_calls = []

    async for message in query(
        prompt=task,
        options=ClaudeCodeOptions(
            system_prompt="You are a senior developer fixing bugs. Be thorough.",
            allowed_tools=["Read", "Write", "Edit", "Glob", "Grep"],
            permission_mode="acceptEdits",
            max_turns=max_turns,
            cwd=str(project_dir),
        ),
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    result_text += block.text
                if hasattr(block, "name"):
                    tool_calls.append(block.name)

    # Read all modified files
    files = {}
    for f in (project_dir / "src").glob("*.py"):
        files[f.name] = f.read_text()

    return {
        "response": result_text,
        "tool_calls": tool_calls,
        "files": files,
    }


def evaluate_state_machine_fix(files: dict) -> dict:
    """Check if bugs are properly fixed with cross-file consistency."""
    issues = []
    score = 0
    max_score = 6

    api_py = files.get("api.py", "")
    order_py = files.get("order.py", "")
    notifications_py = files.get("notifications.py", "")

    # Check 1: API should use can_cancel() or check TRANSITIONS
    if "can_cancel()" in api_py or "TRANSITIONS" in api_py:
        score += 1
    else:
        issues.append("api.py: Still uses hardcoded state check instead of can_cancel()")

    # Check 2: API should handle CONFIRMED state for cancel
    if "CONFIRMED" in api_py or "can_cancel" in api_py:
        score += 1
    else:
        issues.append("api.py: Doesn't handle CONFIRMED cancellation")

    # Check 3: get_order_status_display should have all states
    if "delivered" in api_py.lower() and "cancelled" in api_py.lower():
        score += 1
    else:
        issues.append("api.py: Status display missing delivered/cancelled")

    # Check 4: notifications.py should have CANCELLED template
    if "CANCELLED" in notifications_py:
        score += 1
    else:
        issues.append("notifications.py: Missing CANCELLED template")

    # Check 5: notifications should_notify should be more complete
    if "CANCELLED" in notifications_py and "should_notify" in notifications_py:
        # Check if CANCELLED is considered in should_notify
        lines = notifications_py.split('\n')
        in_should_notify = False
        cancelled_handled = False
        for line in lines:
            if "def should_notify" in line:
                in_should_notify = True
            if in_should_notify and "CANCELLED" in line:
                cancelled_handled = True
        if cancelled_handled:
            score += 1
        else:
            issues.append("notifications.py: should_notify doesn't handle CANCELLED")
    else:
        issues.append("notifications.py: should_notify incomplete")

    # Check 6: Cross-file consistency - all files should use TRANSITIONS or can_cancel
    consistency = 0
    if "TRANSITIONS" in api_py or "can_cancel" in api_py:
        consistency += 1
    if "TRANSITIONS" in order_py:
        consistency += 1
    if consistency >= 1:
        score += 1
    else:
        issues.append("Cross-file: Inconsistent state checking logic")

    return {
        "score": score,
        "max_score": max_score,
        "percentage": score / max_score * 100,
        "issues": issues,
    }


async def run_experiment():
    console.print("\n[bold cyan]Experiment: Hard Tasks (State Machine Consistency)[/bold cyan]\n")
    console.print("This tests cross-file refactoring with implicit constraints.")
    console.print("The bug spans multiple files that all 'know' about states.\n")

    base_dir = Path("/tmp/ralph_exp_hard")
    base_dir.mkdir(exist_ok=True)

    results = []
    runs = 2

    for run_idx in range(runs):
        console.print(f"[bold]Run {run_idx + 1}/{runs}[/bold]")

        project_dir = setup_state_machine_project(base_dir, f"run_{run_idx}")
        result = await run_worker(project_dir, TASK_1_PROMPT)
        evaluation = evaluate_state_machine_fix(result["files"])
        results.append(evaluation)

        console.print(f"  Score: {evaluation['score']}/{evaluation['max_score']} ({evaluation['percentage']:.0f}%)")
        for issue in evaluation["issues"][:3]:
            console.print(f"  [red]x {issue}[/red]")
        console.print()

    # Summary
    avg_score = sum(r["percentage"] for r in results) / len(results)
    console.print(f"\n[bold]Average: {avg_score:.0f}%[/bold]")

    if avg_score < 70:
        console.print("[yellow]As expected, cross-file consistency is challenging![/yellow]")
    elif avg_score < 90:
        console.print("[blue]Partial success - some issues missed[/blue]")
    else:
        console.print("[green]Surprisingly good! All issues found.[/green]")

    # Show what a perfect fix would look like
    console.print("\n[dim]Expected fixes:[/dim]")
    console.print("[dim]1. api.py: Use order.can_cancel() instead of hardcoded check[/dim]")
    console.print("[dim]2. api.py: Add 'delivered' and 'cancelled' to status display[/dim]")
    console.print("[dim]3. notifications.py: Add CANCELLED to templates[/dim]")
    console.print("[dim]4. notifications.py: Handle CANCELLED in should_notify[/dim]")


if __name__ == "__main__":
    asyncio.run(run_experiment())
