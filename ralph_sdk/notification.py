"""
Notification system for Ralph SDK.

Provides non-blocking notifications to users while agent continues working.
MVP: CLI output using rich console.
Future: macOS system notifications, webhook integrations, etc.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.style import Style

console = Console()


class NotificationType(Enum):
    """Types of notifications with different urgency levels."""
    INFO = "info"           # General information
    PROGRESS = "progress"   # Task progress update
    WARNING = "warning"     # Something needs attention
    PIVOT = "pivot"         # Agent is changing direction
    CHECKPOINT = "checkpoint"  # Checkpoint ready for testing
    DECISION = "decision"   # Agent made an autonomous decision
    COMPLETE = "complete"   # Task/goal completed


@dataclass
class Notification:
    """A notification to the user."""
    type: NotificationType
    title: str
    message: str
    task_id: Optional[str] = None
    timestamp: str = ""
    details: Optional[dict] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%H:%M:%S")


# Style configurations for different notification types
NOTIFICATION_STYLES = {
    NotificationType.INFO: {
        "icon": "ℹ️",
        "border_style": "blue",
        "title_style": "bold blue",
    },
    NotificationType.PROGRESS: {
        "icon": "⏳",
        "border_style": "cyan",
        "title_style": "bold cyan",
    },
    NotificationType.WARNING: {
        "icon": "⚠️",
        "border_style": "yellow",
        "title_style": "bold yellow",
    },
    NotificationType.PIVOT: {
        "icon": "🔄",
        "border_style": "magenta",
        "title_style": "bold magenta",
    },
    NotificationType.CHECKPOINT: {
        "icon": "📍",
        "border_style": "green",
        "title_style": "bold green",
    },
    NotificationType.DECISION: {
        "icon": "🤖",
        "border_style": "bright_cyan",
        "title_style": "bold bright_cyan",
    },
    NotificationType.COMPLETE: {
        "icon": "✅",
        "border_style": "bright_green",
        "title_style": "bold bright_green",
    },
}


def notify(notification: Notification) -> None:
    """
    Display a notification to the user.

    This is non-blocking - the notification is displayed and execution continues.
    """
    style = NOTIFICATION_STYLES.get(notification.type, NOTIFICATION_STYLES[NotificationType.INFO])

    # Build title with icon and timestamp
    title = f"{style['icon']} {notification.title}"
    if notification.task_id:
        title = f"{title} [{notification.task_id}]"

    # Build content
    content_lines = [notification.message]

    if notification.details:
        content_lines.append("")
        for key, value in notification.details.items():
            content_lines.append(f"[dim]{key}:[/dim] {value}")

    content = "\n".join(content_lines)

    # Display as panel
    console.print()
    console.print(
        Panel(
            content,
            title=f"[{style['title_style']}]{title}[/{style['title_style']}]",
            subtitle=f"[dim]{notification.timestamp}[/dim]",
            border_style=style["border_style"],
            padding=(0, 1),
        )
    )


def notify_info(title: str, message: str, task_id: str = None, **details) -> None:
    """Send an info notification."""
    notify(Notification(
        type=NotificationType.INFO,
        title=title,
        message=message,
        task_id=task_id,
        details=details if details else None,
    ))


def notify_progress(title: str, message: str, task_id: str = None, **details) -> None:
    """Send a progress notification."""
    notify(Notification(
        type=NotificationType.PROGRESS,
        title=title,
        message=message,
        task_id=task_id,
        details=details if details else None,
    ))


def notify_warning(title: str, message: str, task_id: str = None, **details) -> None:
    """Send a warning notification."""
    notify(Notification(
        type=NotificationType.WARNING,
        title=title,
        message=message,
        task_id=task_id,
        details=details if details else None,
    ))


def notify_pivot(
    task_id: str,
    reason: str,
    from_approach: str,
    to_approach: str,
    trigger: str = "research",
) -> None:
    """
    Notify user that agent is pivoting to a different approach.

    Args:
        task_id: The task being pivoted
        reason: Why the pivot is happening
        from_approach: What approach is being abandoned
        to_approach: What new approach is being tried
        trigger: What triggered the pivot (research, wait, iteration)
    """
    trigger_descriptions = {
        "research": "深度研究确认不可行",
        "wait": "等待期间探索替代方案",
        "iteration": "多次尝试效果不达预期",
    }

    notify(Notification(
        type=NotificationType.PIVOT,
        title="方向转换",
        message=reason,
        task_id=task_id,
        details={
            "触发条件": trigger_descriptions.get(trigger, trigger),
            "原方案": from_approach,
            "新方案": to_approach,
        },
    ))


def notify_checkpoint(
    checkpoint_id: str,
    task_id: str,
    proxy_score: float = None,
    description: str = "",
) -> None:
    """
    Notify user that a checkpoint is ready for testing.

    This is informational - agent continues working.
    """
    message = f"新 checkpoint 已创建: {checkpoint_id}"
    if description:
        message += f"\n{description}"

    details = {}
    if proxy_score is not None:
        details["代理分数"] = f"{proxy_score:.0f}/100"
    details["提示"] = "可随时测试，Agent 继续工作中"

    notify(Notification(
        type=NotificationType.CHECKPOINT,
        title="Checkpoint 就绪",
        message=message,
        task_id=task_id,
        details=details,
    ))


def notify_decision(
    decision: str,
    reason: str,
    task_id: str = None,
    alternatives_considered: list[str] = None,
) -> None:
    """
    Notify user that agent made an autonomous decision.

    Args:
        decision: What decision was made
        reason: Why this decision was made
        task_id: Related task (if any)
        alternatives_considered: Other options that were considered
    """
    details = {"决策": decision}
    if alternatives_considered:
        details["备选方案"] = ", ".join(alternatives_considered)

    notify(Notification(
        type=NotificationType.DECISION,
        title="自主决策",
        message=reason,
        task_id=task_id,
        details=details,
    ))


def notify_complete(
    title: str,
    message: str,
    task_id: str = None,
    score: float = None,
    **details,
) -> None:
    """Send a completion notification."""
    all_details = dict(details) if details else {}
    if score is not None:
        all_details["质量分数"] = f"{score:.0f}/100"

    notify(Notification(
        type=NotificationType.COMPLETE,
        title=title,
        message=message,
        task_id=task_id,
        details=all_details if all_details else None,
    ))


# -----------------------------------------------------------------------------
# Batch notifications (for displaying multiple checkpoints)
# -----------------------------------------------------------------------------

def notify_batch_checkpoints(
    checkpoints: list[dict],
    instructions: str = "",
) -> None:
    """
    Display a summary of multiple checkpoints ready for batch testing.

    Args:
        checkpoints: List of checkpoint info dicts
        instructions: Testing instructions
    """
    console.print()
    console.print(Panel(
        f"[bold]有 {len(checkpoints)} 个 checkpoint 待测试[/bold]\n\n"
        f"[dim]{instructions}[/dim]\n\n"
        "[cyan]Agent 继续在后台探索其他可能性...[/cyan]",
        title="[bold green]📦 批量测试就绪[/bold green]",
        border_style="green",
    ))

    # List checkpoints
    for cp in checkpoints:
        cp_id = cp.get("id", "unknown")
        proxy = cp.get("proxy_overall", 0)
        status = cp.get("status", "pending")
        console.print(f"  [cyan]•[/cyan] {cp_id} [dim](代理分数: {proxy:.0f})[/dim]")

    console.print()
    console.print("[dim]测试完成后运行 ralph resume 继续[/dim]")


# -----------------------------------------------------------------------------
# Progress bar for long operations
# -----------------------------------------------------------------------------

def create_progress_context(description: str):
    """
    Create a progress context for long-running operations.

    Usage:
        with create_progress_context("Exploring approaches..."):
            # do work
            pass
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn

    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    )


# -----------------------------------------------------------------------------
# Future: System notifications (macOS)
# -----------------------------------------------------------------------------

def _send_system_notification(title: str, message: str) -> bool:
    """
    Send a macOS system notification (placeholder for future implementation).

    Returns True if notification was sent successfully.
    """
    # TODO: Implement using osascript or py-notifier
    # For now, this is a no-op placeholder
    return False
