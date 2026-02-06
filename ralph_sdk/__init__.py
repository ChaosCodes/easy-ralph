"""Ralph SDK - Autonomous agent with dynamic task pool."""

from .clarifier import (
    clarify_requirements,
    clarify_requirements_v2,
    explore_and_propose,
    quick_clarify,
)
from .evaluator import (
    AutomationLevel as EvalAutomationLevel,
    EvaluationResult,
    Metric,
    MetricType,
    evaluate,
    get_attempt_history,
)
from .metrics import (
    AutomationLevel,
    EvalConfig,
    MetricDefinition,
    MetricsConfig,
    TaskCategory,
    detect_category,
    get_default_metrics,
)
from .notification import (
    Notification,
    NotificationType,
    notify,
    notify_checkpoint,
    notify_complete,
    notify_decision,
    notify_info,
    notify_pivot,
    notify_progress,
    notify_warning,
)
from .orchestrator import resume, run
from .planner import Action, PlannerDecision
from .pool import (
    Checkpoint,
    ProxyScore,
    add_checkpoint,
    create_checkpoint,
    get_best_checkpoint,
    get_pending_checkpoints,
    get_task_checkpoints,
    is_waiting_for_user,
    parse_feedback,
    read_checkpoint_manifest,
    update_checkpoint,
)
from .reviewer import ReviewResult, Verdict
from .worker import WorkerResult

__version__ = "0.3.0"  # Version bump for architecture improvements
__all__ = [
    # Core
    "run",
    "resume",
    # Clarifier
    "clarify_requirements",
    "clarify_requirements_v2",
    "explore_and_propose",
    "quick_clarify",
    # Planner
    "Action",
    "PlannerDecision",
    # Worker
    "WorkerResult",
    # Reviewer
    "Verdict",
    "ReviewResult",
    # Evaluator
    "EvaluationResult",
    "MetricType",
    "Metric",
    "EvalAutomationLevel",
    "evaluate",
    "get_attempt_history",
    # Metrics
    "MetricsConfig",
    "MetricDefinition",
    "TaskCategory",
    "AutomationLevel",
    "EvalConfig",
    "get_default_metrics",
    "detect_category",
    # Notification
    "Notification",
    "NotificationType",
    "notify",
    "notify_info",
    "notify_progress",
    "notify_warning",
    "notify_pivot",
    "notify_checkpoint",
    "notify_decision",
    "notify_complete",
    # Checkpoint & Feedback
    "Checkpoint",
    "ProxyScore",
    "create_checkpoint",
    "add_checkpoint",
    "update_checkpoint",
    "get_task_checkpoints",
    "get_best_checkpoint",
    "get_pending_checkpoints",
    "is_waiting_for_user",
    "parse_feedback",
    "read_checkpoint_manifest",
]
