"""
Evaluator agent: evaluates quality metrics for task outputs.

Three metric types:
- HARD: Must pass (tests, type check, builds)
- SOFT: Measurable, optimizable (performance, coverage)
- SUBJECTIVE: AI-evaluated quality (code quality, UI, API design)
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from claude_code_sdk import AssistantMessage, ClaudeCodeOptions, query
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .pool import read_goal, read_task
from .worker import format_tool_line

console = Console()


# -----------------------------------------------------------------------------
# Data Types
# -----------------------------------------------------------------------------

class MetricType(Enum):
    """Type of metric - determines how it's evaluated."""
    HARD = "hard"           # Must pass, binary (tests, builds)
    SOFT = "soft"           # Measurable, has target value (performance)
    SUBJECTIVE = "subjective"  # AI-evaluated quality


class AutomationLevel(Enum):
    """How the metric is evaluated."""
    AUTO = "auto"           # Fully automated (tests, benchmarks)
    MANUAL = "manual"       # Requires human testing
    HYBRID = "hybrid"       # Proxy metric auto, final needs human


@dataclass
class Metric:
    """A single metric to evaluate."""
    name: str
    type: MetricType
    description: str
    # For HARD: pass/fail
    # For SOFT: numeric with target
    target: Optional[str] = None  # e.g., ">= 95%", "<= 100ms"
    # How to measure (command, file pattern, or "ai")
    measure_by: str = "ai"
    # Automation level
    automation: AutomationLevel = AutomationLevel.AUTO
    # Proxy metric for hybrid/manual types
    proxy_metric: Optional[str] = None


@dataclass
class MetricResult:
    """Result of evaluating a single metric."""
    metric: Metric
    passed: bool
    value: Optional[str] = None  # Actual measured value
    score: Optional[float] = None  # 0-100 for subjective
    reason: str = ""
    pending_manual: bool = False  # True if awaiting user testing
    proxy_score: Optional[float] = None  # Proxy score for hybrid metrics
    proxy_notes: str = ""  # Notes about proxy evaluation


@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    task_id: str
    overall_passed: bool
    overall_score: float  # 0-100
    metrics: list[MetricResult] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    needs_user_testing: bool = False  # True if manual metrics need testing

    # Autonomous judgment fields
    attempt_number: int = 1           # Which attempt this is
    previous_scores: list[float] = field(default_factory=list)  # Scores from previous attempts
    should_pivot: bool = False        # Agent's recommendation to pivot
    pivot_reason: str = ""            # Why pivoting is recommended

    def summary(self) -> str:
        """Generate a summary string."""
        passed = sum(1 for m in self.metrics if m.passed)
        pending = sum(1 for m in self.metrics if m.pending_manual)
        total = len(self.metrics)
        summary = f"{passed}/{total} metrics passed, score: {self.overall_score:.0f}/100"
        if pending:
            summary += f" ({pending} pending user testing)"
        if self.should_pivot:
            summary += f" [PIVOT RECOMMENDED: {self.pivot_reason}]"
        return summary

    def get_pending_manual_metrics(self) -> list[MetricResult]:
        """Get metrics that need user testing."""
        return [m for m in self.metrics if m.pending_manual]

    def get_proxy_scores(self) -> dict[str, float]:
        """Get proxy scores for all hybrid metrics."""
        return {
            m.metric.name: m.proxy_score
            for m in self.metrics
            if m.proxy_score is not None
        }

    def get_auto_scores(self) -> dict[str, float]:
        """Get scores for all auto-evaluated metrics."""
        return {
            m.metric.name: m.score
            for m in self.metrics
            if m.score is not None and not m.pending_manual
        }

    def is_improving(self) -> bool:
        """Check if scores are improving across attempts."""
        if len(self.previous_scores) < 2:
            return True  # Not enough data
        # Check if last 2 scores show improvement
        return self.previous_scores[-1] > self.previous_scores[-2]

    def get_score_trend(self) -> str:
        """Get the trend of scores across attempts."""
        if not self.previous_scores:
            return "first_attempt"
        if len(self.previous_scores) == 1:
            if self.overall_score > self.previous_scores[0]:
                return "improving"
            elif self.overall_score < self.previous_scores[0]:
                return "declining"
            return "stable"

        # Check trend across all attempts
        scores = self.previous_scores + [self.overall_score]
        improvements = sum(1 for i in range(1, len(scores)) if scores[i] > scores[i-1])
        declines = sum(1 for i in range(1, len(scores)) if scores[i] < scores[i-1])

        if improvements > declines:
            return "improving"
        elif declines > improvements:
            return "declining"
        return "stable"


# -----------------------------------------------------------------------------
# Default Metric Templates
# -----------------------------------------------------------------------------

DEFAULT_METRICS = {
    "algorithm": [
        Metric("tests_pass", MetricType.HARD, "All tests pass", measure_by="bash:pytest"),
        Metric("no_errors", MetricType.HARD, "No runtime errors", measure_by="bash:python -c 'import main'"),
        Metric("accuracy", MetricType.SOFT, "Algorithm accuracy", target=">= 90%", measure_by="ai"),
        Metric("code_quality", MetricType.SUBJECTIVE, "Code readability and structure", measure_by="ai"),
    ],
    "web": [
        Metric("builds", MetricType.HARD, "Project builds successfully", measure_by="bash:npm run build"),
        Metric("no_console_errors", MetricType.HARD, "No console errors on load", measure_by="ai"),
        Metric("responsive", MetricType.SUBJECTIVE, "Works on mobile and desktop", measure_by="ai"),
        Metric("code_quality", MetricType.SUBJECTIVE, "Code readability and structure", measure_by="ai"),
    ],
    "api": [
        Metric("tests_pass", MetricType.HARD, "All tests pass", measure_by="bash:pytest"),
        Metric("type_check", MetricType.HARD, "Type check passes", measure_by="bash:mypy ."),
        Metric("api_design", MetricType.SUBJECTIVE, "RESTful, consistent naming, proper errors", measure_by="ai"),
        Metric("code_quality", MetricType.SUBJECTIVE, "Code readability and structure", measure_by="ai"),
    ],
    "general": [
        Metric("no_errors", MetricType.HARD, "Code runs without errors", measure_by="ai"),
        Metric("requirements_met", MetricType.HARD, "All requirements satisfied", measure_by="ai"),
        Metric("code_quality", MetricType.SUBJECTIVE, "Code readability and structure", measure_by="ai"),
    ],
}


# -----------------------------------------------------------------------------
# Evaluator Prompts
# -----------------------------------------------------------------------------

EVALUATOR_SYSTEM_PROMPT = """You are a quality evaluator for software tasks.

Your job is to evaluate the quality of completed work against specific metrics.

## Evaluation Process

1. Read the goal and task details
2. For each metric, evaluate whether it passes and provide a score
3. Identify specific issues with file:line references when possible
4. Provide actionable suggestions for improvement

## Important Guidelines

- Be objective and specific
- Reference actual code/files when pointing out issues
- Give scores based on real quality, not just "it works"
- Suggestions should be concrete and actionable

## Scoring Guide

For SUBJECTIVE metrics (0-100):
- 90-100: Excellent, production-ready
- 80-89: Good, minor improvements possible
- 70-79: Acceptable, some issues to address
- 60-69: Below standard, needs work
- <60: Poor, significant problems

<cross_agent_communication>
## Cross-Agent Communication

When you recommend pivoting, you MUST also write this recommendation to pool.md so the Planner can see it.

### Pivot Conditions (自动触发)
Recommend pivot when ANY of these conditions are met:
1. **Too many attempts**: attempt_number >= 3 AND average improvement < 5 points per attempt
2. **Declining scores**: 3 consecutive scores show decline (e.g., 40 → 35 → 30)
3. **Stuck at same score**: Last 3 scores within 3 points of each other
4. **Hard constraint keeps failing**: Same hard metric (tests, builds) fails 2+ times
5. **Very low score after multiple attempts**: Score < 40 after 2+ attempts

### Action Required When Pivot Is Recommended
1. Read current `.ralph/pool.md` using Read tool (use relative path from cwd)
2. Use Edit tool to append to the Findings section in `.ralph/pool.md`:
   `- [PIVOT_RECOMMENDED] {task_id}: {reason}`
3. This ensures the Planner sees your recommendation in the next iteration

**IMPORTANT**: Use the path `.ralph/pool.md` (relative to cwd), NOT any other path like `~/.ralph/` or absolute paths.

### Why This Matters
The Planner only reads pool.md, not your evaluation output directly. Without writing to pool.md, your pivot recommendation will be lost.
</cross_agent_communication>

## Output Format

For each metric, output:
```
METRIC: <metric_name>
PASSED: <yes|no>
VALUE: <measured value if applicable>
SCORE: <0-100 for subjective metrics>
REASON: <explanation>
```

After all metrics:
```
ISSUES:
- <issue 1 with file:line if applicable>
- <issue 2>

SUGGESTIONS:
- <suggestion 1>
- <suggestion 2>

OVERALL_SCORE: <0-100>

PIVOT_RECOMMENDED: <yes|no>
PIVOT_REASON: <if yes, explain why based on conditions above>
```
"""


def build_evaluator_prompt(
    task_id: str,
    goal: str,
    task_detail: str,
    metrics: list[Metric],
    include_proxy: bool = False,
) -> str:
    """Build the evaluator prompt with metrics to evaluate."""
    metrics_lines = []
    for m in metrics:
        line = f"- **{m.name}** ({m.type.value}): {m.description}"
        if m.target:
            line += f" [Target: {m.target}]"
        if include_proxy and m.automation == AutomationLevel.HYBRID and m.proxy_metric:
            line += f"\n  - **Proxy**: {m.proxy_metric}"
        metrics_lines.append(line)

    metrics_desc = "\n".join(metrics_lines)

    proxy_instructions = ""
    if include_proxy:
        proxy_instructions = """
## Proxy Metric Evaluation

For HYBRID metrics with a proxy specified, evaluate the proxy metric and provide:
- PROXY_SCORE: 0-100 score for the proxy metric
- PROXY_NOTES: Brief explanation of the proxy evaluation

This proxy score will be used to pre-filter checkpoints before user testing.
"""

    return f"""Goal:
---
{goal}
---

Task ({task_id}) to evaluate:
---
{task_detail}
---

## Metrics to Evaluate

{metrics_desc}
{proxy_instructions}
Evaluate each metric. Read relevant files, run commands if needed, and provide your assessment.
"""


# -----------------------------------------------------------------------------
# Parsing
# -----------------------------------------------------------------------------

def parse_evaluator_output(text: str, metrics: list[Metric]) -> EvaluationResult:
    """Parse evaluator output into EvaluationResult."""
    results = []

    # Parse each metric result (extended pattern to include proxy scores)
    metric_pattern = r"METRIC:\s*(\w+)\s*\nPASSED:\s*(yes|no)\s*\n(?:VALUE:\s*(.+?)\s*\n)?(?:SCORE:\s*(\d+)\s*\n)?(?:PROXY_SCORE:\s*(\d+)\s*\n)?(?:PROXY_NOTES:\s*(.+?)\s*\n)?REASON:\s*(.+?)(?=\nMETRIC:|\nISSUES:|\nSUGGESTIONS:|\nOVERALL_SCORE:|$)"

    for match in re.finditer(metric_pattern, text, re.IGNORECASE | re.DOTALL):
        name = match.group(1)
        passed = match.group(2).lower() == "yes"
        value = match.group(3).strip() if match.group(3) else None
        score = float(match.group(4)) if match.group(4) else None
        proxy_score = float(match.group(5)) if match.group(5) else None
        proxy_notes = match.group(6).strip() if match.group(6) else ""
        reason = match.group(7).strip()

        # Find matching metric
        metric = next((m for m in metrics if m.name.lower() == name.lower()), None)
        if metric:
            results.append(MetricResult(
                metric=metric,
                passed=passed,
                value=value,
                score=score,
                reason=reason,
                proxy_score=proxy_score,
                proxy_notes=proxy_notes,
            ))

    # If the extended pattern didn't match, try the simpler pattern
    if not results:
        simple_pattern = r"METRIC:\s*(\w+)\s*\nPASSED:\s*(yes|no)\s*\n(?:VALUE:\s*(.+?)\s*\n)?(?:SCORE:\s*(\d+)\s*\n)?REASON:\s*(.+?)(?=\nMETRIC:|\nISSUES:|\nSUGGESTIONS:|\nOVERALL_SCORE:|$)"
        for match in re.finditer(simple_pattern, text, re.IGNORECASE | re.DOTALL):
            name = match.group(1)
            passed = match.group(2).lower() == "yes"
            value = match.group(3).strip() if match.group(3) else None
            score = float(match.group(4)) if match.group(4) else None
            reason = match.group(5).strip()

            metric = next((m for m in metrics if m.name.lower() == name.lower()), None)
            if metric:
                results.append(MetricResult(
                    metric=metric,
                    passed=passed,
                    value=value,
                    score=score,
                    reason=reason,
                ))

    # Parse issues
    issues = []
    issues_match = re.search(r"ISSUES:\s*\n((?:- .+\n?)+)", text, re.IGNORECASE)
    if issues_match:
        issues = [line.strip("- \n") for line in issues_match.group(1).split("\n") if line.strip().startswith("-")]

    # Parse suggestions
    suggestions = []
    suggestions_match = re.search(r"SUGGESTIONS:\s*\n((?:- .+\n?)+)", text, re.IGNORECASE)
    if suggestions_match:
        suggestions = [line.strip("- \n") for line in suggestions_match.group(1).split("\n") if line.strip().startswith("-")]

    # Parse overall score
    overall_match = re.search(r"OVERALL_SCORE:\s*(\d+)", text, re.IGNORECASE)
    overall_score = float(overall_match.group(1)) if overall_match else 0

    # Determine overall passed
    hard_metrics = [r for r in results if r.metric.type == MetricType.HARD]
    overall_passed = all(r.passed for r in hard_metrics) if hard_metrics else True

    return EvaluationResult(
        task_id="",  # Will be set by caller
        overall_passed=overall_passed,
        overall_score=overall_score,
        metrics=results,
        issues=issues,
        suggestions=suggestions,
    )


# -----------------------------------------------------------------------------
# Main Evaluate Function
# -----------------------------------------------------------------------------

async def evaluate(
    task_id: str,
    cwd: str = ".",
    metrics: Optional[list[Metric]] = None,
    task_type: str = "general",
    verbose: bool = False,
    evaluate_proxy: bool = True,
    previous_scores: Optional[list[float]] = None,
    attempt_number: int = 1,
    pivot_threshold: int = 3,
    min_improvement: float = 5.0,
) -> EvaluationResult:
    """
    Evaluate a task's output against quality metrics.

    Args:
        task_id: The task ID to evaluate
        cwd: Working directory
        metrics: Custom metrics to evaluate (uses defaults if None)
        task_type: Type of task for default metrics (algorithm, web, api, general)
        verbose: Show detailed output
        evaluate_proxy: If True, evaluate proxy metrics for HYBRID metrics
        previous_scores: List of scores from previous attempts (for trend analysis)
        attempt_number: Which attempt this is (1-indexed)
        pivot_threshold: Number of attempts before recommending pivot
        min_improvement: Minimum score improvement per attempt to not recommend pivot

    Returns:
        EvaluationResult with scores, suggestions, and pivot recommendation
    """
    # Use provided metrics or defaults
    if metrics is None:
        metrics = DEFAULT_METRICS.get(task_type, DEFAULT_METRICS["general"])

    # Separate metrics by automation level
    auto_metrics = [m for m in metrics if m.automation == AutomationLevel.AUTO]
    hybrid_metrics = [m for m in metrics if m.automation == AutomationLevel.HYBRID]
    manual_metrics = [m for m in metrics if m.automation == AutomationLevel.MANUAL]

    if hybrid_metrics or manual_metrics:
        console.print(f"[dim]Note: {len(hybrid_metrics) + len(manual_metrics)} metric(s) require user testing[/dim]")
        if hybrid_metrics and evaluate_proxy:
            console.print(f"[dim]      {len(hybrid_metrics)} will have proxy scores evaluated[/dim]")

    console.print(f"\n[bold cyan]Evaluating {task_id}...[/bold cyan]")

    # Read context
    goal = read_goal(cwd)
    task_detail = read_task(task_id, cwd)

    if not task_detail:
        return EvaluationResult(
            task_id=task_id,
            overall_passed=False,
            overall_score=0,
            issues=[f"Task file not found: tasks/{task_id}.md"],
        )

    # Determine which metrics to evaluate
    # Auto metrics + hybrid metrics (for proxy evaluation)
    metrics_to_eval = auto_metrics.copy()
    if evaluate_proxy:
        metrics_to_eval.extend(hybrid_metrics)

    # Build prompt
    prompt = build_evaluator_prompt(
        task_id=task_id,
        goal=goal,
        task_detail=task_detail,
        metrics=metrics_to_eval if metrics_to_eval else metrics,  # Fallback to all if empty
        include_proxy=evaluate_proxy and bool(hybrid_metrics),
    )

    # Run evaluator
    result_text = ""
    tool_count = 0

    async for message in query(
        prompt=prompt,
        options=ClaudeCodeOptions(
            system_prompt=EVALUATOR_SYSTEM_PROMPT,
            allowed_tools=[
                "Read", "Write", "Edit", "Bash", "Glob", "Grep", "LSP",
                "WebFetch", "WebSearch",  # Research best practices for evaluation
            ],
            permission_mode="acceptEdits",
            max_turns=15,
            cwd=cwd,
        ),
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text") and block.text:
                    result_text += block.text
                    if verbose:
                        text = block.text.strip()
                        if text and len(text) > 20:
                            lines = text.split('\n')
                            first_line = lines[0][:80]
                            if len(lines[0]) > 80:
                                first_line += "..."
                            console.print(f"     [italic bright_black]📊 {first_line}[/italic bright_black]")

                if hasattr(block, "name") and hasattr(block, "input"):
                    tool_count += 1
                    if verbose:
                        tool_line = format_tool_line(block.name, block.input, cwd)
                        console.print(f"[bright_black][{tool_count:2d}][/bright_black] {tool_line}")

    # Parse result for evaluated metrics
    result = parse_evaluator_output(result_text, metrics_to_eval if metrics_to_eval else metrics)
    result.task_id = task_id

    # Mark hybrid metrics as pending manual (they have proxy scores but need user testing)
    for mr in result.metrics:
        if mr.metric.automation == AutomationLevel.HYBRID:
            mr.pending_manual = True

    # Add pending results for purely manual metrics (no proxy evaluation)
    for m in manual_metrics:
        result.metrics.append(MetricResult(
            metric=m,
            passed=False,  # Not yet evaluated
            reason=f"Pending user testing. Proxy: {m.proxy_metric or 'N/A'}",
            pending_manual=True,
        ))

    # Update flags
    result.needs_user_testing = len(manual_metrics) > 0 or len(hybrid_metrics) > 0

    # Set attempt tracking
    result.attempt_number = attempt_number
    result.previous_scores = previous_scores or []

    # Autonomous judgment: should we recommend pivoting?
    result.should_pivot, result.pivot_reason = _assess_pivot_recommendation(
        result=result,
        previous_scores=previous_scores or [],
        attempt_number=attempt_number,
        pivot_threshold=pivot_threshold,
        min_improvement=min_improvement,
    )

    # Display results
    display_evaluation_result(result)

    return result


def _assess_pivot_recommendation(
    result: EvaluationResult,
    previous_scores: list[float],
    attempt_number: int,
    pivot_threshold: int,
    min_improvement: float,
) -> tuple[bool, str]:
    """
    Assess whether the agent should recommend pivoting.

    This is the agent's autonomous judgment - it can recommend pivoting
    without user confirmation.

    Returns:
        (should_pivot, reason)
    """
    # Case 1: Too many attempts
    if attempt_number >= pivot_threshold:
        # Check if we're still making progress
        if previous_scores:
            avg_improvement = (result.overall_score - previous_scores[0]) / attempt_number
            if avg_improvement < min_improvement:
                return True, f"尝试 {attempt_number} 次，平均每次改进 {avg_improvement:.1f} 分，低于阈值 {min_improvement}"

    # Case 2: Score is declining
    if len(previous_scores) >= 2:
        recent_scores = previous_scores[-2:] + [result.overall_score]
        if recent_scores[2] < recent_scores[1] < recent_scores[0]:
            return True, f"分数连续下降: {recent_scores[0]:.0f} → {recent_scores[1]:.0f} → {recent_scores[2]:.0f}"

    # Case 3: Stuck at same score
    if len(previous_scores) >= 2:
        recent_scores = previous_scores[-2:] + [result.overall_score]
        score_range = max(recent_scores) - min(recent_scores)
        if score_range < 3:  # Less than 3 points difference
            return True, f"分数停滞在 {result.overall_score:.0f} 左右，连续 3 次无显著改进"

    # Case 4: Hard constraint keeps failing
    hard_failures = [m for m in result.metrics if m.metric.type == MetricType.HARD and not m.passed]
    if hard_failures and attempt_number >= 2:
        failing_names = [m.metric.name for m in hard_failures]
        return True, f"硬性指标持续失败: {', '.join(failing_names)}"

    # Case 5: Very low score after multiple attempts
    if result.overall_score < 40 and attempt_number >= 2:
        return True, f"尝试 {attempt_number} 次后分数仍只有 {result.overall_score:.0f}，可能需要换方向"

    return False, ""


def get_attempt_history(task_id: str, cwd: str = ".") -> tuple[int, list[float]]:
    """
    Get attempt history from task file.

    Returns:
        (attempt_number, previous_scores)
    """
    task_content = read_task(task_id, cwd)
    if not task_content:
        return 1, []

    # Parse attempt history from task file
    # Look for "## Quality Evaluation" sections with scores
    import re
    scores = re.findall(r"\*\*Score\*\*:\s*(\d+(?:\.\d+)?)/100", task_content)

    previous_scores = [float(s) for s in scores]
    attempt_number = len(previous_scores) + 1

    return attempt_number, previous_scores


def display_evaluation_result(result: EvaluationResult):
    """Display evaluation result in a nice format."""
    # Create metrics table
    table = Table(title=f"Evaluation: {result.task_id}", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Type", style="dim")
    table.add_column("Status", justify="center")
    table.add_column("Score", justify="right")
    table.add_column("Proxy", justify="right", style="dim")
    table.add_column("Reason", style="dim", max_width=35)

    for mr in result.metrics:
        if mr.pending_manual:
            if mr.proxy_score is not None:
                status = "[yellow]⏳ PROXY[/yellow]"
            else:
                status = "[yellow]⏳ PENDING[/yellow]"
        elif mr.passed:
            status = "[green]✓ PASS[/green]"
        else:
            status = "[red]✗ FAIL[/red]"

        score = f"{mr.score:.0f}" if mr.score is not None else "-"
        proxy = f"{mr.proxy_score:.0f}" if mr.proxy_score is not None else "-"

        table.add_row(
            mr.metric.name,
            mr.metric.type.value,
            status,
            score,
            proxy,
            mr.reason[:35] + "..." if len(mr.reason) > 35 else mr.reason,
        )

    console.print(table)

    # Overall result
    status_color = "green" if result.overall_passed else "red"
    console.print(
        Panel(
            f"[bold {status_color}]{'PASSED' if result.overall_passed else 'FAILED'}[/bold {status_color}]  "
            f"Score: [bold]{result.overall_score:.0f}/100[/bold]",
            title="Overall",
            border_style=status_color,
        )
    )

    # Issues
    if result.issues:
        console.print("\n[bold red]Issues:[/bold red]")
        for issue in result.issues:
            console.print(f"  [red]•[/red] {issue}")

    # Suggestions
    if result.suggestions:
        console.print("\n[bold yellow]Suggestions:[/bold yellow]")
        for suggestion in result.suggestions:
            console.print(f"  [yellow]•[/yellow] {suggestion}")
