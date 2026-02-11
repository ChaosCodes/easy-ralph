"""
Evaluator agent: evaluates quality metrics for task outputs.

Three metric types:
- HARD: Must pass (tests, type check, builds)
- SOFT: Measurable, optimizable (performance, coverage)
- SUBJECTIVE: AI-evaluated quality (code quality, UI, API design)
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from claude_agent_sdk import ClaudeAgentOptions
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .logger import log_tool_call, stream_query
from .pool import AUDITS_DIR, RALPH_DIR, read_goal, read_task, write_task
from .prompts import EVALUATOR_ADVERSARIAL_SECTION
from .utils import extract_json

console = Console()

EVALUATOR_OUTPUT_SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "metrics": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "passed": {"type": "boolean"},
                        "value": {"type": ["string", "null"]},
                        "score": {"type": ["number", "null"]},
                        "proxy_score": {"type": ["number", "null"]},
                        "proxy_notes": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["name", "passed", "reason"],
                },
            },
            "issues": {"type": "array", "items": {"type": "string"}},
            "suggestions": {"type": "array", "items": {"type": "string"}},
            "overall_score": {"type": "number"},
            "pivot_recommended": {"type": "boolean"},
            "pivot_reason": {"type": "string"},
            "only_cosmetic": {"type": "boolean"},
            "issue_completeness": {"type": "string"},
        },
        "required": ["metrics", "overall_score"],
    },
}


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

    # Execution stats
    result_stats: object = None  # ResultMessage from SDK

    # How many metrics the agent attempted to evaluate (from its JSON output)
    metrics_attempted: int = 0

    # Autonomous judgment fields
    attempt_number: int = 1           # Which attempt this is
    previous_scores: list[float] = field(default_factory=list)  # Scores from previous attempts
    should_pivot: bool = False        # Agent's recommendation to pivot
    pivot_reason: str = ""            # Why pivoting is recommended

    # Infrastructure failure detection
    is_infra_failure: bool = False  # True when eval failed due to turn limit / parse failure (not real score)

    # Adversarial testing fields
    adversarial_findings: str = ""         # Content of findings file
    adversarial_findings_path: str = ""    # Path to findings file

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
2. If the goal contains a "## Success Metrics" section, use those metrics as your evaluation criteria
   - Hard Constraints → must pass (PASSED: yes/no)
   - Performance Targets → measure against targets (VALUE + SCORE)
   - Quality Criteria → AI-evaluate (SCORE 0-100)
3. If no Success Metrics section exists, use the metrics provided in the prompt
4. For each metric, evaluate whether it passes and provide a score
5. Identify specific issues with file:line references when possible
6. Provide actionable suggestions for improvement

## Important Guidelines

- Be objective and specific
- Reference actual code/files when pointing out issues
- Give scores based on real quality, not just "it works"
- Suggestions should be concrete and actionable
- When running tests or scripts, use `python3` (not `python`). If the project has a `src/` layout, set `PYTHONPATH=src` (e.g. `PYTHONPATH=src python3 -m pytest tests/ -v`)

## Scoring Guide

For SUBJECTIVE metrics (0-100):
- 90-100: Excellent, production-ready
- 80-89: Good, minor improvements possible
- 70-79: Acceptable, some issues to address
- 60-69: Below standard, needs work
- <60: Poor, significant problems

## Anti-Anchoring Rules

- Score based ONLY on current code quality. Previous scores are irrelevant.
- If all identified issues are cosmetic/style-only (docstrings, naming, type hints),
  the score MUST be >= 95. Cosmetic issues alone cannot deduct more than 5 points.
- Distinguish between:
  - FUNCTIONAL issues (bugs, missing features, wrong behavior): up to -30 points
  - STRUCTURAL issues (DRY violations, missing error handling, bad architecture): up to -20 points
  - COSMETIC issues (style, naming, docstrings, type annotations): up to -5 points
- When listing issues, tag each as [FUNCTIONAL], [STRUCTURAL], or [COSMETIC]

## Pivot Assessment

When the prompt includes an "Attempt History" section, you MUST assess whether the current approach should pivot:
- Consider the score trend, number of attempts, and whether hard metrics keep failing
- Output your judgment in PIVOT_RECOMMENDED and PIVOT_REASON fields
- You do NOT need to write any files — the system handles propagation automatically

## Temporal Verification Check (from Reviewer)

As part of your evaluation, check for unverified time-sensitive information:
1. Does the implementation use external APIs/libraries with specific versions but no verification source?
2. Are there API usage patterns without `[已搜索验证]` annotation?
3. Are there assumptions about "current best practices" that may be outdated?
4. Are there possibly deprecated patterns being used?

If any items fail verification, include them as [FUNCTIONAL] issues (unverified dependencies
can cause runtime failures). Tag with "NEEDS_VERIFICATION:" prefix.

## Adversarial Testing

After structural evaluation, you may be asked to perform adversarial testing.
If the prompt includes an "Adversarial Verification Phase" section, follow those instructions
to write adversarial tests and findings. This is separate from and in addition to your JSON evaluation output.

## Output Format

Output your evaluation as a JSON object:

```json
{
  "metrics": [
    {
      "name": "<metric_name>",
      "passed": true/false,
      "value": "<measured value if applicable>",
      "score": <0-100 for subjective metrics, null otherwise>,
      "proxy_score": <0-100 if applicable, null otherwise>,
      "proxy_notes": "<if applicable>",
      "reason": "<explanation>"
    }
  ],
  "issues": ["[FUNCTIONAL] <issue 1 with file:line>", "[STRUCTURAL] <issue 2>", "[COSMETIC] <issue 3>"],
  "suggestions": ["<suggestion 1>", "<suggestion 2>"],
  "overall_score": <0-100>,
  "pivot_recommended": true/false,
  "pivot_reason": "<if true, explain why>",
  "only_cosmetic": true/false,
  "issue_completeness": "exhaustive"
}
```

"""


def _describe_trend(scores: list[float]) -> str:
    """Describe the trend of a list of scores."""
    if len(scores) < 2:
        return "insufficient data"
    improvements = sum(1 for i in range(1, len(scores)) if scores[i] > scores[i-1])
    declines = sum(1 for i in range(1, len(scores)) if scores[i] < scores[i-1])
    if improvements > declines:
        return "improving"
    elif declines > improvements:
        return "declining"
    return "stable"


def build_evaluator_prompt(
    task_id: str,
    goal: str,
    task_detail: str,
    metrics: list[Metric],
    include_proxy: bool = False,
    attempt_number: int = 1,
    previous_scores: Optional[list[float]] = None,
    audits_dir: str = "",
    previous_adversarial_responses: Optional[str] = None,
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

    history_section = ""
    if attempt_number > 1 and previous_scores:
        history_section = f"""
## Attempt History

This is attempt **#{attempt_number}** for this task.
The worker has made {attempt_number - 1} previous attempts at this task.

IMPORTANT: Evaluate the code AS-IS based solely on its current quality.
Do NOT consider how many attempts have been made when assigning scores.
A perfect implementation should score 95+ regardless of attempt number.

Based on the code quality, assess whether the current approach should continue or pivot.
Output your judgment in PIVOT_RECOMMENDED and PIVOT_REASON fields.
"""

    adversarial_section = ""
    if audits_dir:
        adversarial_section = EVALUATOR_ADVERSARIAL_SECTION.format(
            audits_dir=audits_dir,
            task_id=task_id,
            attempt=attempt_number,
        )

    adversarial_response_section = ""
    if previous_adversarial_responses:
        adversarial_response_section = f"""
## Worker's Previous Adversarial Response

The Worker investigated the findings from the previous round and provided this response.
Evaluate whether their rebuttals are valid. Focus your new adversarial tests on areas NOT yet covered.

---
{previous_adversarial_responses}
---
"""

    metric_count = len(metrics)
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
{proxy_instructions}{history_section}
Evaluate each metric. Read relevant files, run commands if needed, and provide your assessment.

**IMPORTANT**: Your JSON output MUST contain exactly {metric_count} metrics entries — one for each metric listed above. Do not skip any.
{adversarial_section}{adversarial_response_section}"""


# -----------------------------------------------------------------------------
# Parsing
# -----------------------------------------------------------------------------

def _parse_evaluator_json(obj: dict, metrics: list[Metric]) -> EvaluationResult:
    """Parse a JSON object into an EvaluationResult.

    When the agent outputs metric names not in the provided metrics list
    (e.g. it evaluated goal.md's custom metrics instead of DEFAULT_METRICS),
    create ad-hoc Metric objects so results are not silently discarded.
    """
    agent_metrics = obj.get("metrics", [])
    results = []
    for m_data in agent_metrics:
        name = m_data.get("name", "")
        metric = next((m for m in metrics if m.name.lower() == name.lower()), None)
        if not metric:
            # Create ad-hoc Metric from agent output
            has_score = m_data.get("score") is not None
            metric = Metric(
                name=name,
                type=MetricType.SUBJECTIVE if has_score else MetricType.HARD,
                description=m_data.get("reason", name),
            )
        results.append(MetricResult(
            metric=metric,
            passed=bool(m_data.get("passed", False)),
            value=m_data.get("value"),
            score=float(m_data["score"]) if m_data.get("score") is not None else None,
            reason=m_data.get("reason", ""),
            proxy_score=float(m_data["proxy_score"]) if m_data.get("proxy_score") is not None else None,
            proxy_notes=m_data.get("proxy_notes", ""),
        ))

    issues = obj.get("issues", [])
    suggestions = obj.get("suggestions", [])
    overall_score = float(obj.get("overall_score", 0))

    should_pivot = obj.get("pivot_recommended", False)
    pivot_reason = obj.get("pivot_reason", "")

    hard_metrics = [r for r in results if r.metric.type == MetricType.HARD]
    overall_passed = all(r.passed for r in hard_metrics) if hard_metrics else True

    result = EvaluationResult(
        task_id="",
        overall_passed=overall_passed,
        overall_score=overall_score,
        metrics=results,
        issues=issues,
        suggestions=suggestions,
    )
    result.metrics_attempted = len(agent_metrics)
    result.should_pivot = bool(should_pivot)
    result.pivot_reason = pivot_reason

    return result


def _validate_score_consistency(result: EvaluationResult) -> None:
    """Log warning if score seems inconsistent with issue severity."""
    if not result.issues:
        return

    cosmetic_keywords = ["docstring", "type hint", "naming", "style", "annotation",
                         "comment", "logging", "magic number", "cosmetic"]

    cosmetic_count = sum(
        1 for issue in result.issues
        if any(kw.lower() in issue.lower() for kw in cosmetic_keywords)
    )

    total = len(result.issues)
    if cosmetic_count == total and result.overall_score < 95:
        console.print(
            f"[yellow]\u26a0 All {total} issues are cosmetic but score is {result.overall_score:.0f} "
            f"(expected >= 95 for cosmetic-only issues)[/yellow]"
        )


def _detect_cosmetic_only(issues: list[str]) -> bool:
    """Detect if all issues in a list are cosmetic-only.

    Uses [COSMETIC] tags from evaluator output. If any issue has [FUNCTIONAL]
    or [STRUCTURAL] tag, returns False. If any issue has no tag at all,
    conservatively returns False (assumes non-cosmetic).

    Returns True only when ALL issues are tagged [COSMETIC].
    """
    if not issues:
        return False
    for issue in issues:
        upper = issue.upper()
        if "[FUNCTIONAL]" in upper or "[STRUCTURAL]" in upper:
            return False
        if "[COSMETIC]" not in upper:
            return False  # No tag → conservative assumption: not cosmetic
    return True


def parse_evaluator_output(structured_output: dict | None, text: str, metrics: list[Metric]) -> EvaluationResult:
    """Parse evaluator output. Prefers structured_output, falls back to JSON extraction.

    When neither structured_output nor text contains valid JSON with metrics,
    this is an infrastructure failure (e.g. turn limit hit before agent produced output),
    NOT a real score of 0.
    """
    obj = structured_output
    if not obj:
        obj = extract_json(text)
    if obj and "metrics" in obj:
        return _parse_evaluator_json(obj, metrics)
    # Infrastructure failure: agent never produced evaluation JSON
    result = EvaluationResult(
        task_id="", overall_passed=False, overall_score=0,
        issues=["Failed to parse evaluator output"],
    )
    result.is_infra_failure = True
    return result


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
    thinking_budget: int | None = None,
    previous_adversarial_responses: Optional[str] = None,
    skip_adversarial: bool = False,
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

    # Set up adversarial audits directory
    audits_path = Path(cwd) / AUDITS_DIR
    audits_path.mkdir(parents=True, exist_ok=True)
    audits_dir = str(audits_path)

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

    # When skip_adversarial, don't pass audits_dir to prompt (Exp 4)
    effective_audits_dir = "" if skip_adversarial else audits_dir
    effective_prev_adv = None if skip_adversarial else previous_adversarial_responses

    # Build prompt
    prompt = build_evaluator_prompt(
        task_id=task_id,
        goal=goal,
        task_detail=task_detail,
        metrics=metrics_to_eval if metrics_to_eval else metrics,  # Fallback to all if empty
        include_proxy=evaluate_proxy and bool(hybrid_metrics),
        attempt_number=attempt_number,
        previous_scores=previous_scores,
        audits_dir=effective_audits_dir,
        previous_adversarial_responses=effective_prev_adv,
    )

    # Configure thinking for evaluator (default: 10000 — quality assessment benefits)
    eval_options = ClaudeAgentOptions(
        system_prompt=EVALUATOR_SYSTEM_PROMPT,
        allowed_tools=[
            "Read", "Write", "Bash", "Glob", "Grep", "LSP",
            "WebFetch", "WebSearch",
        ],
        permission_mode="bypassPermissions",  # Bash needs bypass to run tests/scripts
        max_turns=25,
        cwd=cwd,
        output_format=EVALUATOR_OUTPUT_SCHEMA,
    )
    effective = thinking_budget if thinking_budget is not None else 10_000
    if effective > 0:
        eval_options.max_thinking_tokens = effective

    # Run evaluator
    sr = await stream_query(
        prompt=prompt,
        options=eval_options,
        agent_name="evaluator",
        emoji="📊",
        cwd=cwd,
        verbose=verbose,
    )
    result_text = sr.text
    result_stats = sr.result_stats

    # Parse result for evaluated metrics
    result = parse_evaluator_output(sr.structured_output, result_text, metrics_to_eval if metrics_to_eval else metrics)
    result.task_id = task_id
    result.result_stats = result_stats

    # Validate score consistency with issue severity
    _validate_score_consistency(result)

    # Warn if evaluator skipped some metrics
    metrics_expected = len(metrics_to_eval if metrics_to_eval else metrics)
    if result.metrics_attempted > 0 and result.metrics_attempted < metrics_expected:
        console.print(
            f"[yellow]\u26a0 Evaluator evaluated {result.metrics_attempted}/{metrics_expected} metrics[/yellow]"
        )

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

    # Read adversarial findings if the evaluator wrote them
    adversarial_findings_path = audits_path / f"adversarial_{task_id}_{attempt_number}.md"
    if adversarial_findings_path.exists():
        findings_content = adversarial_findings_path.read_text()
        if "No adversarial issues found" not in findings_content:
            result.adversarial_findings = findings_content
            result.adversarial_findings_path = str(adversarial_findings_path)
            console.print(f"[yellow]⚔ Adversarial findings written to {adversarial_findings_path.name}[/yellow]")
        else:
            console.print(f"[green]⚔ Adversarial testing: no issues found[/green]")

    # Pivot detection: prompt-driven (agent judgment) with code fallback
    # parse_evaluator_output already sets should_pivot from agent output
    # If agent didn't output PIVOT fields (should_pivot still default False
    # and no pivot_reason), fallback to deterministic code judgment
    if not result.should_pivot and not result.pivot_reason:
        result.should_pivot, result.pivot_reason = _assess_pivot_recommendation(
            result=result,
            previous_scores=previous_scores or [],
            attempt_number=attempt_number,
            pivot_threshold=pivot_threshold,
            min_improvement=min_improvement,
        )

    # Write structured evaluation summary to task file (not full output)
    # Skip writing eval section for infra failures to avoid polluting task history
    task_content = read_task(task_id, cwd)
    if task_content and not result.is_infra_failure:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        eval_section = f"\n\n## Evaluation ({now})\n\n"
        eval_section += f"**Score**: {result.overall_score:.0f}/100\n"
        denominator = result.metrics_attempted if result.metrics_attempted > 0 else len(result.metrics)
        eval_section += f"**Metrics parsed**: {len(result.metrics)}/{denominator}\n\n"
        if result.issues:
            eval_section += "### Issues\n\n"
            for issue in result.issues:
                eval_section += f"- {issue}\n"
        if result.suggestions:
            eval_section += "\n### Suggestions\n\n"
            for suggestion in result.suggestions:
                eval_section += f"- {suggestion}\n"
        write_task(task_id, task_content + eval_section, cwd)
    elif result.is_infra_failure:
        console.print("[yellow]⚠ EVAL_INFRA_FAILURE: evaluator did not produce valid output (turn limit?). Not recording to task history.[/yellow]")

    # Save full evaluator output to separate log file
    eval_log_path = Path(cwd) / RALPH_DIR / "logs" / f"eval_{task_id}_{attempt_number}.md"
    eval_log_path.parent.mkdir(parents=True, exist_ok=True)
    eval_log_path.write_text(result_text)

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

    Only searches within ## Evaluation sections to avoid matching
    scores mentioned in Notes, Description, or other contexts.

    Filters out infra failures (score=0 with 0 metrics parsed) to avoid
    polluting pivot detection with false signals.

    Returns:
        (attempt_number, previous_scores)
    """
    task_content = read_task(task_id, cwd)
    if not task_content:
        return 1, []

    # Extract only ## Evaluation sections, then search for scores within them
    previous_scores = []
    for match in re.finditer(r"## Evaluation\b[^\n]*\n(.*?)(?=\n## |\Z)", task_content, re.DOTALL):
        section = match.group(1)
        score_match = re.search(r"\*\*Score\*\*:\s*(\d+(?:\.\d+)?)/100", section)
        if not score_match:
            continue
        score = float(score_match.group(1))

        # Filter out infra failures: score=0 with 0 metrics parsed
        # Pattern: **Metrics parsed**: 0/0 or 0/N where score is also 0
        if score == 0:
            metrics_match = re.search(r"\*\*Metrics parsed\*\*:\s*0/", section)
            if metrics_match:
                continue  # Skip this infra failure entry

        previous_scores.append(score)

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


def read_adversarial_response(task_id: str, attempt: int, cwd: str = ".") -> Optional[str]:
    """Read Worker's adversarial response file (rebuttal/fix report).

    Returns the file content if it exists, None otherwise.
    """
    path = Path(cwd) / AUDITS_DIR / f"response_{task_id}_{attempt}.md"
    return path.read_text() if path.exists() else None
