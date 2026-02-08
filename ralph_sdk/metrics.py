"""
Metrics system: defines success criteria for tasks.

Used by:
- Clarifier: to define metrics during requirement clarification
- Evaluator: to evaluate task outputs against metrics
- Planner: to decide if optimization is needed
"""

import re
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class MetricType(Enum):
    """Type of metric - determines evaluation method."""
    HARD = "hard"           # Must pass, binary (tests, builds)
    SOFT = "soft"           # Measurable, has target (performance, accuracy)
    SUBJECTIVE = "subjective"  # AI-evaluated quality


class AutomationLevel(Enum):
    """How the metric is evaluated."""
    AUTO = "auto"           # Fully automated (tests, benchmarks)
    MANUAL = "manual"       # Requires human testing
    HYBRID = "hybrid"       # Proxy metric auto, final needs human


class TaskCategory(Enum):
    """Category of task - determines default metrics."""
    ALGORITHM = "algorithm"  # Algorithms, data processing, ML
    WEB = "web"              # Web applications, frontend
    API = "api"              # Backend APIs, services
    CLI = "cli"              # Command-line tools
    LIBRARY = "library"      # Libraries, SDKs
    GENERAL = "general"      # Default / unknown


@dataclass
class MetricDefinition:
    """Definition of a metric."""
    name: str
    type: MetricType
    description: str
    target: Optional[str] = None  # e.g., ">= 95%", "<= 100ms"
    measure_command: Optional[str] = None  # Bash command to measure
    priority: str = "medium"  # high / medium / low
    automation: AutomationLevel = AutomationLevel.AUTO  # How to evaluate
    proxy_metric: Optional[str] = None  # Proxy metric for hybrid/manual
    batch_suggestion: Optional[str] = None  # How to batch test


@dataclass
class EvalConfig:
    """Evaluation configuration - how testing should be done."""
    mode: str = "全自动"  # 全自动 / 半自动 / 人工为主
    test_frequency: Optional[str] = None  # 实时 / 每小时 / 每天 / 更久
    batch_preference: Optional[str] = None  # 一个个测 / 批量测 / 自动筛选
    batch_size: int = 3  # Number of checkpoints to accumulate before pausing

    def needs_user_testing(self) -> bool:
        """Check if user testing is required."""
        return self.mode in ["半自动", "人工为主"]

    def should_batch(self) -> bool:
        """Check if batch testing is preferred."""
        return self.batch_preference and "批量" in self.batch_preference

    def should_auto_filter(self) -> bool:
        """Check if auto-filtering is preferred."""
        return self.batch_preference and "自动筛选" in self.batch_preference


@dataclass
class MetricsConfig:
    """Complete metrics configuration for a task."""
    category: TaskCategory
    hard_constraints: list[MetricDefinition] = field(default_factory=list)
    soft_targets: list[MetricDefinition] = field(default_factory=list)
    subjective_criteria: list[MetricDefinition] = field(default_factory=list)
    checkpoints: list[str] = field(default_factory=list)  # When to pause for user review
    eval_config: EvalConfig = field(default_factory=EvalConfig)  # Evaluation configuration

    def all_metrics(self) -> list[MetricDefinition]:
        """Get all metrics as a flat list."""
        return self.hard_constraints + self.soft_targets + self.subjective_criteria

    def to_markdown(self) -> str:
        """Convert to markdown for goal.md."""
        lines = ["## Success Metrics", ""]

        if self.hard_constraints:
            lines.append("### Hard Constraints (must pass)")
            for m in self.hard_constraints:
                lines.append(f"- [ ] **{m.name}**: {m.description}")
            lines.append("")

        if self.soft_targets:
            lines.append("### Optimization Targets")
            lines.append("| Metric | Target | Priority |")
            lines.append("|--------|--------|----------|")
            for m in self.soft_targets:
                lines.append(f"| {m.name} | {m.target or 'N/A'} | {m.priority.upper()} |")
            lines.append("")

        if self.subjective_criteria:
            lines.append("### Quality Criteria (AI-evaluated)")
            for m in self.subjective_criteria:
                lines.append(f"- **{m.name}**: {m.description}")
            lines.append("")

        if self.checkpoints:
            lines.append("### Checkpoints")
            for cp in self.checkpoints:
                lines.append(f"- {cp}")
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "category": self.category.value,
            "hard_constraints": [
                {"name": m.name, "type": m.type.value, "description": m.description, "target": m.target}
                for m in self.hard_constraints
            ],
            "soft_targets": [
                {"name": m.name, "type": m.type.value, "description": m.description, "target": m.target, "priority": m.priority}
                for m in self.soft_targets
            ],
            "subjective_criteria": [
                {"name": m.name, "type": m.type.value, "description": m.description}
                for m in self.subjective_criteria
            ],
            "checkpoints": self.checkpoints,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MetricsConfig":
        """Create from dictionary."""
        return cls(
            category=TaskCategory(data.get("category", "general")),
            hard_constraints=[
                MetricDefinition(
                    name=m["name"],
                    type=MetricType(m.get("type", "hard")),
                    description=m["description"],
                    target=m.get("target"),
                )
                for m in data.get("hard_constraints", [])
            ],
            soft_targets=[
                MetricDefinition(
                    name=m["name"],
                    type=MetricType(m.get("type", "soft")),
                    description=m["description"],
                    target=m.get("target"),
                    priority=m.get("priority", "medium"),
                )
                for m in data.get("soft_targets", [])
            ],
            subjective_criteria=[
                MetricDefinition(
                    name=m["name"],
                    type=MetricType(m.get("type", "subjective")),
                    description=m["description"],
                )
                for m in data.get("subjective_criteria", [])
            ],
            checkpoints=data.get("checkpoints", []),
        )


# -----------------------------------------------------------------------------
# Default Metrics by Category
# -----------------------------------------------------------------------------

DEFAULT_METRICS: dict[TaskCategory, MetricsConfig] = {
    TaskCategory.ALGORITHM: MetricsConfig(
        category=TaskCategory.ALGORITHM,
        hard_constraints=[
            MetricDefinition("tests_pass", MetricType.HARD, "All tests pass"),
            MetricDefinition("no_runtime_errors", MetricType.HARD, "No runtime errors"),
        ],
        soft_targets=[
            MetricDefinition("accuracy", MetricType.SOFT, "Algorithm accuracy", target=">= 90%", priority="high"),
            MetricDefinition("performance", MetricType.SOFT, "Execution time", target="<= 1s", priority="medium"),
        ],
        subjective_criteria=[
            MetricDefinition("code_quality", MetricType.SUBJECTIVE, "Code readability, structure, and maintainability"),
            MetricDefinition("algorithm_elegance", MetricType.SUBJECTIVE, "Algorithmic clarity and efficiency"),
        ],
    ),

    TaskCategory.WEB: MetricsConfig(
        category=TaskCategory.WEB,
        hard_constraints=[
            MetricDefinition("builds", MetricType.HARD, "Project builds without errors"),
            MetricDefinition("no_console_errors", MetricType.HARD, "No JavaScript console errors"),
        ],
        soft_targets=[
            MetricDefinition("lighthouse_score", MetricType.SOFT, "Lighthouse performance score", target=">= 80", priority="medium"),
            MetricDefinition("bundle_size", MetricType.SOFT, "JavaScript bundle size", target="<= 500KB", priority="low"),
        ],
        subjective_criteria=[
            MetricDefinition("ui_quality", MetricType.SUBJECTIVE, "Visual design, layout, and consistency"),
            MetricDefinition("ux_quality", MetricType.SUBJECTIVE, "User experience, interaction flow"),
            MetricDefinition("responsive", MetricType.SUBJECTIVE, "Works well on mobile and desktop"),
            MetricDefinition("code_quality", MetricType.SUBJECTIVE, "Code readability and structure"),
        ],
    ),

    TaskCategory.API: MetricsConfig(
        category=TaskCategory.API,
        hard_constraints=[
            MetricDefinition("tests_pass", MetricType.HARD, "All tests pass"),
            MetricDefinition("type_check", MetricType.HARD, "Type checking passes"),
        ],
        soft_targets=[
            MetricDefinition("response_time", MetricType.SOFT, "API response time P95", target="<= 200ms", priority="medium"),
            MetricDefinition("test_coverage", MetricType.SOFT, "Test coverage", target=">= 80%", priority="low"),
        ],
        subjective_criteria=[
            MetricDefinition("api_design", MetricType.SUBJECTIVE, "RESTful conventions, naming consistency, error handling"),
            MetricDefinition("code_quality", MetricType.SUBJECTIVE, "Code readability and structure"),
        ],
    ),

    TaskCategory.CLI: MetricsConfig(
        category=TaskCategory.CLI,
        hard_constraints=[
            MetricDefinition("runs", MetricType.HARD, "CLI runs without errors"),
            MetricDefinition("help_works", MetricType.HARD, "--help shows usage"),
        ],
        soft_targets=[],
        subjective_criteria=[
            MetricDefinition("cli_ux", MetricType.SUBJECTIVE, "Clear output, helpful error messages, intuitive flags"),
            MetricDefinition("code_quality", MetricType.SUBJECTIVE, "Code readability and structure"),
        ],
    ),

    TaskCategory.LIBRARY: MetricsConfig(
        category=TaskCategory.LIBRARY,
        hard_constraints=[
            MetricDefinition("tests_pass", MetricType.HARD, "All tests pass"),
            MetricDefinition("type_check", MetricType.HARD, "Type checking passes"),
        ],
        soft_targets=[
            MetricDefinition("test_coverage", MetricType.SOFT, "Test coverage", target=">= 90%", priority="high"),
        ],
        subjective_criteria=[
            MetricDefinition("api_design", MetricType.SUBJECTIVE, "Clean, intuitive API surface"),
            MetricDefinition("documentation", MetricType.SUBJECTIVE, "Clear docstrings and examples"),
            MetricDefinition("code_quality", MetricType.SUBJECTIVE, "Code readability and structure"),
        ],
    ),

    TaskCategory.GENERAL: MetricsConfig(
        category=TaskCategory.GENERAL,
        hard_constraints=[
            MetricDefinition("no_errors", MetricType.HARD, "Code runs without errors"),
            MetricDefinition("requirements_met", MetricType.HARD, "All stated requirements satisfied"),
        ],
        soft_targets=[],
        subjective_criteria=[
            MetricDefinition("code_quality", MetricType.SUBJECTIVE, "Code readability, structure, and maintainability"),
        ],
    ),
}


def get_default_metrics(category: TaskCategory) -> MetricsConfig:
    """Get default metrics for a task category."""
    return DEFAULT_METRICS.get(category, DEFAULT_METRICS[TaskCategory.GENERAL])


# -----------------------------------------------------------------------------
# Category Detection
# -----------------------------------------------------------------------------

def detect_category(goal: str, file_patterns: list[str] | None = None) -> TaskCategory:
    """
    Detect task category from goal description and file patterns.

    .. deprecated::
        Category detection is now handled by the agent via prompt.
        Kept for backward compatibility with experiment tests.

    Args:
        goal: The goal description text
        file_patterns: List of files in the project (optional)

    Returns:
        Detected TaskCategory
    """
    warnings.warn(
        "detect_category() is deprecated. Category detection is now handled by the agent via prompt.",
        DeprecationWarning,
        stacklevel=2,
    )
    goal_lower = goal.lower()

    # Check for algorithm-related keywords
    algorithm_keywords = [
        "algorithm", "sort", "search", "graph", "tree", "optimize",
        "calculate", "compute", "parse", "process", "analyze",
        "machine learning", "ml", "model", "prediction", "accuracy",
    ]
    if any(kw in goal_lower for kw in algorithm_keywords):
        return TaskCategory.ALGORITHM

    # Check for web-related keywords
    web_keywords = [
        "web", "website", "frontend", "react", "vue", "angular",
        "html", "css", "ui", "user interface", "page", "component",
        "responsive", "mobile", "browser",
    ]
    if any(kw in goal_lower for kw in web_keywords):
        return TaskCategory.WEB

    # Check for API-related keywords
    api_keywords = [
        "api", "endpoint", "rest", "graphql", "backend", "server",
        "route", "controller", "middleware", "authentication", "jwt",
        "database", "crud",
    ]
    if any(kw in goal_lower for kw in api_keywords):
        return TaskCategory.API

    # Check for CLI-related keywords
    cli_keywords = [
        "cli", "command line", "terminal", "console", "script",
        "argparse", "click", "typer",
    ]
    if any(kw in goal_lower for kw in cli_keywords):
        return TaskCategory.CLI

    # Check for library-related keywords
    library_keywords = [
        "library", "sdk", "package", "module", "pip", "npm publish",
        "reusable", "api surface",
    ]
    if any(kw in goal_lower for kw in library_keywords):
        return TaskCategory.LIBRARY

    # Check file patterns if provided
    if file_patterns:
        patterns_str = " ".join(file_patterns).lower()
        if any(ext in patterns_str for ext in [".tsx", ".jsx", ".vue", ".svelte"]):
            return TaskCategory.WEB
        if "routes" in patterns_str or "controllers" in patterns_str:
            return TaskCategory.API

    return TaskCategory.GENERAL


# -----------------------------------------------------------------------------
# Metrics Clarification Prompts
# -----------------------------------------------------------------------------

METRICS_CLARIFICATION_TEMPLATE = """
Based on your goal, I've detected this as a **{category}** task.

Here are the suggested success metrics:

### Hard Constraints (must pass)
{hard_constraints}

### Optimization Targets
{soft_targets}

### Quality Criteria (AI-evaluated)
{subjective_criteria}

---

**Questions:**

1. Does this categorization look right?
   A. Yes, looks good
   B. No, this is more of a {alt_categories}
   C. Other: [please specify]

2. Any metrics you want to add or remove?
   A. Use these defaults
   B. Add custom metric: [specify]
   C. Remove a metric: [specify]

3. What are your target values for optimization metrics?
{target_questions}

4. Do you want checkpoints for manual review?
   A. No, fully automatic (Recommended)
   B. Yes, review after each major step
   C. Yes, review only at the end
"""


def generate_metrics_questions(config: MetricsConfig) -> str:
    """Generate clarification questions for metrics configuration."""
    hard_str = "\n".join([f"- {m.name}: {m.description}" for m in config.hard_constraints]) or "- (none)"
    soft_str = "\n".join([f"- {m.name}: {m.description} [default: {m.target}]" for m in config.soft_targets]) or "- (none)"
    subj_str = "\n".join([f"- {m.name}: {m.description}" for m in config.subjective_criteria]) or "- (none)"

    # Generate target questions
    target_qs = []
    for i, m in enumerate(config.soft_targets):
        target_qs.append(f"   {i+1}. {m.name} target? (default: {m.target})")
    target_str = "\n".join(target_qs) if target_qs else "   (no optimization targets)"

    # Alternative categories
    other_cats = [c.value for c in TaskCategory if c != config.category][:3]
    alt_str = "/".join(other_cats)

    return METRICS_CLARIFICATION_TEMPLATE.format(
        category=config.category.value,
        hard_constraints=hard_str,
        soft_targets=soft_str,
        subjective_criteria=subj_str,
        target_questions=target_str,
        alt_categories=alt_str,
    )


# -----------------------------------------------------------------------------
# Parse Metrics from goal.md
# -----------------------------------------------------------------------------

def parse_metrics_from_goal(goal_content: str) -> Optional[MetricsConfig]:
    """
    Parse metrics configuration from goal.md content.

    DEPRECATED: Metrics extraction is now handled by the evaluator agent via prompt.
    The evaluator reads goal.md directly and extracts metrics from the Success Metrics section.
    Kept for backward compatibility with experiment tests.

    Returns None if no metrics section found.
    """
    # Check if there's a metrics section
    if "## Success Metrics" not in goal_content:
        return None

    # Extract the metrics section
    metrics_match = re.search(
        r"## Success Metrics\s*\n(.*?)(?=\n## |\Z)",
        goal_content,
        re.DOTALL
    )
    if not metrics_match:
        return None

    metrics_section = metrics_match.group(1)

    # Parse hard constraints
    hard_constraints = []
    hard_match = re.search(r"### Hard Constraints.*?\n((?:- .*\n)+)", metrics_section)
    if hard_match:
        for line in hard_match.group(1).strip().split("\n"):
            name_match = re.search(r"\*\*(\w+)\*\*:\s*(.+)", line)
            if name_match:
                hard_constraints.append(MetricDefinition(
                    name=name_match.group(1),
                    type=MetricType.HARD,
                    description=name_match.group(2).strip(),
                ))

    # Parse soft targets
    soft_targets = []
    soft_match = re.search(r"### Optimization Targets.*?\n\|.*\|\n\|[-|]+\|\n((?:\|.*\|\n)+)", metrics_section)
    if soft_match:
        for line in soft_match.group(1).strip().split("\n"):
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) >= 3:
                soft_targets.append(MetricDefinition(
                    name=parts[0],
                    type=MetricType.SOFT,
                    description="",
                    target=parts[1],
                    priority=parts[2].lower(),
                ))

    # Parse subjective criteria
    subjective_criteria = []
    subj_match = re.search(r"### Quality Criteria.*?\n((?:- .*\n)+)", metrics_section)
    if subj_match:
        for line in subj_match.group(1).strip().split("\n"):
            name_match = re.search(r"\*\*(\w+)\*\*:\s*(.+)", line)
            if name_match:
                subjective_criteria.append(MetricDefinition(
                    name=name_match.group(1),
                    type=MetricType.SUBJECTIVE,
                    description=name_match.group(2).strip(),
                ))

    # Parse checkpoints
    checkpoints = []
    cp_match = re.search(r"### Checkpoints.*?\n((?:- .*\n)+)", metrics_section)
    if cp_match:
        checkpoints = [line.strip("- \n") for line in cp_match.group(1).strip().split("\n")]

    # Detect category (default to general)
    category = TaskCategory.GENERAL
    cat_match = re.search(r"Category:\s*(\w+)", metrics_section, re.IGNORECASE)
    if cat_match:
        try:
            category = TaskCategory(cat_match.group(1).lower())
        except ValueError:
            pass

    return MetricsConfig(
        category=category,
        hard_constraints=hard_constraints,
        soft_targets=soft_targets,
        subjective_criteria=subjective_criteria,
        checkpoints=checkpoints,
    )
