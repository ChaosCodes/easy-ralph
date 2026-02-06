"""
Experiment runner for Ralph SDK pipeline evaluation.

Real-world scenarios to test different hypotheses about the pipeline.
"""

import asyncio
import json
import os
import shutil
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ralph_sdk.orchestrator import run
from ralph_sdk.pool import read_pool, read_task, read_goal
from ralph_sdk.evaluator import evaluate, Metric, MetricType, AutomationLevel


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    experiment_name: str
    variant: str
    task_description: str

    # Timing
    start_time: str
    end_time: str
    duration_seconds: float

    # Outcome
    success: bool
    iterations_used: int

    # Quality metrics
    reviewer_verdict: Optional[str] = None
    evaluator_score: Optional[float] = None

    # Details
    retries_needed: int = 0
    errors: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    name: str
    description: str
    task_prompt: str

    # Variants to test
    variants: dict  # e.g., {"baseline": {...}, "treatment": {...}}

    # How many times to run each variant
    runs_per_variant: int = 3

    # Evaluation criteria
    metrics: list[Metric] = field(default_factory=list)


def setup_test_project(base_dir: Path, project_name: str) -> Path:
    """Create a fresh test project directory."""
    project_dir = base_dir / project_name
    if project_dir.exists():
        shutil.rmtree(project_dir)
    project_dir.mkdir(parents=True)

    # Create a minimal Python project
    (project_dir / "src").mkdir()
    (project_dir / "tests").mkdir()

    # Create a simple module to work with
    (project_dir / "src" / "__init__.py").write_text("")
    (project_dir / "src" / "utils.py").write_text('''"""Utility functions."""

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b
''')

    (project_dir / "tests" / "__init__.py").write_text("")
    (project_dir / "tests" / "test_utils.py").write_text('''"""Tests for utils."""
from src.utils import add, subtract

def test_add():
    assert add(1, 2) == 3

def test_subtract():
    assert subtract(5, 3) == 2
''')

    (project_dir / "pyproject.toml").write_text('''[project]
name = "test-project"
version = "0.1.0"

[tool.pytest.ini_options]
testpaths = ["tests"]
''')

    return project_dir


def clean_ralph_dir(project_dir: Path):
    """Remove .ralph directory if exists."""
    ralph_dir = project_dir / ".ralph"
    if ralph_dir.exists():
        shutil.rmtree(ralph_dir)


async def run_single_experiment(
    project_dir: Path,
    task_prompt: str,
    max_iterations: int = 15,
    variant_config: dict = None,
) -> ExperimentResult:
    """Run a single experiment and collect results."""

    clean_ralph_dir(project_dir)

    start_time = datetime.now()

    try:
        success = await run(
            goal=task_prompt,
            cwd=str(project_dir),
            max_iterations=max_iterations,
            skip_clarify=True,  # Skip for reproducibility
            verbose=False,
        )
    except Exception as e:
        success = False
        error_msg = str(e)
    else:
        error_msg = None

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Collect results from .ralph directory
    pool_content = read_pool(str(project_dir))

    # Count iterations from pool progress log
    iterations = pool_content.count("### 20")  # Timestamp pattern

    # Check for retries
    retries = pool_content.lower().count("retry")

    result = ExperimentResult(
        experiment_name="",  # Will be set by caller
        variant="",  # Will be set by caller
        task_description=task_prompt,
        start_time=start_time.isoformat(),
        end_time=end_time.isoformat(),
        duration_seconds=duration,
        success=success,
        iterations_used=iterations,
        retries_needed=retries,
        errors=[error_msg] if error_msg else [],
    )

    return result


# ============================================================================
# Experiment Definitions
# ============================================================================

EXPERIMENT_1_WORKER_REFLECTION = """
## Experiment 1: Worker Reflection

Hypothesis: Adding explicit self-check step improves first-pass success rate.

Task: Add a new function to utils.py that validates email addresses.
This task has clear boundary cases that are easy to miss:
- Empty string
- No @ symbol
- Multiple @ symbols
- Invalid domain

We compare:
- Baseline: Current Worker prompt
- Treatment: Worker prompt with explicit reflection step
"""

EXPERIMENT_2_PLANNER_CONTEXT = """
## Experiment 2: Planner Context Depth

Hypothesis: Planner with deeper context makes better decisions after failures.

Scenario: Deliberately create a failing task, then see if Planner can
correctly diagnose and create appropriate follow-up tasks.

We compare:
- Baseline: Planner reads only pool.md
- Treatment: Planner reads pool.md + relevant task details
"""

EXPERIMENT_3_EVALUATOR_ANCHORING = """
## Experiment 3: Evaluator Anchoring Effect

Hypothesis: Evaluator scores are influenced by seeing Reviewer verdict first.

We use the same completed code and compare:
- A: Evaluator sees no Reviewer info
- B: Evaluator sees "Reviewer: PASSED"
- C: Evaluator sees "Reviewer: RETRY needed"
"""


def get_experiment_1_config() -> ExperimentConfig:
    """Worker reflection experiment config."""
    return ExperimentConfig(
        name="worker_reflection",
        description="Test if explicit reflection improves first-pass success",
        task_prompt="""Add a new function `validate_email(email: str) -> bool` to src/utils.py.

Requirements:
- Return True if email is valid, False otherwise
- Valid email must have exactly one @ symbol
- Must have non-empty local part (before @)
- Must have valid domain (after @) with at least one dot
- Handle edge cases: empty string, None, whitespace

Also add tests in tests/test_utils.py for the new function.
""",
        variants={
            "baseline": {
                "worker_prompt_suffix": "",
            },
            "reflection": {
                "worker_prompt_suffix": """

After implementing, perform these self-checks:
1. Read your code again and trace through edge cases
2. Run the tests if possible
3. List any potential issues you find
4. Fix issues before marking complete
""",
            },
        },
        runs_per_variant=2,
    )


def get_experiment_2_config() -> ExperimentConfig:
    """Planner context depth experiment config."""
    return ExperimentConfig(
        name="planner_context",
        description="Test if deeper context helps Planner make better decisions",
        task_prompt="""Add a function `divide(a: int, b: int) -> float` to src/utils.py.

Requirements:
- Return a / b
- Handle division by zero gracefully (return None or raise ValueError)
- Add appropriate tests
""",
        variants={
            "shallow": {
                "planner_reads_tasks": False,
            },
            "deep": {
                "planner_reads_tasks": True,
            },
        },
        runs_per_variant=2,
    )


async def run_experiment_1():
    """Run Worker reflection experiment."""
    print("\n" + "="*60)
    print("EXPERIMENT 1: Worker Reflection")
    print("="*60)

    config = get_experiment_1_config()
    base_dir = Path("/tmp/ralph_experiments")
    results = []

    for variant_name, variant_config in config.variants.items():
        print(f"\n--- Variant: {variant_name} ---")

        for run_idx in range(config.runs_per_variant):
            print(f"\nRun {run_idx + 1}/{config.runs_per_variant}")

            project_dir = setup_test_project(base_dir, f"exp1_{variant_name}_{run_idx}")

            # For now, we just run with default settings
            # In a full implementation, we'd modify the Worker prompt based on variant_config

            result = await run_single_experiment(
                project_dir=project_dir,
                task_prompt=config.task_prompt,
                max_iterations=10,
                variant_config=variant_config,
            )
            result.experiment_name = config.name
            result.variant = variant_name

            results.append(result)

            print(f"  Success: {result.success}")
            print(f"  Duration: {result.duration_seconds:.1f}s")
            print(f"  Iterations: {result.iterations_used}")
            print(f"  Retries: {result.retries_needed}")

    return results


def analyze_results(results: list[ExperimentResult]) -> dict:
    """Analyze experiment results and generate summary."""
    by_variant = {}

    for r in results:
        if r.variant not in by_variant:
            by_variant[r.variant] = []
        by_variant[r.variant].append(r)

    summary = {}
    for variant, runs in by_variant.items():
        summary[variant] = {
            "total_runs": len(runs),
            "successes": sum(1 for r in runs if r.success),
            "success_rate": sum(1 for r in runs if r.success) / len(runs),
            "avg_duration": sum(r.duration_seconds for r in runs) / len(runs),
            "avg_iterations": sum(r.iterations_used for r in runs) / len(runs),
            "avg_retries": sum(r.retries_needed for r in runs) / len(runs),
        }

    return summary


async def main():
    """Run all experiments."""
    print("Ralph SDK Pipeline Experiments")
    print("="*60)

    # Run experiment 1
    results = await run_experiment_1()

    # Analyze
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    summary = analyze_results(results)
    for variant, stats in summary.items():
        print(f"\n{variant}:")
        print(f"  Success rate: {stats['success_rate']*100:.0f}%")
        print(f"  Avg duration: {stats['avg_duration']:.1f}s")
        print(f"  Avg iterations: {stats['avg_iterations']:.1f}")
        print(f"  Avg retries: {stats['avg_retries']:.1f}")

    # Save detailed results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"experiment_results_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump({
            "results": [asdict(r) for r in results],
            "summary": summary,
        }, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
