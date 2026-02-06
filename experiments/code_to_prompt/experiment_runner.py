"""
Experiment runner for Code → Prompt migration experiments.

Compares control (current code) vs experimental (prompt-based) paths
by running the same inputs through both and measuring agreement.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass
class TrialResult:
    """Result of a single trial comparing control vs experimental."""
    trial_id: int
    scenario: str
    control_output: Any
    experimental_output: Any
    match: bool
    control_time_ms: float = 0
    experimental_time_ms: float = 0
    error: Optional[str] = None
    notes: str = ""


@dataclass
class ExperimentReport:
    """Aggregated report for an experiment."""
    experiment_name: str
    description: str
    total_trials: int
    matches: int
    mismatches: int
    errors: int
    match_rate: float
    success_criteria: str
    passed: bool
    trials: list[TrialResult] = field(default_factory=list)
    mismatch_analysis: list[dict] = field(default_factory=list)
    timestamp: str = ""

    def summary(self) -> str:
        lines = [
            f"=== {self.experiment_name} ===",
            f"{self.description}",
            f"",
            f"Results: {self.matches}/{self.total_trials} matched ({self.match_rate:.1%})",
            f"Errors: {self.errors}",
            f"Criteria: {self.success_criteria}",
            f"PASSED: {'YES' if self.passed else 'NO'}",
        ]
        if self.mismatch_analysis:
            lines.append(f"\nMismatch Analysis:")
            for m in self.mismatch_analysis:
                lines.append(f"  - [{m['scenario']}] control={m['control']} exp={m['experimental']} | {m.get('notes', '')}")
        return "\n".join(lines)


def run_experiment_sync(
    name: str,
    description: str,
    scenarios: list[dict],
    control_fn: Callable,
    experimental_fn: Callable,
    compare_fn: Callable,
    success_threshold: float = 0.9,
    repeats: int = 1,
) -> ExperimentReport:
    """
    Run a synchronous experiment comparing control vs experimental functions.

    Args:
        name: Experiment name
        description: What we're testing
        scenarios: List of dicts, each with 'name' and 'input' keys
        control_fn: The current code path (input -> output)
        experimental_fn: The new prompt-based path (input -> output)
        compare_fn: Function(control_output, experimental_output) -> bool
        success_threshold: Required match rate (0.0-1.0)
        repeats: Number of times to repeat each scenario

    Returns:
        ExperimentReport
    """
    trials = []
    trial_id = 0

    for scenario in scenarios:
        for rep in range(repeats):
            trial_id += 1
            scenario_name = scenario["name"]
            if repeats > 1:
                scenario_name += f"_r{rep+1}"

            try:
                # Run control
                t0 = time.time()
                control_out = control_fn(scenario["input"])
                control_ms = (time.time() - t0) * 1000

                # Run experimental
                t0 = time.time()
                exp_out = experimental_fn(scenario["input"])
                exp_ms = (time.time() - t0) * 1000

                # Compare
                match = compare_fn(control_out, exp_out)

                trials.append(TrialResult(
                    trial_id=trial_id,
                    scenario=scenario_name,
                    control_output=_serialize(control_out),
                    experimental_output=_serialize(exp_out),
                    match=match,
                    control_time_ms=control_ms,
                    experimental_time_ms=exp_ms,
                ))

            except Exception as e:
                trials.append(TrialResult(
                    trial_id=trial_id,
                    scenario=scenario_name,
                    control_output=None,
                    experimental_output=None,
                    match=False,
                    error=str(e),
                ))

    # Aggregate
    matches = sum(1 for t in trials if t.match)
    errors = sum(1 for t in trials if t.error)
    total = len(trials)
    match_rate = matches / total if total > 0 else 0

    # Analyze mismatches
    mismatch_analysis = []
    for t in trials:
        if not t.match and not t.error:
            mismatch_analysis.append({
                "scenario": t.scenario,
                "control": t.control_output,
                "experimental": t.experimental_output,
                "notes": t.notes,
            })

    report = ExperimentReport(
        experiment_name=name,
        description=description,
        total_trials=total,
        matches=matches,
        mismatches=total - matches - errors,
        errors=errors,
        match_rate=match_rate,
        success_criteria=f">= {success_threshold:.0%} match rate",
        passed=match_rate >= success_threshold,
        trials=trials,
        mismatch_analysis=mismatch_analysis,
        timestamp=datetime.now().isoformat(),
    )

    return report


async def run_experiment_async(
    name: str,
    description: str,
    scenarios: list[dict],
    control_fn: Callable,
    experimental_fn: Callable,
    compare_fn: Callable,
    success_threshold: float = 0.9,
    repeats: int = 1,
) -> ExperimentReport:
    """
    Run an async experiment comparing control vs experimental functions.
    Same as run_experiment_sync but supports async control/experimental functions.
    """
    import asyncio
    import inspect

    trials = []
    trial_id = 0

    for scenario in scenarios:
        for rep in range(repeats):
            trial_id += 1
            scenario_name = scenario["name"]
            if repeats > 1:
                scenario_name += f"_r{rep+1}"

            try:
                # Run control
                t0 = time.time()
                if inspect.iscoroutinefunction(control_fn):
                    control_out = await control_fn(scenario["input"])
                else:
                    control_out = control_fn(scenario["input"])
                control_ms = (time.time() - t0) * 1000

                # Run experimental
                t0 = time.time()
                if inspect.iscoroutinefunction(experimental_fn):
                    exp_out = await experimental_fn(scenario["input"])
                else:
                    exp_out = experimental_fn(scenario["input"])
                exp_ms = (time.time() - t0) * 1000

                # Compare
                match = compare_fn(control_out, exp_out)

                trials.append(TrialResult(
                    trial_id=trial_id,
                    scenario=scenario_name,
                    control_output=_serialize(control_out),
                    experimental_output=_serialize(exp_out),
                    match=match,
                    control_time_ms=control_ms,
                    experimental_time_ms=exp_ms,
                ))

            except Exception as e:
                trials.append(TrialResult(
                    trial_id=trial_id,
                    scenario=scenario_name,
                    control_output=None,
                    experimental_output=None,
                    match=False,
                    error=str(e),
                ))

    # Aggregate
    matches = sum(1 for t in trials if t.match)
    errors = sum(1 for t in trials if t.error)
    total = len(trials)
    match_rate = matches / total if total > 0 else 0

    mismatch_analysis = []
    for t in trials:
        if not t.match and not t.error:
            mismatch_analysis.append({
                "scenario": t.scenario,
                "control": t.control_output,
                "experimental": t.experimental_output,
                "notes": t.notes,
            })

    report = ExperimentReport(
        experiment_name=name,
        description=description,
        total_trials=total,
        matches=matches,
        mismatches=total - matches - errors,
        errors=errors,
        match_rate=match_rate,
        success_criteria=f">= {success_threshold:.0%} match rate",
        passed=match_rate >= success_threshold,
        trials=trials,
        mismatch_analysis=mismatch_analysis,
        timestamp=datetime.now().isoformat(),
    )

    return report


def save_report(report: ExperimentReport, output_dir: str = None) -> Path:
    """Save experiment report to JSON and markdown."""
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{report.experiment_name}_{timestamp}"

    # Save JSON
    json_path = output_dir / f"{base_name}.json"
    with open(json_path, "w") as f:
        json.dump(asdict(report), f, indent=2, default=str)

    # Save summary markdown
    md_path = output_dir / f"{base_name}.md"
    with open(md_path, "w") as f:
        f.write(f"# {report.experiment_name}\n\n")
        f.write(f"**Date**: {report.timestamp}\n\n")
        f.write(f"## Summary\n\n")
        f.write(report.summary())
        f.write("\n\n## Trial Details\n\n")
        f.write("| # | Scenario | Match | Control | Experimental | Error |\n")
        f.write("|---|----------|-------|---------|-------------|-------|\n")
        for t in report.trials:
            ctrl = str(t.control_output)[:40]
            exp = str(t.experimental_output)[:40]
            err = t.error or ""
            f.write(f"| {t.trial_id} | {t.scenario} | {'Y' if t.match else 'N'} | {ctrl} | {exp} | {err} |\n")

    print(f"Report saved: {json_path}")
    print(f"Summary saved: {md_path}")
    return json_path


def _serialize(obj: Any) -> Any:
    """Make an object JSON-serializable."""
    if hasattr(obj, "__dict__"):
        return {k: _serialize(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if hasattr(obj, "value"):  # Enum
        return obj.value
    return obj
