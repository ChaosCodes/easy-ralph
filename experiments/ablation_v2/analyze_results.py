#!/usr/bin/env python3
"""
Analyze results from all ablation experiments.
"""

import json
from pathlib import Path
from datetime import datetime


def load_latest_results(results_dir: Path, prefix: str) -> dict | None:
    """Load the most recent results file for an experiment."""
    files = sorted(results_dir.glob(f"{prefix}_*.json"), reverse=True)
    if not files:
        return None
    with open(files[0]) as f:
        return json.load(f)


def analyze_experiment1(results: dict) -> dict:
    """Analyze Clarifier mode comparison results."""
    analysis = {
        "experiment": "Clarifier Mode Comparison",
        "hypothesis": "Explore+propose mode helps users clarify vague requirements better",
    }

    ask_results = [r for r in results["results"] if r["mode"] == "ask"]
    explore_results = [r for r in results["results"] if r["mode"] == "explore"]

    # Compare clarity scores
    ask_clarity = [r.get("clarity_eval", {}).get("overall", 0) for r in ask_results if r.get("success")]
    explore_clarity = [r.get("clarity_eval", {}).get("overall", 0) for r in explore_results if r.get("success")]

    ask_avg = sum(ask_clarity) / len(ask_clarity) if ask_clarity else 0
    explore_avg = sum(explore_clarity) / len(explore_clarity) if explore_clarity else 0

    # Compare time
    ask_time = [r.get("elapsed_seconds", 0) for r in ask_results if r.get("success")]
    explore_time = [r.get("elapsed_seconds", 0) for r in explore_results if r.get("success")]

    analysis["metrics"] = {
        "ask_mode": {
            "avg_clarity": ask_avg,
            "avg_time": sum(ask_time) / len(ask_time) if ask_time else 0,
            "success_rate": len([r for r in ask_results if r.get("success")]) / len(ask_results) if ask_results else 0,
        },
        "explore_mode": {
            "avg_clarity": explore_avg,
            "avg_time": sum(explore_time) / len(explore_time) if explore_time else 0,
            "success_rate": len([r for r in explore_results if r.get("success")]) / len(explore_results) if explore_results else 0,
        },
    }

    # Conclusion
    if explore_avg > ask_avg + 5:
        analysis["conclusion"] = "CONFIRMED: Explore mode produces clearer goals"
    elif ask_avg > explore_avg + 5:
        analysis["conclusion"] = "REJECTED: Ask mode produces clearer goals"
    else:
        analysis["conclusion"] = "INCONCLUSIVE: No significant difference"

    return analysis


def analyze_experiment2(results: dict) -> dict:
    """Analyze Pivot mechanism value results."""
    analysis = {
        "experiment": "Pivot Mechanism Value",
        "hypothesis": "Pivot triggers reduce wasted effort on dead-end approaches",
    }

    no_pivot = next((r for r in results["results"] if r["mode"] == "no_pivot"), {})
    with_pivot = next((r for r in results["results"] if r["mode"] == "with_pivot"), {})

    analysis["metrics"] = {
        "no_pivot": {
            "success": no_pivot.get("success", False),
            "time": no_pivot.get("elapsed_seconds", 0),
            "pivot_count": no_pivot.get("pivot_count", 0),
        },
        "with_pivot": {
            "success": with_pivot.get("success", False),
            "time": with_pivot.get("elapsed_seconds", 0),
            "pivot_count": with_pivot.get("pivot_count", 0),
        },
    }

    # Compare
    time_saved = no_pivot.get("elapsed_seconds", 0) - with_pivot.get("elapsed_seconds", 0)
    if time_saved > 0 and with_pivot.get("pivot_count", 0) > 0:
        analysis["conclusion"] = f"CONFIRMED: Pivot saved {time_saved:.1f}s"
    else:
        analysis["conclusion"] = "INCONCLUSIVE: Need more data"

    return analysis


def analyze_experiment3(results: dict) -> dict:
    """Analyze Reviewer vs Combined Evaluator results."""
    analysis = {
        "experiment": "Reviewer vs Combined Evaluator",
        "hypothesis": "Merging doesn't reduce issue detection quality",
    }

    separate = next((r for r in results["results"] if r["mode"] == "separate"), {})
    combined = next((r for r in results["results"] if r["mode"] == "combined"), {})

    analysis["metrics"] = {
        "separate": {
            "time": separate.get("total_time", 0),
            "issues_found": len(separate.get("eval_issues", [])),
            "score": separate.get("eval_score", 0),
        },
        "combined": {
            "time": combined.get("total_time", 0),
            "issues_found": len(combined.get("eval_issues", [])),
            "score": combined.get("eval_score", 0),
        },
    }

    comparison = results.get("comparison", {})
    analysis["issue_comparison"] = {
        "issues_in_both": len(comparison.get("issues_in_both", [])),
        "only_in_separate": len(comparison.get("issues_only_in_separate", [])),
        "only_in_combined": len(comparison.get("issues_only_in_combined", [])),
    }

    # Determine if quality is maintained
    missed_issues = len(comparison.get("issues_only_in_separate", []))
    if missed_issues == 0:
        analysis["conclusion"] = "CONFIRMED: Combined mode finds same issues faster"
    elif missed_issues <= 1:
        analysis["conclusion"] = "PARTIAL: Combined mode misses minor issues"
    else:
        analysis["conclusion"] = f"REJECTED: Combined mode misses {missed_issues} issues"

    return analysis


def analyze_experiment4(results: dict) -> dict:
    """Analyze Autonomous Judgment vs Wait results."""
    analysis = {
        "experiment": "Autonomous Judgment vs Wait",
        "hypothesis": "Agent self-judgment is more efficient than waiting",
    }

    user_wait = next((r for r in results["results"] if r["mode"] == "user_wait"), {})
    autonomous = next((r for r in results["results"] if r["mode"] == "autonomous"), {})

    analysis["metrics"] = {
        "user_wait": {
            "total_time": user_wait.get("total_time", 0),
            "user_wait_time": user_wait.get("total_user_wait", 0),
            "iterations": user_wait.get("iterations", 0),
            "pivot_at": user_wait.get("pivot_at_iteration"),
        },
        "autonomous": {
            "total_time": autonomous.get("total_time", 0),
            "user_wait_time": autonomous.get("total_user_wait", 0),
            "iterations": autonomous.get("iterations", 0),
            "pivot_at": autonomous.get("pivot_at_iteration"),
        },
    }

    # Calculate efficiency gain
    time_saved = user_wait.get("total_user_wait", 0)
    iterations_saved = user_wait.get("iterations", 0) - autonomous.get("iterations", 0)

    analysis["efficiency_gain"] = {
        "time_saved": time_saved,
        "iterations_saved": iterations_saved,
    }

    if time_saved > 0 and iterations_saved > 0:
        analysis["conclusion"] = f"CONFIRMED: Autonomous mode saves {time_saved:.1f}s and {iterations_saved} iterations"
    elif time_saved > 0:
        analysis["conclusion"] = f"PARTIAL: Autonomous mode saves {time_saved:.1f}s but same iterations"
    else:
        analysis["conclusion"] = "INCONCLUSIVE: No significant efficiency gain"

    return analysis


def main():
    """Analyze all experiment results."""
    results_dir = Path(__file__).parent / "results"

    if not results_dir.exists():
        print("No results directory found. Run experiments first.")
        return

    print("=" * 70)
    print("Ralph SDK Ablation Study v2 - Results Analysis")
    print("=" * 70)
    print()

    analyses = []

    # Experiment 1
    exp1_results = load_latest_results(results_dir, "experiment1")
    if exp1_results:
        analyses.append(analyze_experiment1(exp1_results))
    else:
        print("Experiment 1: No results found")

    # Experiment 2
    exp2_results = load_latest_results(results_dir, "experiment2")
    if exp2_results:
        analyses.append(analyze_experiment2(exp2_results))
    else:
        print("Experiment 2: No results found")

    # Experiment 3
    exp3_results = load_latest_results(results_dir, "experiment3")
    if exp3_results:
        analyses.append(analyze_experiment3(exp3_results))
    else:
        print("Experiment 3: No results found")

    # Experiment 4
    exp4_results = load_latest_results(results_dir, "experiment4")
    if exp4_results:
        analyses.append(analyze_experiment4(exp4_results))
    else:
        print("Experiment 4: No results found")

    # Print analyses
    for analysis in analyses:
        print(f"\n## {analysis['experiment']}")
        print(f"Hypothesis: {analysis['hypothesis']}")
        print()

        if "metrics" in analysis:
            print("Metrics:")
            for mode, metrics in analysis["metrics"].items():
                print(f"  {mode}:")
                for k, v in metrics.items():
                    if isinstance(v, float):
                        print(f"    {k}: {v:.2f}")
                    else:
                        print(f"    {k}: {v}")

        if "issue_comparison" in analysis:
            print("\nIssue Comparison:")
            for k, v in analysis["issue_comparison"].items():
                print(f"  {k}: {v}")

        if "efficiency_gain" in analysis:
            print("\nEfficiency Gain:")
            for k, v in analysis["efficiency_gain"].items():
                print(f"  {k}: {v}")

        print(f"\n**Conclusion**: {analysis.get('conclusion', 'N/A')}")

    # Overall summary
    print("\n" + "=" * 70)
    print("Overall Summary")
    print("=" * 70)

    confirmed = sum(1 for a in analyses if "CONFIRMED" in a.get("conclusion", ""))
    partial = sum(1 for a in analyses if "PARTIAL" in a.get("conclusion", ""))
    rejected = sum(1 for a in analyses if "REJECTED" in a.get("conclusion", ""))
    inconclusive = sum(1 for a in analyses if "INCONCLUSIVE" in a.get("conclusion", ""))

    print(f"\nHypotheses:")
    print(f"  Confirmed: {confirmed}")
    print(f"  Partial: {partial}")
    print(f"  Rejected: {rejected}")
    print(f"  Inconclusive: {inconclusive}")

    # Save analysis
    analysis_file = results_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(analysis_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "analyses": analyses,
            "summary": {
                "confirmed": confirmed,
                "partial": partial,
                "rejected": rejected,
                "inconclusive": inconclusive,
            },
        }, f, indent=2, ensure_ascii=False)

    print(f"\nAnalysis saved to: {analysis_file}")


if __name__ == "__main__":
    main()
