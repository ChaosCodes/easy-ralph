#!/usr/bin/env python3
"""
Run all ablation experiments.
"""

import asyncio
import subprocess
import sys
from pathlib import Path


async def main():
    """Run all experiments in sequence."""
    experiments_dir = Path(__file__).parent

    experiments = [
        ("Experiment 1: Clarifier Mode Comparison", "run_experiment1.py"),
        ("Experiment 2: Pivot Mechanism Value", "run_experiment2.py"),
        ("Experiment 3: Reviewer vs Combined Evaluator", "run_experiment3.py"),
        ("Experiment 4: Autonomous Judgment vs Wait", "run_experiment4.py"),
    ]

    print("=" * 70)
    print("Ralph SDK Ablation Study v2")
    print("=" * 70)
    print()

    results = []

    for name, script in experiments:
        print(f"\n{'#' * 70}")
        print(f"# {name}")
        print(f"{'#' * 70}\n")

        script_path = experiments_dir / script

        try:
            # Run experiment as subprocess
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=False,
                text=True,
                cwd=str(experiments_dir),
            )

            results.append({
                "name": name,
                "script": script,
                "success": result.returncode == 0,
                "returncode": result.returncode,
            })

            if result.returncode != 0:
                print(f"[ERROR] {name} failed with return code {result.returncode}")

        except Exception as e:
            print(f"[ERROR] Failed to run {script}: {e}")
            results.append({
                "name": name,
                "script": script,
                "success": False,
                "error": str(e),
            })

    # Print summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    successful = sum(1 for r in results if r["success"])
    total = len(results)

    print(f"\nCompleted: {successful}/{total} experiments\n")

    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"  [{status}] {r['name']}")

    print("\nResults saved in: experiments/ablation_v2/results/")


if __name__ == "__main__":
    asyncio.run(main())
