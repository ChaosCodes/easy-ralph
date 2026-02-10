"""
Run all MCP validation experiments.

Usage:
    python -m experiments.mcp_validation.run_all          # Unit tests only (free)
    python -m experiments.mcp_validation.run_all --full    # Include integration tests (requires API)
"""

import asyncio
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.mcp_validation.exp1_verified_info import (
    run_unit_tests as exp1_unit,
    run_integration_tests as exp1_int,
)
from experiments.mcp_validation.exp2_atomic_task import (
    run_unit_tests as exp2_unit,
    run_integration_tests as exp2_int,
)
from experiments.mcp_validation.exp3_pivot_signals import (
    run_unit_tests as exp3_unit,
    run_integration_tests as exp3_int,
)
from experiments.mcp_validation.exp4_concurrent_writes import (
    run_unit_tests as exp4_unit,
    run_integration_tests as exp4_int,
)


def run_unit_tests():
    """Run all unit tests (no API calls, free)."""
    print("\n" + "=" * 70)
    print("  MCP Validation Experiments — Unit Tests")
    print("=" * 70)

    results = []

    for name, fn in [
        ("Exp1: Verified Info", exp1_unit),
        ("Exp2: Atomic Task Creation", exp2_unit),
        ("Exp3: Pivot Signal Detection", exp3_unit),
        ("Exp4: Concurrent Writes", exp4_unit),
    ]:
        print(f"\n{'─' * 60}")
        print(f"  {name}")
        print(f"{'─' * 60}")
        r = fn()
        print(r.summary())
        r.save()
        results.append(r)

    # Final summary
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)

    total = sum(r.total for r in results)
    passed = sum(r.passed for r in results)
    failed = sum(r.failed for r in results)

    for r in results:
        status = "PASS" if r.failed == 0 else "FAIL"
        print(f"  [{status}] {r.experiment_name}: {r.passed}/{r.total}")

    print(f"\n  Overall: {passed}/{total} ({passed/total:.0%})")
    print(f"  Failed: {failed}")

    # Collect design gap findings
    design_gaps = []
    for r in results:
        for tc in r.test_cases:
            if tc.category == "design_gap":
                design_gaps.append((r.experiment_name, tc.name, tc.details))

    if design_gaps:
        print(f"\n{'─' * 60}")
        print("  DESIGN GAPS IDENTIFIED")
        print(f"{'─' * 60}")
        for exp, name, details in design_gaps:
            print(f"\n  [{exp}] {name}")
            for line in details.split(". "):
                if line.strip():
                    print(f"    → {line.strip()}")

    return results


async def run_full(runs: int = 5):
    """Run all tests including integration (requires API)."""
    # First run unit tests
    unit_results = run_unit_tests()

    # Then integration tests
    print("\n" + "=" * 70)
    print("  MCP Validation Experiments — Integration Tests")
    print("=" * 70)

    int_results = []
    for name, fn, run_count in [
        ("Exp1: Verified Info", exp1_int, runs),
        ("Exp2: Atomic Task Creation", exp2_int, runs * 2),  # More runs for CREATE
        ("Exp3: Pivot Signal Detection", exp3_int, runs * 2),
        ("Exp4: Concurrent Writes", exp4_int, runs),
    ]:
        print(f"\n{'─' * 60}")
        print(f"  {name} ({run_count} runs)")
        print(f"{'─' * 60}")
        r = await fn(runs=run_count)
        print(r.summary())
        r.save()
        int_results.append(r)

    # Combined summary
    print("\n" + "=" * 70)
    print("  COMBINED SUMMARY (Unit + Integration)")
    print("=" * 70)

    all_results = unit_results + int_results
    total = sum(r.total for r in all_results)
    passed = sum(r.passed for r in all_results)
    failed = sum(r.failed for r in all_results)

    for r in all_results:
        status = "PASS" if r.failed == 0 else "MIXED"
        print(f"  [{status}] {r.experiment_name}: {r.passed}/{r.total}")

    print(f"\n  Overall: {passed}/{total} ({passed/total:.0%})")

    return unit_results, int_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MCP validation experiments")
    parser.add_argument("--full", action="store_true", help="Include integration tests (requires Claude API)")
    parser.add_argument("--runs", type=int, default=5, help="Base number of integration runs")
    args = parser.parse_args()

    if args.full:
        asyncio.run(run_full(runs=args.runs))
    else:
        run_unit_tests()
