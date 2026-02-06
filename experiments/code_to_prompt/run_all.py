"""
Run all Code → Prompt experiments.

Phase 1: Sanity checks (no API calls, instant)
Phase 2: Claude API calls (takes time, costs money)
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


async def main():
    print("=" * 70)
    print("Code → Prompt Experiments: Full Suite")
    print("=" * 70)

    results = {}

    # ---- Experiment 1 Phase 1 ----
    from experiments.code_to_prompt.exp1_reviewer_json import run_phase1 as exp1_p1
    print("\n\n" + "=" * 70)
    exp1_result = exp1_p1()
    results["exp1_phase1"] = exp1_result["json_accuracy"] >= 0.95

    # ---- Experiment 2 Phase 1 ----
    from experiments.code_to_prompt.exp2_evaluator_pivot import run_phase1 as exp2_p1
    print("\n\n" + "=" * 70)
    code_correct, code_total = exp2_p1()
    results["exp2_phase1"] = code_correct == code_total

    # ---- Experiment 3 Phase 1 ----
    from experiments.code_to_prompt.exp3_planner_tooluse import run_phase1 as exp3_p1
    print("\n\n" + "=" * 70)
    regex_correct, regex_total = exp3_p1()
    results["exp3_phase1"] = regex_correct == regex_total

    # ---- Phase 2: API calls ----
    print("\n\n" + "=" * 70)
    print("Phase 2: Claude API Experiments")
    print("=" * 70)

    # Experiment 1 Phase 2
    from experiments.code_to_prompt.exp1_reviewer_json_phase2 import run_phase2 as exp1_p2
    results["exp1_phase2"] = await exp1_p2(repeats=5)

    # Experiment 2 Phase 2
    from experiments.code_to_prompt.exp2_evaluator_pivot import run_phase2 as exp2_p2
    results["exp2_phase2"] = await exp2_p2(repeats=5)

    # Experiment 3 Phase 2
    from experiments.code_to_prompt.exp3_planner_tooluse import run_phase2 as exp3_p2
    results["exp3_phase2"] = await exp3_p2(repeats=3)

    # ---- Summary ----
    print("\n\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    all_passed = True
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {name:<25} {status}")
        if not passed:
            all_passed = False

    print(f"\n  Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")

    # Decision for Experiment 4
    api_passed = all(results.get(f"exp{i}_phase2", False) for i in [1, 2, 3])
    if api_passed:
        print("\n  Experiments 1-3 all passed → Experiment 4 (Orchestrator) is worth attempting")
    else:
        print("\n  Some experiments failed → Experiment 4 should wait until failures are addressed")

    return results


if __name__ == "__main__":
    asyncio.run(main())
