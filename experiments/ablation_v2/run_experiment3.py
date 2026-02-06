#!/usr/bin/env python3
"""
Experiment 3: Reviewer vs Combined Evaluator

Hypothesis: Merging Reviewer and Evaluator doesn't reduce issue detection quality.

A: Separate pipeline (Worker → Reviewer → Evaluator)
B: Combined pipeline (Worker → CombinedEvaluator)
"""

import asyncio
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ralph_sdk.evaluator import evaluate, Metric, MetricType, AutomationLevel
from ralph_sdk.reviewer import review
from ralph_sdk.combined_evaluator import combined_evaluate
from ralph_sdk.pool import init_ralph_dir, init_task, write_goal


# Test task with known issues that should be detected
TEST_TASK_ID = "T001"
TEST_TASK_CONTENT = """# T001: Implement User Authentication

## Type
IMPLEMENT

## Status
pending

## Description
Add user authentication to the API.

## Acceptance Criteria
- [ ] Users can register with email/password
- [ ] Users can login and receive JWT token
- [ ] Protected routes require valid token
- [ ] Passwords are hashed

## Execution Log

### Attempt 1 (simulated)
Implemented basic auth:
- Added /register endpoint
- Added /login endpoint
- Using bcrypt for password hashing
- JWT tokens generated on login

### Files Changed
- api/auth.py (new)
- api/routes.py (modified)
- requirements.txt (modified)

### Code (simulated issues for testing)
```python
# api/auth.py
import jwt
from flask import request

SECRET_KEY = "hardcoded_secret"  # ISSUE: Hardcoded secret

def register(email, password):
    # ISSUE: No email validation
    # ISSUE: No password strength check
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    save_user(email, hashed)

def login(email, password):
    user = get_user(email)
    if bcrypt.checkpw(password.encode(), user.password):
        token = jwt.encode({"user": email}, SECRET_KEY)
        return token
    # ISSUE: No rate limiting on login attempts
```

## Notes
Basic implementation done, but there are security concerns.
"""

TEST_GOAL = """# Goal

## Original Request
Add user authentication to the API

## Success Metrics
### Hard Constraints
- [ ] **runs_without_error** [auto]: pass
- [ ] **tests_pass** [auto]: pass

### Performance Targets
| Metric | Target | Automation |
|--------|--------|------------|
| security_score | >= 80% | auto |
| code_quality | >= 70 | auto |
"""

TEST_METRICS = [
    Metric("runs_without_error", MetricType.HARD, "Code runs without errors"),
    Metric("tests_pass", MetricType.HARD, "All tests pass"),
    Metric("security_score", MetricType.SOFT, "Security best practices", target=">= 80%"),
    Metric("code_quality", MetricType.SUBJECTIVE, "Code readability and structure"),
]


async def setup_test_environment(work_dir: str):
    """Set up test environment with task and goal files."""
    init_ralph_dir(work_dir)
    write_goal(TEST_GOAL, work_dir)

    # Write test task
    task_path = Path(work_dir) / ".ralph" / "tasks" / f"{TEST_TASK_ID}.md"
    task_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.write_text(TEST_TASK_CONTENT)


async def run_separate_pipeline(work_dir: str) -> dict:
    """Run separate Reviewer then Evaluator."""
    print(f"\n{'='*60}")
    print("Running SEPARATE pipeline (Reviewer → Evaluator)")
    print(f"{'='*60}\n")

    start_time = time.time()
    result = {
        "mode": "separate",
        "start_time": datetime.now().isoformat(),
    }

    try:
        # Run reviewer
        review_start = time.time()
        review_result = await review(TEST_TASK_ID, work_dir)
        review_time = time.time() - review_start

        result["review_verdict"] = review_result.verdict.value
        result["review_reason"] = review_result.reason
        result["review_time"] = review_time

        # Run evaluator
        eval_start = time.time()
        eval_result = await evaluate(TEST_TASK_ID, work_dir, metrics=TEST_METRICS)
        eval_time = time.time() - eval_start

        result["eval_score"] = eval_result.overall_score
        result["eval_passed"] = eval_result.overall_passed
        result["eval_issues"] = eval_result.issues
        result["eval_time"] = eval_time

        result["total_time"] = time.time() - start_time
        result["success"] = True

    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        result["total_time"] = time.time() - start_time

    return result


async def run_combined_pipeline(work_dir: str) -> dict:
    """Run combined evaluator."""
    print(f"\n{'='*60}")
    print("Running COMBINED pipeline (CombinedEvaluator)")
    print(f"{'='*60}\n")

    start_time = time.time()
    result = {
        "mode": "combined",
        "start_time": datetime.now().isoformat(),
    }

    try:
        combined_result = await combined_evaluate(TEST_TASK_ID, work_dir, metrics=TEST_METRICS)

        result["review_verdict"] = combined_result.verdict.value
        result["review_reason"] = combined_result.review_reason
        result["eval_score"] = combined_result.overall_score
        result["eval_passed"] = combined_result.overall_passed
        result["eval_issues"] = combined_result.issues

        result["total_time"] = time.time() - start_time
        result["success"] = True

    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        result["total_time"] = time.time() - start_time

    return result


def compare_issue_detection(result_a: dict, result_b: dict) -> dict:
    """Compare issue detection between two approaches."""
    issues_a = set(result_a.get("eval_issues", []))
    issues_b = set(result_b.get("eval_issues", []))

    return {
        "issues_only_in_separate": list(issues_a - issues_b),
        "issues_only_in_combined": list(issues_b - issues_a),
        "issues_in_both": list(issues_a & issues_b),
        "separate_count": len(issues_a),
        "combined_count": len(issues_b),
    }


async def main():
    """Run experiment 3."""
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    base_work_dir = Path(__file__).parent / "work"

    # Test with separate pipeline
    work_dir_a = base_work_dir / "eval_test_separate"
    if work_dir_a.exists():
        shutil.rmtree(work_dir_a)
    work_dir_a.mkdir(parents=True)
    await setup_test_environment(str(work_dir_a))

    result_a = await run_separate_pipeline(str(work_dir_a))

    # Test with combined pipeline
    work_dir_b = base_work_dir / "eval_test_combined"
    if work_dir_b.exists():
        shutil.rmtree(work_dir_b)
    work_dir_b.mkdir(parents=True)
    await setup_test_environment(str(work_dir_b))

    result_b = await run_combined_pipeline(str(work_dir_b))

    # Compare results
    comparison = compare_issue_detection(result_a, result_b)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"experiment3_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump({
            "experiment": "reviewer_vs_combined_evaluator",
            "timestamp": timestamp,
            "results": [result_a, result_b],
            "comparison": comparison,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*60}")

    # Print summary
    print("\n## Summary\n")
    print(f"Separate Pipeline:")
    print(f"  - Total time: {result_a.get('total_time', 0):.1f}s")
    print(f"  - Issues found: {len(result_a.get('eval_issues', []))}")
    print(f"  - Score: {result_a.get('eval_score', 0):.0f}")

    print(f"\nCombined Pipeline:")
    print(f"  - Total time: {result_b.get('total_time', 0):.1f}s")
    print(f"  - Issues found: {len(result_b.get('eval_issues', []))}")
    print(f"  - Score: {result_b.get('eval_score', 0):.0f}")

    print(f"\nComparison:")
    print(f"  - Issues in both: {len(comparison['issues_in_both'])}")
    print(f"  - Only in separate: {len(comparison['issues_only_in_separate'])}")
    print(f"  - Only in combined: {len(comparison['issues_only_in_combined'])}")


if __name__ == "__main__":
    asyncio.run(main())
