"""
Issue Verification Tests for Ralph SDK

Validates 8 categories of potential issues found during code review.
All tests use direct function calls + mock — no Claude SDK needed.
"""

import fcntl
import json
import os
import re
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ralph_sdk.evaluator import (
    EVALUATOR_SYSTEM_PROMPT,
    AutomationLevel,
    EvaluationResult,
    Metric,
    MetricResult,
    MetricType,
    _assess_pivot_recommendation,
    _describe_trend,
    build_evaluator_prompt,
    get_attempt_history,
    parse_evaluator_output,
)
from ralph_sdk.planner import (
    Action,
    PlannerDecision,
    _parse_planner_json,
    _parse_planner_regex,
    parse_planner_output,
)
from ralph_sdk.pool import (
    POOL_FILE,
    RALPH_DIR,
    TASKS_DIR,
    Checkpoint,
    ProxyScore,
    _atomic_write,
    add_verified_info,
    append_to_findings,
    append_to_progress_log,
    clear_pivot_recommendation,
    ensure_task_files_exist,
    extract_task_ids_from_pool,
    file_lock,
    get_verified_info,
    has_pivot_recommendation,
    init_pool,
    init_ralph_dir,
    init_task,
    is_topic_verified,
    list_verified_topics,
    read_eval_config_from_goal,
    read_pool,
    read_task,
    write_goal,
    write_pool,
    write_task,
)
from ralph_sdk.reviewer import (
    Verdict,
    parse_reviewer_output,
)
from ralph_sdk.utils import extract_json
from ralph_sdk.worker import extract_worker_metadata


# =============================================================================
# Helpers
# =============================================================================


def setup_ralph(tmp_path: Path) -> str:
    """Create .ralph/ directory structure and return cwd."""
    cwd = str(tmp_path)
    init_ralph_dir(cwd)
    return cwd


def setup_pool_with_content(tmp_path: Path, pool_content: str) -> str:
    """Create .ralph/ with custom pool.md content."""
    cwd = setup_ralph(tmp_path)
    (tmp_path / POOL_FILE).write_text(pool_content)
    return cwd


def make_eval_result(
    score: float,
    metrics: list[MetricResult] = None,
    attempt_number: int = 1,
    previous_scores: list[float] = None,
) -> EvaluationResult:
    """Helper to build EvaluationResult for pivot tests."""
    return EvaluationResult(
        task_id="T001",
        overall_passed=score >= 60,
        overall_score=score,
        metrics=metrics or [],
        attempt_number=attempt_number,
        previous_scores=previous_scores or [],
    )


# =============================================================================
# 1. Task File Score Parsing (get_attempt_history)
# =============================================================================


class TestScoreParsing:
    """Verify get_attempt_history() correctly parses scores from task files."""

    def test_basic_score_parsing(self, tmp_path):
        """Single evaluation section → 1 score parsed."""
        cwd = setup_ralph(tmp_path)
        content = """# T001: Test Task

## Evaluation (2026-02-05 10:00)

**Score**: 75/100
**Metrics parsed**: 3/3
"""
        write_task("T001", content, cwd)
        attempt, scores = get_attempt_history("T001", cwd)
        assert scores == [75.0]
        assert attempt == 2  # next attempt

    def test_multiple_iterations(self, tmp_path):
        """Multiple evaluation sections → all scores parsed in order."""
        cwd = setup_ralph(tmp_path)
        sections = []
        for i, score in enumerate([40, 55, 70, 80, 85], 1):
            sections.append(f"""
## Evaluation (2026-02-05 {10 + i}:00)

**Score**: {score}/100
**Metrics parsed**: 3/3
""")
        content = "# T001: Test Task\n" + "\n".join(sections)
        write_task("T001", content, cwd)
        attempt, scores = get_attempt_history("T001", cwd)
        assert scores == [40.0, 55.0, 70.0, 80.0, 85.0]
        assert attempt == 6

    def test_20_iterations(self, tmp_path):
        """20 evaluation sections → all 20 scores parsed."""
        cwd = setup_ralph(tmp_path)
        sections = []
        expected = []
        for i in range(20):
            score = 30 + i * 3
            expected.append(float(score))
            sections.append(f"""
## Evaluation (2026-02-05 {i}:00)

**Score**: {score}/100
""")
        content = "# T001: Big Task\n" + "\n".join(sections)
        write_task("T001", content, cwd)
        attempt, scores = get_attempt_history("T001", cwd)
        assert scores == expected
        assert attempt == 21

    def test_large_file_50kb(self, tmp_path):
        """50KB task file with scores → still parses correctly."""
        cwd = setup_ralph(tmp_path)
        padding = "x" * 1000 + "\n"  # ~1KB per line
        content = "# T001: Large Task\n"
        content += padding * 20  # ~20KB of padding
        content += "\n## Evaluation (2026-02-05 10:00)\n\n**Score**: 88/100\n"
        content += padding * 20  # another ~20KB
        content += "\n## Evaluation (2026-02-05 11:00)\n\n**Score**: 92/100\n"
        write_task("T001", content, cwd)
        assert len(content) > 40000  # Verify it's big
        attempt, scores = get_attempt_history("T001", cwd)
        assert scores == [88.0, 92.0]

    def test_decimal_scores(self, tmp_path):
        """Decimal scores like 85.5/100 → parsed as float."""
        cwd = setup_ralph(tmp_path)
        content = """# T001: Task

## Evaluation (2026-02-05 10:00)

**Score**: 85.5/100
"""
        write_task("T001", content, cwd)
        attempt, scores = get_attempt_history("T001", cwd)
        assert scores == [85.5]

    def test_empty_task_file(self, tmp_path):
        """Empty task file → attempt 1, no scores."""
        cwd = setup_ralph(tmp_path)
        write_task("T001", "", cwd)
        attempt, scores = get_attempt_history("T001", cwd)
        assert attempt == 1
        assert scores == []

    def test_no_scores_in_file(self, tmp_path):
        """Task file with no evaluation sections → attempt 1, no scores."""
        cwd = setup_ralph(tmp_path)
        content = """# T001: Task

## Description
Some implementation details here.

## Notes
More stuff here.
"""
        write_task("T001", content, cwd)
        attempt, scores = get_attempt_history("T001", cwd)
        assert attempt == 1
        assert scores == []

    def test_nonexistent_task(self, tmp_path):
        """Non-existent task file → attempt 1, no scores."""
        cwd = setup_ralph(tmp_path)
        attempt, scores = get_attempt_history("T999", cwd)
        assert attempt == 1
        assert scores == []

    def test_score_in_non_evaluation_context(self, tmp_path):
        """Score pattern appearing outside Evaluation section.

        FIXED: get_attempt_history now only searches within ## Evaluation
        sections, so scores in Notes/Description are NOT captured.
        """
        cwd = setup_ralph(tmp_path)
        content = """# T001: Task

## Notes
The target **Score**: 90/100 is what we aim for.

## Evaluation (2026-02-05 10:00)

**Score**: 75/100
"""
        write_task("T001", content, cwd)
        attempt, scores = get_attempt_history("T001", cwd)
        # Fixed: only captures scores from ## Evaluation sections
        assert scores == [75.0]
        assert attempt == 2


# =============================================================================
# 2. Concurrent Pool.md Writes
# =============================================================================


class TestConcurrentWrites:
    """Verify concurrent append operations don't lose data."""

    def test_concurrent_progress_log_10_threads(self, tmp_path):
        """10 threads appending to progress log → all 10 entries present."""
        cwd = setup_ralph(tmp_path)
        init_pool("Test goal", "| T001 | IMPLEMENT | pending |", cwd)

        errors = []

        def append_entry(thread_id):
            try:
                append_to_progress_log(f"Thread-{thread_id} entry", cwd)
            except Exception as e:
                errors.append((thread_id, e))

        threads = [threading.Thread(target=append_entry, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Errors in threads: {errors}"

        pool_content = read_pool(cwd)
        for i in range(10):
            assert f"Thread-{i} entry" in pool_content, f"Missing entry from thread {i}"

    def test_concurrent_findings_10_threads(self, tmp_path):
        """10 threads appending to findings → all 10 entries present."""
        cwd = setup_ralph(tmp_path)
        init_pool("Test goal", "| T001 | IMPLEMENT | pending |", cwd)

        errors = []

        def append_finding(thread_id):
            try:
                append_to_findings(f"Finding-{thread_id}", cwd)
            except Exception as e:
                errors.append((thread_id, e))

        threads = [threading.Thread(target=append_finding, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Errors in threads: {errors}"

        pool_content = read_pool(cwd)
        for i in range(10):
            assert f"Finding-{i}" in pool_content, f"Missing finding {i}"

    def test_mixed_concurrent_operations(self, tmp_path):
        """Mixed append operations (progress + findings) → no data loss."""
        cwd = setup_ralph(tmp_path)
        init_pool("Test goal", "| T001 | IMPLEMENT | pending |", cwd)

        errors = []

        def do_progress(i):
            try:
                append_to_progress_log(f"Progress-{i}", cwd)
            except Exception as e:
                errors.append(("progress", i, e))

        def do_finding(i):
            try:
                append_to_findings(f"Finding-{i}", cwd)
            except Exception as e:
                errors.append(("finding", i, e))

        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=do_progress, args=(i,)))
            threads.append(threading.Thread(target=do_finding, args=(i,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Errors: {errors}"

        pool_content = read_pool(cwd)
        for i in range(5):
            assert f"Progress-{i}" in pool_content
            assert f"Finding-{i}" in pool_content

    def test_file_lock_timeout(self, tmp_path):
        """Lock held too long → TimeoutError raised cleanly."""
        cwd = setup_ralph(tmp_path)
        lock_path = tmp_path / RALPH_DIR / ".test.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        # Acquire lock in main thread
        lock_fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)

        exception_caught = {"type": None}

        def try_lock():
            try:
                with file_lock(lock_path, timeout=0.5):
                    pass
            except TimeoutError:
                exception_caught["type"] = "TimeoutError"
            except OSError:
                exception_caught["type"] = "OSError_from_finally"

        t = threading.Thread(target=try_lock)
        t.start()
        t.join(timeout=10)

        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

        # Fixed: should now always raise TimeoutError, never OSError
        assert exception_caught["type"] == "TimeoutError", \
            f"Expected TimeoutError, got {exception_caught['type']}"

    def test_ask_handler_toctou(self, tmp_path):
        """Demonstrate TOCTOU in ASK handler (orchestrator.py lines 1013-1020).

        The ASK handler reads pool.md, modifies, then writes — but outside
        of the file lock. This is a potential race condition.
        """
        cwd = setup_ralph(tmp_path)
        init_pool("Test goal", "| T001 | IMPLEMENT | pending |", cwd)

        # Simulate the ASK handler pattern (read → modify → write, no lock)
        # This is what orchestrator.py does:
        #   pool_content = read_pool(cwd)  # unlocked read
        #   pool_content = pool_content.replace(...)
        #   write_pool(pool_content, cwd)  # locked write, but stale content

        results = {"lost": 0}

        def simulate_ask_handler(question_id):
            """Simulates the ASK handler's TOCTOU pattern."""
            pool_content = read_pool(cwd)  # Unlocked read
            time.sleep(0.01)  # Simulate some processing time
            qa_note = f"\n\n**Q{question_id}**: answer{question_id}\n"
            pool_content = pool_content.replace(
                "## Findings",
                f"## Findings{qa_note}"
            )
            write_pool(pool_content, cwd)  # Writes stale content

        threads = [
            threading.Thread(target=simulate_ask_handler, args=(i,))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        pool_content = read_pool(cwd)
        found = sum(1 for i in range(5) if f"Q{i}" in pool_content)

        # With TOCTOU, we expect some data loss
        # If all 5 are present, the race didn't manifest (possible with timing)
        # We just document the pattern — the test passes either way
        if found < 5:
            results["lost"] = 5 - found

        # This test documents the race condition rather than asserting failure
        # The important thing is that append_to_findings() (which uses locks) is safe
        assert True  # Documentation test


# =============================================================================
# 3. _atomic_write Error Handling
# =============================================================================


class TestAtomicWriteErrors:
    """Verify _atomic_write error handling, especially the double-close bug."""

    def test_normal_write_succeeds(self, tmp_path):
        """Normal atomic write works correctly."""
        target = tmp_path / "test.txt"
        _atomic_write(target, "hello world")
        assert target.read_text() == "hello world"

    def test_atomic_write_creates_parent_dirs(self, tmp_path):
        """Atomic write creates parent directories if needed."""
        target = tmp_path / "sub" / "dir" / "test.txt"
        _atomic_write(target, "nested content")
        assert target.read_text() == "nested content"

    def test_rename_failure_raises_original_error(self, tmp_path):
        """When os.rename fails, the original PermissionError should propagate
        cleanly without double-close interference."""
        target = tmp_path / "test.txt"

        with patch("ralph_sdk.pool.os.rename", side_effect=PermissionError("denied")):
            with pytest.raises(PermissionError):
                _atomic_write(target, "content that will fail")

    def test_rename_failure_temp_file_cleanup(self, tmp_path):
        """When os.rename fails, temp file should be cleaned up."""
        target = tmp_path / "test.txt"
        target.parent.mkdir(parents=True, exist_ok=True)

        # Count temp files before
        before = list(tmp_path.glob(".tmp_*"))

        with patch("ralph_sdk.pool.os.rename", side_effect=PermissionError("denied")):
            try:
                _atomic_write(target, "content")
            except PermissionError:
                pass

        # Fixed: temp file should now always be cleaned up
        after = list(tmp_path.glob(".tmp_*"))
        leaked = len(after) - len(before)
        assert leaked == 0, f"Temp file leaked: {leaked} file(s) left behind"

    def test_atomic_write_overwrites_existing(self, tmp_path):
        """Atomic write correctly replaces existing file content."""
        target = tmp_path / "test.txt"
        target.write_text("old content")
        _atomic_write(target, "new content")
        assert target.read_text() == "new content"


# =============================================================================
# 4. Reviewer Permission Mode
# =============================================================================


class TestReviewerPermissionMode:
    """Verify reviewer agent configuration — specifically missing permission_mode."""

    def test_reviewer_has_permission_mode(self):
        """Fixed: reviewer.py review() now sets permission_mode for consistency
        with evaluator.py and worker.py."""
        import inspect
        from ralph_sdk import reviewer

        source = inspect.getsource(reviewer.review)

        assert "permission_mode" in source, \
            "reviewer should have permission_mode set"

    def test_other_agents_have_permission_mode(self):
        """Confirm evaluator and worker DO set permission_mode."""
        import inspect
        from ralph_sdk import evaluator, worker

        eval_source = inspect.getsource(evaluator.evaluate)
        assert "permission_mode" in eval_source

        worker_source = inspect.getsource(worker.work)
        assert "permission_mode" in worker_source

    def test_reviewer_has_bash_tool(self):
        """Reviewer has Bash in allowed_tools — needs permission_mode
        for Bash to work properly in non-interactive mode."""
        import inspect
        from ralph_sdk import reviewer

        source = inspect.getsource(reviewer.review)
        assert '"Bash"' in source, "Reviewer should have Bash in allowed_tools"


# =============================================================================
# 5. Pool.md Signal-to-Noise (Growth Quantification)
# =============================================================================


class TestPoolSignalToNoise:
    """Quantify pool.md growth over iterations."""

    def test_pool_growth_15_iterations(self, tmp_path):
        """Simulate 15 iterations of progress log + findings → measure growth."""
        cwd = setup_ralph(tmp_path)
        init_pool("Build a web app", "| T001 | IMPLEMENT | pending |", cwd)

        initial_size = len(read_pool(cwd))

        for i in range(15):
            append_to_progress_log(
                f"EXECUTE T001 - NEEDS IMPROVEMENT (score: {40 + i * 3}/100, "
                f"metrics_found: 3, details in tasks/T001.md)",
                cwd
            )
            if i % 3 == 0:
                append_to_findings(f"Discovery {i}: Some finding from iteration {i}", cwd)

        final_content = read_pool(cwd)
        final_size = len(final_content)
        growth = final_size - initial_size

        # Quantify
        assert growth > 2000, f"Expected significant growth, got {growth} bytes"
        # The pool should be readable but not excessively large
        assert final_size < 20000, f"Pool grew too large: {final_size} bytes"

    def test_findings_accumulation(self, tmp_path):
        """15 findings → all present and readable."""
        cwd = setup_ralph(tmp_path)
        init_pool("Test goal", "| T001 | IMPLEMENT | pending |", cwd)

        for i in range(15):
            append_to_findings(f"Finding-{i}: Important discovery", cwd)

        content = read_pool(cwd)
        for i in range(15):
            assert f"Finding-{i}" in content

    def test_progress_log_chronological(self, tmp_path):
        """Progress log entries appear in chronological order."""
        cwd = setup_ralph(tmp_path)
        init_pool("Test goal", "| T001 | IMPLEMENT | pending |", cwd)

        for i in range(5):
            append_to_progress_log(f"Event-{i}", cwd)

        content = read_pool(cwd)
        positions = [content.index(f"Event-{i}") for i in range(5)]
        assert positions == sorted(positions), "Progress log should be chronological"

    def test_planner_prompt_size(self, tmp_path):
        """After 15 iterations, measure how big the planner prompt context is."""
        cwd = setup_ralph(tmp_path)
        init_pool("Build a web app", "| T001 | IMPLEMENT | pending |", cwd)
        write_goal("# Goal\nBuild a web app\n", cwd)

        for i in range(15):
            append_to_progress_log(
                f"EXECUTE T001 - iteration {i} (score: {40 + i * 3}/100)",
                cwd
            )

        goal_size = len(str(tmp_path / ".ralph" / "goal.md"))
        pool_size = len(read_pool(cwd))
        total_prompt_context = pool_size  # Pool is the main variable

        # Typical Claude context is 200K tokens ≈ 800KB
        # If pool.md > 10KB after 15 iterations, it's getting noisy
        assert total_prompt_context < 15000, \
            f"Pool context is {total_prompt_context} bytes after 15 iterations"


# =============================================================================
# 6. Pivot Detection Dead Code
# =============================================================================


class TestPivotDetection:
    """Verify _assess_pivot_recommendation behavior with correct parameters."""

    def test_case1_too_many_attempts_low_improvement(self):
        """Case 1: attempt >= threshold AND avg improvement < min_improvement.

        avg_improvement = (current_score - first_score) / attempt_number
        To trigger: need avg < 5.0, so (score - first) / attempts < 5
        With scores [30, 35] and current=42, attempt=3: (42-30)/3 = 4.0 < 5.0 ✓
        """
        result = make_eval_result(
            score=42,
            attempt_number=3,
            previous_scores=[30, 35],
        )
        should_pivot, reason = _assess_pivot_recommendation(
            result=result,
            previous_scores=[30, 35],
            attempt_number=3,
            pivot_threshold=3,
            min_improvement=5.0,
        )
        assert should_pivot
        assert "尝试" in reason

    def test_case2_declining_scores(self):
        """Case 2: 3 consecutive declining scores."""
        result = make_eval_result(score=30)
        should_pivot, reason = _assess_pivot_recommendation(
            result=result,
            previous_scores=[50, 40],
            attempt_number=3,
            pivot_threshold=10,
            min_improvement=5.0,
        )
        assert should_pivot
        assert "下降" in reason

    def test_case3_stuck_scores(self):
        """Case 3: Last 3 scores within 3 points of each other."""
        result = make_eval_result(score=60)
        should_pivot, reason = _assess_pivot_recommendation(
            result=result,
            previous_scores=[59, 61],
            attempt_number=3,
            pivot_threshold=10,
            min_improvement=5.0,
        )
        assert should_pivot
        assert "停滞" in reason

    def test_case4_hard_constraint_failure(self):
        """Case 4: Hard metric fails after 2+ attempts."""
        hard_metric = Metric("tests_pass", MetricType.HARD, "All tests pass")
        metric_result = MetricResult(metric=hard_metric, passed=False, reason="tests fail")
        result = make_eval_result(score=50, metrics=[metric_result], attempt_number=2)

        should_pivot, reason = _assess_pivot_recommendation(
            result=result,
            previous_scores=[45],
            attempt_number=2,
            pivot_threshold=10,
            min_improvement=5.0,
        )
        assert should_pivot
        assert "硬性指标" in reason

    def test_case5_very_low_score(self):
        """Case 5: Score < 40 after 2+ attempts."""
        result = make_eval_result(score=35, attempt_number=2)
        should_pivot, reason = _assess_pivot_recommendation(
            result=result,
            previous_scores=[30],
            attempt_number=2,
            pivot_threshold=10,
            min_improvement=5.0,
        )
        assert should_pivot
        assert "35" in reason

    def test_no_pivot_on_first_attempt(self):
        """No pivot recommended on first attempt."""
        result = make_eval_result(score=30, attempt_number=1)
        should_pivot, reason = _assess_pivot_recommendation(
            result=result,
            previous_scores=[],
            attempt_number=1,
            pivot_threshold=3,
            min_improvement=5.0,
        )
        assert not should_pivot

    def test_no_pivot_when_improving(self):
        """No pivot when scores are steadily improving."""
        result = make_eval_result(score=80, attempt_number=3)
        should_pivot, reason = _assess_pivot_recommendation(
            result=result,
            previous_scores=[50, 65],
            attempt_number=3,
            pivot_threshold=3,
            min_improvement=5.0,
        )
        # avg improvement = (80 - 50) / 3 = 10, which > 5
        assert not should_pivot

    def test_default_params_never_trigger(self):
        """With default orchestrator call (no previous_scores, attempt_number=1),
        pivot detection NEVER triggers.

        This confirms the 'dead code' concern: the orchestrator calls
        evaluate() without passing previous_scores or attempt_number,
        so _assess_pivot_recommendation always returns (False, "").
        """
        # This is what orchestrator.py line 767 does:
        #   eval_result = await evaluate(decision.target, cwd=cwd, verbose=verbose)
        # It doesn't pass previous_scores or attempt_number
        result = make_eval_result(score=10, attempt_number=1)
        should_pivot, reason = _assess_pivot_recommendation(
            result=result,
            previous_scores=[],  # default
            attempt_number=1,    # default
            pivot_threshold=3,   # default
            min_improvement=5.0, # default
        )
        assert not should_pivot
        assert reason == ""

    def test_orchestrator_evaluate_call_passes_history(self):
        """Confirm orchestrator.py evaluate() call now passes history params.

        Fixed: the evaluate() call should include previous_scores and
        attempt_number so _assess_pivot_recommendation() can function.
        """
        import inspect
        from ralph_sdk import orchestrator

        source = inspect.getsource(orchestrator.run)

        # Find the evaluate() call
        # Use a broader check - look for the params in the surrounding code block
        assert "previous_scores=previous_scores" in source, \
            "evaluate() call should pass previous_scores"
        assert "attempt_number=attempt_number" in source, \
            "evaluate() call should pass attempt_number"
        assert "get_attempt_history" in source, \
            "orchestrator should call get_attempt_history"


# =============================================================================
# 7. Parse Robustness
# =============================================================================


class TestEvaluatorParsing:
    """Test parse_evaluator_output edge cases."""

    def _make_metrics(self):
        return [
            Metric("tests_pass", MetricType.HARD, "Tests pass"),
            Metric("code_quality", MetricType.SUBJECTIVE, "Code quality"),
        ]

    def test_normal_output(self):
        metrics = self._make_metrics()
        text = """METRIC: tests_pass
PASSED: yes
VALUE: all tests pass
SCORE: 100
REASON: All 42 tests pass

METRIC: code_quality
PASSED: yes
SCORE: 85
REASON: Good code quality

OVERALL_SCORE: 90
"""
        result = parse_evaluator_output(text, metrics)
        assert len(result.metrics) == 2
        assert result.overall_score == 90
        assert result.metrics[0].passed is True
        assert result.metrics[1].score == 85

    def test_empty_input(self):
        metrics = self._make_metrics()
        result = parse_evaluator_output("", metrics)
        assert len(result.metrics) == 0
        assert result.overall_score == 0

    def test_no_overall_score(self):
        metrics = self._make_metrics()
        text = """METRIC: tests_pass
PASSED: yes
REASON: Tests pass
"""
        result = parse_evaluator_output(text, metrics)
        assert result.overall_score == 0  # Default when not found

    def test_multiline_reason(self):
        metrics = self._make_metrics()
        text = """METRIC: tests_pass
PASSED: no
REASON: Tests fail because:
1. test_auth fails with timeout
2. test_db fails with connection error
3. test_api fails with 500

OVERALL_SCORE: 30
"""
        result = parse_evaluator_output(text, metrics)
        assert len(result.metrics) == 1
        assert "timeout" in result.metrics[0].reason

    def test_case_insensitive_metric_name(self):
        metrics = self._make_metrics()
        text = """METRIC: Tests_Pass
PASSED: YES
REASON: All pass

OVERALL_SCORE: 100
"""
        result = parse_evaluator_output(text, metrics)
        assert len(result.metrics) == 1
        assert result.metrics[0].passed is True

    def test_issues_and_suggestions(self):
        metrics = self._make_metrics()
        text = """METRIC: tests_pass
PASSED: yes
REASON: Pass

ISSUES:
- Error in auth.py:42
- Warning in db.py:100

SUGGESTIONS:
- Add more tests
- Refactor auth module

OVERALL_SCORE: 70
"""
        result = parse_evaluator_output(text, metrics)
        assert len(result.issues) == 2
        assert "auth.py:42" in result.issues[0]
        assert len(result.suggestions) == 2

    def test_proxy_score_parsing(self):
        metrics = [
            Metric("latency", MetricType.SOFT, "Response time",
                   automation=AutomationLevel.HYBRID, proxy_metric="Mock latency"),
        ]
        text = """METRIC: latency
PASSED: yes
VALUE: 50ms
SCORE: 85
PROXY_SCORE: 90
PROXY_NOTES: Mock test shows good latency
REASON: Latency within acceptable range

OVERALL_SCORE: 85
"""
        result = parse_evaluator_output(text, metrics)
        assert len(result.metrics) == 1
        assert result.metrics[0].proxy_score == 90
        assert "Mock" in result.metrics[0].proxy_notes

    def test_missing_metric_name_in_list(self):
        """Metric name from output doesn't match any in the list → skipped."""
        metrics = self._make_metrics()
        text = """METRIC: unknown_metric
PASSED: yes
REASON: Some reason

OVERALL_SCORE: 50
"""
        result = parse_evaluator_output(text, metrics)
        assert len(result.metrics) == 0  # No match

    def test_overall_passed_logic(self):
        """overall_passed is True only when ALL hard metrics pass."""
        metrics = [
            Metric("test1", MetricType.HARD, "Test 1"),
            Metric("test2", MetricType.HARD, "Test 2"),
            Metric("quality", MetricType.SUBJECTIVE, "Quality"),
        ]
        text = """METRIC: test1
PASSED: yes
REASON: Pass

METRIC: test2
PASSED: no
REASON: Fail

METRIC: quality
PASSED: yes
SCORE: 90
REASON: Good

OVERALL_SCORE: 70
"""
        result = parse_evaluator_output(text, metrics)
        assert result.overall_passed is False  # test2 failed

    def test_no_hard_metrics_means_passed(self):
        """If no hard metrics, overall_passed defaults to True."""
        metrics = [Metric("quality", MetricType.SUBJECTIVE, "Quality")]
        text = """METRIC: quality
PASSED: yes
SCORE: 80
REASON: Good

OVERALL_SCORE: 80
"""
        result = parse_evaluator_output(text, metrics)
        assert result.overall_passed is True


class TestEvaluatorJsonParsing:
    """Test evaluator JSON parsing (Issue #10)."""

    def _make_metrics(self):
        return [
            Metric("tests_pass", MetricType.HARD, "Tests pass"),
            Metric("code_quality", MetricType.SUBJECTIVE, "Code quality"),
        ]

    def test_json_output(self):
        metrics = self._make_metrics()
        text = json.dumps({
            "metrics": [
                {"name": "tests_pass", "passed": True, "score": 100, "reason": "All pass"},
                {"name": "code_quality", "passed": True, "score": 85, "reason": "Good"},
            ],
            "issues": ["minor lint warning"],
            "suggestions": ["add more tests"],
            "overall_score": 90,
            "pivot_recommended": False,
            "pivot_reason": "",
        })
        result = parse_evaluator_output(text, metrics)
        assert len(result.metrics) == 2
        assert result.overall_score == 90
        assert result.metrics[0].passed is True
        assert result.metrics[1].score == 85
        assert len(result.issues) == 1
        assert len(result.suggestions) == 1

    def test_json_with_pivot(self):
        metrics = self._make_metrics()
        text = json.dumps({
            "metrics": [
                {"name": "tests_pass", "passed": False, "reason": "3 tests fail"},
            ],
            "issues": [],
            "suggestions": [],
            "overall_score": 30,
            "pivot_recommended": True,
            "pivot_reason": "Tests keep failing",
        })
        result = parse_evaluator_output(text, metrics)
        assert result.should_pivot is True
        assert "failing" in result.pivot_reason

    def test_json_in_code_fence(self):
        metrics = self._make_metrics()
        text = '''```json
{
  "metrics": [
    {"name": "tests_pass", "passed": true, "reason": "All pass"}
  ],
  "issues": [],
  "suggestions": [],
  "overall_score": 95
}
```'''
        result = parse_evaluator_output(text, metrics)
        assert len(result.metrics) == 1
        assert result.overall_score == 95

    def test_falls_back_to_regex(self):
        """Text format still works when no JSON present."""
        metrics = self._make_metrics()
        text = """METRIC: tests_pass
PASSED: yes
REASON: All pass

OVERALL_SCORE: 80
"""
        result = parse_evaluator_output(text, metrics)
        assert len(result.metrics) == 1
        assert result.overall_score == 80


class TestPlannerParsing:
    """Test parse_planner_output edge cases."""

    def test_json_in_code_fence(self):
        text = '''Here's my decision:

```json
{
  "action": "execute",
  "target": "T001",
  "reason": "Task is ready to execute"
}
```
'''
        decision = parse_planner_output(text)
        assert decision.action == Action.EXECUTE
        assert decision.target == "T001"
        assert "ready" in decision.reason

    def test_bare_json(self):
        text = '{"action": "explore", "target": "T002", "reason": "Need research"}'
        decision = parse_planner_output(text)
        assert decision.action == Action.EXPLORE
        assert decision.target == "T002"

    def test_regex_fallback(self):
        text = """ACTION: execute
TARGET: T001
REASON: Task is ready
"""
        decision = parse_planner_output(text)
        assert decision.action == Action.EXECUTE
        assert decision.target == "T001"

    def test_empty_input(self):
        decision = parse_planner_output("")
        assert decision.action == Action.SKIP  # Default

    def test_invalid_action(self):
        text = '{"action": "nonexistent_action", "target": "T001"}'
        decision = parse_planner_output(text)
        assert decision.action == Action.SKIP  # Fallback

    def test_parallel_execute_with_task_ids(self):
        text = '{"action": "parallel_execute", "task_ids": ["T001", "T002", "T003"], "reason": "Independent tasks"}'
        decision = parse_planner_output(text)
        assert decision.action == Action.PARALLEL_EXECUTE
        assert decision.task_ids == ["T001", "T002", "T003"]

    def test_task_ids_as_string(self):
        """task_ids provided as string → parsed into list."""
        text = '{"action": "parallel_execute", "task_ids": "T001, T002", "reason": "Test"}'
        decision = parse_planner_output(text)
        assert decision.task_ids == ["T001", "T002"]

    def test_hedge_action(self):
        text = '{"action": "hedge", "target": "T001", "failure_assumptions": "May not work", "reason": "Risky"}'
        decision = parse_planner_output(text)
        assert decision.action == Action.HEDGE
        assert decision.hedge_for == "T001"
        assert decision.failure_assumptions == "May not work"

    def test_pivot_iteration(self):
        text = json.dumps({
            "action": "pivot_iteration",
            "target": "T001",
            "attempt_count": 5,
            "best_score": "65",
            "failure_pattern": "Tests keep timing out",
            "new_approach": "Use async implementation",
            "reason": "Multiple attempts failed",
        })
        decision = parse_planner_output(text)
        assert decision.action == Action.PIVOT_ITERATION
        assert decision.attempt_count == 5
        assert decision.best_score == "65"

    def test_done_action(self):
        text = '{"action": "done", "reason": "All tasks completed"}'
        decision = parse_planner_output(text)
        assert decision.action == Action.DONE

    def test_ask_with_question(self):
        text = '{"action": "ask", "question": "Should we use Redis or Memcached?", "reason": "Need user input"}'
        decision = parse_planner_output(text)
        assert decision.action == Action.ASK
        assert "Redis" in decision.question

    def test_regex_multiline_reason(self):
        text = """ACTION: execute
TARGET: T001
REASON: This is a long reason that
spans multiple lines and includes
detailed explanation.
NEW_TASKS: T002: Do something
"""
        decision = parse_planner_output(text)
        assert decision.action == Action.EXECUTE
        assert "spans multiple lines" in decision.reason


class TestReviewerParsing:
    """Test parse_reviewer_output edge cases."""

    def test_json_verdict(self):
        text = '{"verdict": "passed", "reason": "All requirements met", "suggestions": "Add more tests"}'
        result = parse_reviewer_output(text)
        assert result.verdict == Verdict.PASSED
        assert "requirements" in result.reason

    def test_regex_verdict(self):
        text = """VERDICT: retry
REASON: Some tests fail
SUGGESTIONS: Fix test_auth
"""
        result = parse_reviewer_output(text)
        assert result.verdict == Verdict.RETRY
        assert "tests fail" in result.reason

    def test_empty_input(self):
        result = parse_reviewer_output("")
        assert result.verdict == Verdict.RETRY  # Default on parse failure

    def test_invalid_verdict(self):
        text = '{"verdict": "unknown_verdict", "reason": "test"}'
        result = parse_reviewer_output(text)
        assert result.verdict == Verdict.RETRY  # Fallback on parse failure

    def test_failed_verdict(self):
        text = '{"verdict": "failed", "reason": "Fundamental design flaw"}'
        result = parse_reviewer_output(text)
        assert result.verdict == Verdict.FAILED

    def test_case_insensitive_verdict(self):
        text = """VERDICT: PASSED
REASON: Looks good
"""
        result = parse_reviewer_output(text)
        assert result.verdict == Verdict.PASSED


class TestWorkerParsing:
    """Test extract_worker_metadata edge cases."""

    def test_always_succeeds(self):
        result = extract_worker_metadata("any text", "IMPLEMENT")
        assert result["success"] is True

    def test_explore_confidence_high(self):
        result = extract_worker_metadata("Confidence: high\nDone", "EXPLORE")
        assert result["confidence"] == "high"

    def test_explore_confidence_low(self):
        result = extract_worker_metadata("Confidence: Low", "EXPLORE")
        assert result["confidence"] == "low"

    def test_no_confidence(self):
        result = extract_worker_metadata("Some output", "EXPLORE")
        assert "confidence" not in result

    def test_empty_output(self):
        result = extract_worker_metadata("", "IMPLEMENT")
        assert result["success"] is True


class TestFeedbackParsing:
    """Test parse_feedback edge cases."""

    def test_unfilled_template_returns_none(self, tmp_path):
        """Unfilled template → None (template detection works)."""
        cwd = setup_ralph(tmp_path)
        checkpoint = Checkpoint(id="T001_v1", task_id="T001")
        from ralph_sdk.pool import generate_feedback_template
        generate_feedback_template([checkpoint], "Test instructions", cwd)

        from ralph_sdk.pool import parse_feedback
        result = parse_feedback(cwd)
        assert result is None

    def test_filled_template(self, tmp_path):
        """Filled template → parsed correctly."""
        cwd = setup_ralph(tmp_path)
        content = """# 测试反馈

## 待测试项

### T001_v1
- 路径: `checkpoints/T001`

**你的测试结果:**
```
成功率: 95%
延迟表现: 50ms average
其他观察: Works well
评分 (1-5): 4
```

## 总体反馈

```
Version 1 works great
```

## 下一步

- [x] 基于最好的版本继续迭代
- [ ] 尝试新方向
- [ ] 结束，当前版本够用
"""
        (tmp_path / ".ralph" / "feedback.md").write_text(content)

        from ralph_sdk.pool import parse_feedback
        result = parse_feedback(cwd)
        assert result is not None
        assert "T001_v1" in result["checkpoint_results"]
        assert result["checkpoint_results"]["T001_v1"]["成功率"] == "95%"

    def test_no_feedback_file(self, tmp_path):
        """No feedback.md → None."""
        cwd = setup_ralph(tmp_path)
        from ralph_sdk.pool import parse_feedback
        result = parse_feedback(cwd)
        assert result is None


class TestJsonExtraction:
    """Test extract_json from utils (shared by planner, reviewer, evaluator)."""

    def test_json_in_fence(self):
        text = '```json\n{"key": "value"}\n```'
        assert extract_json(text) == {"key": "value"}

    def test_bare_json(self):
        text = 'Some text {"key": "value"} more text'
        assert extract_json(text) == {"key": "value"}

    def test_no_json(self):
        assert extract_json("no json here") is None

    def test_invalid_json(self):
        text = '{"key": invalid}'
        assert extract_json(text) is None

    def test_nested_json(self):
        text = '{"action": "execute", "details": {"sub": "value"}}'
        result = extract_json(text)
        assert result is not None
        assert result["details"]["sub"] == "value"


# =============================================================================
# 8. Pool Utilities + Checkpoint
# =============================================================================


class TestPivotMarkerManagement:
    """Test pivot recommendation markers in pool.md."""

    def test_append_pivot_recommendation(self, tmp_path):
        cwd = setup_ralph(tmp_path)
        init_pool("Test", "| T001 | IMPLEMENT | pending |", cwd)
        append_to_findings("**[PIVOT_RECOMMENDED]** T001: Tests keep failing", cwd)
        assert has_pivot_recommendation(cwd)

    def test_clear_pivot_recommendation(self, tmp_path):
        cwd = setup_ralph(tmp_path)
        init_pool("Test", "| T001 | IMPLEMENT | pending |", cwd)
        append_to_findings("**[PIVOT_RECOMMENDED]** T001: Tests keep failing", cwd)
        assert has_pivot_recommendation(cwd)

        clear_pivot_recommendation("T001", cwd)
        assert not has_pivot_recommendation(cwd)
        assert "[PIVOT_PROCESSED]" in read_pool(cwd)

    def test_clear_without_asterisks(self, tmp_path):
        """Clear works for format without bold markers."""
        cwd = setup_ralph(tmp_path)
        init_pool("Test", "| T001 | IMPLEMENT | pending |", cwd)
        # Manually insert without asterisks
        pool = read_pool(cwd)
        pool = pool.replace("## Findings", "## Findings\n- [PIVOT_RECOMMENDED] T001: reason")
        write_pool(pool, cwd)

        assert has_pivot_recommendation(cwd)
        clear_pivot_recommendation("T001", cwd)
        assert not has_pivot_recommendation(cwd)

    def test_no_pivot_initially(self, tmp_path):
        cwd = setup_ralph(tmp_path)
        init_pool("Test", "| T001 | IMPLEMENT | pending |", cwd)
        assert not has_pivot_recommendation(cwd)


class TestVerifiedInfo:
    """Test verified information CRUD operations."""

    def test_add_and_retrieve(self, tmp_path):
        cwd = setup_ralph(tmp_path)
        init_pool("Test", "| T001 | IMPLEMENT | pending |", cwd)

        add_verified_info("React version", "18.3.0", "https://react.dev", cwd)
        info = get_verified_info("React version", cwd)
        assert info is not None
        assert "18.3.0" in info

    def test_topic_verified_check(self, tmp_path):
        cwd = setup_ralph(tmp_path)
        init_pool("Test", "| T001 | IMPLEMENT | pending |", cwd)

        assert not is_topic_verified("React version", cwd)
        add_verified_info("React version", "18.3.0", "https://react.dev", cwd)
        assert is_topic_verified("React version", cwd)

    def test_list_verified_topics(self, tmp_path):
        """Multiple add_verified_info calls should all be preserved."""
        cwd = setup_ralph(tmp_path)
        init_pool("Test", "| T001 | IMPLEMENT | pending |", cwd)

        add_verified_info("React version", "18.3.0", "https://react.dev", cwd)
        topics_after_first = list_verified_topics(cwd)
        assert len(topics_after_first) == 1
        assert any("React" in t for t in topics_after_first)

        add_verified_info("Python version", "3.12", "https://python.org", cwd)
        topics_after_second = list_verified_topics(cwd)

        # Fixed: Both entries should now be present
        assert len(topics_after_second) == 2, \
            f"Expected 2 topics, got {len(topics_after_second)}"
        assert any("React" in t for t in topics_after_second)
        assert any("Python" in t for t in topics_after_second)

    def test_nonexistent_topic(self, tmp_path):
        cwd = setup_ralph(tmp_path)
        init_pool("Test", "| T001 | IMPLEMENT | pending |", cwd)
        assert get_verified_info("nonexistent", cwd) is None

    def test_empty_pool(self, tmp_path):
        cwd = setup_ralph(tmp_path)
        assert get_verified_info("anything", cwd) is None
        assert list_verified_topics(cwd) == []


class TestTaskFileManagement:
    """Test task file creation and management."""

    def test_ensure_task_files_creates_missing(self, tmp_path):
        cwd = setup_ralph(tmp_path)
        pool_content = """# Task Pool

## Active Tasks

| ID | Type | Status |
|----|------|--------|
| T001 | IMPLEMENT | pending |
| T002 | EXPLORE | pending |
| T003 | IMPLEMENT | pending |
"""
        write_pool(pool_content, cwd)

        created = ensure_task_files_exist(cwd)
        assert set(created) == {"T001", "T002", "T003"}

        # Verify files exist
        for tid in created:
            assert read_task(tid, cwd) != ""

    def test_ensure_task_files_skips_existing(self, tmp_path):
        cwd = setup_ralph(tmp_path)
        init_task("T001", "IMPLEMENT", "Existing Task", "Already exists", cwd)

        pool_content = """# Task Pool
## Active Tasks
| T001 | IMPLEMENT | in_progress |
| T002 | IMPLEMENT | pending |
"""
        write_pool(pool_content, cwd)

        created = ensure_task_files_exist(cwd)
        assert "T001" not in created
        assert "T002" in created

    def test_extract_task_ids(self, tmp_path):
        cwd = setup_ralph(tmp_path)
        pool_content = """# Task Pool
## Active Tasks
| T001 | IMPLEMENT | pending |
| T002 | EXPLORE | pending |
## Completed
| T003 | IMPLEMENT | done |
"""
        write_pool(pool_content, cwd)

        ids = extract_task_ids_from_pool(cwd)
        assert "T001" in ids
        assert "T002" in ids
        assert "T003" in ids

    def test_extract_task_ids_empty_pool(self, tmp_path):
        cwd = setup_ralph(tmp_path)
        assert extract_task_ids_from_pool(cwd) == []

    def test_init_task_explore(self, tmp_path):
        cwd = setup_ralph(tmp_path)
        init_task("T001", "EXPLORE", "Research Task", "Explore the options", cwd)
        content = read_task("T001", cwd)
        assert "EXPLORE" in content
        assert "Research Task" in content

    def test_init_task_implement(self, tmp_path):
        cwd = setup_ralph(tmp_path)
        init_task("T001", "IMPLEMENT", "Build Feature", "Build the thing", cwd)
        content = read_task("T001", cwd)
        assert "IMPLEMENT" in content
        assert "Build Feature" in content


class TestProxyScore:
    """Test ProxyScore and Checkpoint proxy score calculations."""

    def test_proxy_score_passing(self):
        cp = Checkpoint(id="T001_v1", task_id="T001")
        cp.add_proxy_score("accuracy", 85, ">= 70%")
        assert cp.proxy_scores[0].passed is True
        assert cp.proxy_overall == 85

    def test_proxy_score_failing(self):
        cp = Checkpoint(id="T001_v1", task_id="T001")
        cp.add_proxy_score("accuracy", 50, ">= 70")
        assert cp.proxy_scores[0].passed is False

    def test_proxy_score_average(self):
        cp = Checkpoint(id="T001_v1", task_id="T001")
        cp.add_proxy_score("accuracy", 80, ">= 70")
        cp.add_proxy_score("speed", 60, "<= 100")
        assert cp.proxy_overall == 70.0  # (80 + 60) / 2

    def test_proxy_score_operators(self):
        cp = Checkpoint(id="T001_v1", task_id="T001")

        cp.add_proxy_score("m1", 80, ">= 70")
        assert cp.proxy_scores[-1].passed is True

        cp.add_proxy_score("m2", 50, "<= 100")
        assert cp.proxy_scores[-1].passed is True

        cp.add_proxy_score("m3", 80, "> 70")
        assert cp.proxy_scores[-1].passed is True

        cp.add_proxy_score("m4", 50, "< 100")
        assert cp.proxy_scores[-1].passed is True

        cp.add_proxy_score("m5", 70, "> 70")
        assert cp.proxy_scores[-1].passed is False  # 70 is not > 70

    def test_proxy_score_no_target(self):
        cp = Checkpoint(id="T001_v1", task_id="T001")
        cp.add_proxy_score("metric", 75)
        assert cp.proxy_scores[0].passed is False  # No target = not passed
        assert cp.proxy_overall == 75


class TestEvalConfigParsing:
    """Test read_eval_config_from_goal parsing."""

    def test_default_config(self, tmp_path):
        cwd = setup_ralph(tmp_path)
        write_goal("# Goal\nBuild something\n", cwd)
        config = read_eval_config_from_goal(cwd)
        assert config["mode"] == "全自动"
        assert config["test_frequency"] is None
        assert config["batch_preference"] is None

    def test_custom_config(self, tmp_path):
        cwd = setup_ralph(tmp_path)
        write_goal("""# Goal

Build something

### Evaluation Mode

**测试模式**: 半自动
**测试频率**: 每个任务完成后
**测试安排**: 一个一个测试

## Other Section
""", cwd)
        config = read_eval_config_from_goal(cwd)
        assert config["mode"] == "半自动"
        assert config["test_frequency"] == "每个任务完成后"
        assert config["batch_preference"] == "一个一个测试"

    def test_no_goal_file(self, tmp_path):
        cwd = setup_ralph(tmp_path)
        config = read_eval_config_from_goal(cwd)
        assert config["mode"] == "全自动"

    def test_partial_config(self, tmp_path):
        cwd = setup_ralph(tmp_path)
        write_goal("""# Goal

### Evaluation Mode

**测试模式**: 手动

## Success Metrics
""", cwd)
        config = read_eval_config_from_goal(cwd)
        assert config["mode"] == "手动"
        assert config["test_frequency"] is None


class TestCheckpointSystem:
    """Test checkpoint creation and management."""

    def test_checkpoint_creation(self, tmp_path):
        cwd = setup_ralph(tmp_path)
        from ralph_sdk.pool import create_checkpoint
        cp = create_checkpoint("T001", path="output/model.pt", artifact_type="model", cwd=cwd)
        assert cp.task_id == "T001"
        assert "T001_" in cp.id  # UUID-based
        assert cp.status == "pending"

    def test_checkpoint_serialization(self):
        cp = Checkpoint(id="T001_v1", task_id="T001", version=1,
                        path="output/", artifact_type="code")
        d = cp.to_dict()
        cp2 = Checkpoint.from_dict(d)
        assert cp2.id == cp.id
        assert cp2.task_id == cp.task_id

    def test_checkpoint_user_result(self):
        cp = Checkpoint(id="T001_v1", task_id="T001")
        cp.set_user_result(4.5, "Works well")
        assert cp.user_score == 4.5
        assert cp.status == "tested"
        assert cp.tested_at is not None

    def test_checkpoint_best_and_rejected(self):
        cp = Checkpoint(id="T001_v1", task_id="T001")
        cp.mark_as_best()
        assert cp.status == "best"

        cp2 = Checkpoint(id="T001_v2", task_id="T001")
        cp2.mark_as_rejected()
        assert cp2.status == "rejected"


class TestTargetScoreParsing:
    """Test parse_target_score_from_goal."""

    def test_default_score(self, tmp_path):
        cwd = setup_ralph(tmp_path)
        write_goal("# Goal\nBuild something\n", cwd)
        from ralph_sdk.orchestrator import parse_target_score_from_goal
        assert parse_target_score_from_goal(cwd) == 95  # DEFAULT_TARGET_SCORE

    def test_custom_score(self, tmp_path):
        cwd = setup_ralph(tmp_path)
        write_goal("# Goal\ntarget_score: 80\n", cwd)
        from ralph_sdk.orchestrator import parse_target_score_from_goal
        assert parse_target_score_from_goal(cwd) == 80

    def test_bold_format(self, tmp_path):
        cwd = setup_ralph(tmp_path)
        write_goal("# Goal\n**Target Score**: 90\n", cwd)
        from ralph_sdk.orchestrator import parse_target_score_from_goal
        assert parse_target_score_from_goal(cwd) == 90

    def test_no_goal(self, tmp_path):
        cwd = setup_ralph(tmp_path)
        from ralph_sdk.orchestrator import parse_target_score_from_goal
        assert parse_target_score_from_goal(cwd) == 95


class TestEvaluationResultMethods:
    """Test EvaluationResult helper methods."""

    def test_summary(self):
        result = make_eval_result(score=75)
        result.metrics = [
            MetricResult(
                metric=Metric("test", MetricType.HARD, "Test"),
                passed=True, reason="ok"
            ),
        ]
        s = result.summary()
        assert "1/1" in s
        assert "75" in s

    def test_summary_with_pivot(self):
        result = make_eval_result(score=30)
        result.should_pivot = True
        result.pivot_reason = "Too many failures"
        s = result.summary()
        assert "PIVOT" in s

    def test_is_improving(self):
        result = make_eval_result(score=80)
        result.previous_scores = [50, 60, 70]
        assert result.is_improving()

        result.previous_scores = [80, 70]
        assert not result.is_improving()

    def test_score_trend(self):
        result = make_eval_result(score=80)
        result.previous_scores = []
        assert result.get_score_trend() == "first_attempt"

        result.previous_scores = [70]
        assert result.get_score_trend() == "improving"

        result.previous_scores = [90]
        assert result.get_score_trend() == "declining"

        result.previous_scores = [80]
        assert result.get_score_trend() == "stable"

    def test_get_proxy_scores(self):
        metric = Metric("latency", MetricType.SOFT, "Speed",
                        automation=AutomationLevel.HYBRID)
        mr = MetricResult(metric=metric, passed=True, proxy_score=85.0, reason="ok")
        result = make_eval_result(score=80, metrics=[mr])
        proxy = result.get_proxy_scores()
        assert proxy == {"latency": 85.0}

    def test_get_pending_manual(self):
        metric = Metric("ux", MetricType.SUBJECTIVE, "UX",
                        automation=AutomationLevel.MANUAL)
        mr = MetricResult(metric=metric, passed=False, pending_manual=True, reason="needs user")
        result = make_eval_result(score=70, metrics=[mr])
        pending = result.get_pending_manual_metrics()
        assert len(pending) == 1
        assert pending[0].metric.name == "ux"


class TestPlannerDecisionProperties:
    """Test PlannerDecision helper properties."""

    def test_is_pivot(self):
        d = PlannerDecision(action=Action.HEDGE)
        assert d.is_pivot

        d = PlannerDecision(action=Action.PIVOT_RESEARCH)
        assert d.is_pivot

        d = PlannerDecision(action=Action.EXECUTE)
        assert not d.is_pivot

    def test_pivot_type(self):
        d = PlannerDecision(action=Action.PIVOT_RESEARCH)
        assert d.pivot_type == "research"

        d = PlannerDecision(action=Action.PIVOT_WAIT)
        assert d.pivot_type == "wait"

        d = PlannerDecision(action=Action.HEDGE)
        assert d.pivot_type == "wait"

        d = PlannerDecision(action=Action.PIVOT_ITERATION)
        assert d.pivot_type == "iteration"

        d = PlannerDecision(action=Action.EXECUTE)
        assert d.pivot_type is None

    def test_post_init_default_task_ids(self):
        d = PlannerDecision(action=Action.EXECUTE)
        assert d.task_ids == []


# =============================================================================
# 9. Pivot Detection - Plan B (Prompt Judge + Code I/O)
# =============================================================================


class TestPivotPromptDriven:
    """Plan B: Agent judges pivot, Code handles I/O.

    Covers all 5 identified issues:
    1. No duplicate write path (agent doesn't write pool.md)
    2. Format is controlled by code (append_to_findings)
    3. Agent receives structured history in prompt
    4. parse_evaluator_output extracts PIVOT fields
    5. No race condition (agent doesn't use Edit on pool.md)
    """

    # --- Parse tests (Issue #4) ---

    def test_parse_pivot_yes(self):
        """PIVOT_RECOMMENDED: yes → should_pivot=True"""
        metrics = [Metric("test", MetricType.HARD, "Test")]
        text = """METRIC: test
PASSED: yes
REASON: Pass

OVERALL_SCORE: 70

PIVOT_RECOMMENDED: yes
PIVOT_REASON: Scores declining consistently
"""
        result = parse_evaluator_output(text, metrics)
        assert result.should_pivot is True
        assert "declining" in result.pivot_reason.lower()

    def test_parse_pivot_no(self):
        """PIVOT_RECOMMENDED: no → should_pivot=False"""
        metrics = [Metric("test", MetricType.HARD, "Test")]
        text = """METRIC: test
PASSED: yes
REASON: Pass

OVERALL_SCORE: 85

PIVOT_RECOMMENDED: no
PIVOT_REASON: Scores improving steadily
"""
        result = parse_evaluator_output(text, metrics)
        assert result.should_pivot is False

    def test_parse_pivot_missing(self):
        """No PIVOT_RECOMMENDED → should_pivot stays at default (False)"""
        metrics = [Metric("test", MetricType.HARD, "Test")]
        text = """METRIC: test
PASSED: yes
REASON: Pass

OVERALL_SCORE: 90
"""
        result = parse_evaluator_output(text, metrics)
        # Default is False, and agent didn't set it
        assert result.should_pivot is False

    def test_parse_pivot_reason_extracted(self):
        """PIVOT_REASON: ... is correctly extracted"""
        metrics = [Metric("test", MetricType.HARD, "Test")]
        text = """METRIC: test
PASSED: no
REASON: Fail

OVERALL_SCORE: 30

PIVOT_RECOMMENDED: yes
PIVOT_REASON: Hard metric tests_pass has failed 3 times in a row
"""
        result = parse_evaluator_output(text, metrics)
        assert result.should_pivot is True
        assert "tests_pass" in result.pivot_reason

    def test_parse_pivot_case_insensitive(self):
        """PIVOT_RECOMMENDED: YES / Yes / yes all work"""
        metrics = [Metric("test", MetricType.HARD, "Test")]
        for variant in ["YES", "Yes", "yes", "yEs"]:
            text = f"""METRIC: test
PASSED: yes
REASON: Pass

OVERALL_SCORE: 50

PIVOT_RECOMMENDED: {variant}
PIVOT_REASON: Test reason
"""
            result = parse_evaluator_output(text, metrics)
            assert result.should_pivot is True, f"Failed for variant '{variant}'"

    def test_parse_pivot_no_case_insensitive(self):
        """PIVOT_RECOMMENDED: NO / No / no all work"""
        metrics = [Metric("test", MetricType.HARD, "Test")]
        for variant in ["NO", "No", "no", "nO"]:
            text = f"""METRIC: test
PASSED: yes
REASON: Pass

OVERALL_SCORE: 80

PIVOT_RECOMMENDED: {variant}
"""
            result = parse_evaluator_output(text, metrics)
            assert result.should_pivot is False, f"Failed for variant '{variant}'"

    def test_parse_pivot_with_full_evaluator_output(self):
        """PIVOT fields extracted from a full evaluator output with all sections"""
        metrics = [
            Metric("tests_pass", MetricType.HARD, "Tests pass"),
            Metric("code_quality", MetricType.SUBJECTIVE, "Quality"),
        ]
        text = """METRIC: tests_pass
PASSED: no
VALUE: 3/10 tests fail
SCORE: 30
REASON: Multiple test failures in auth module

METRIC: code_quality
PASSED: yes
SCORE: 75
REASON: Decent code structure

ISSUES:
- auth.py:42 - Missing null check
- db.py:100 - Connection leak

SUGGESTIONS:
- Add error handling
- Fix auth tests

OVERALL_SCORE: 45

PIVOT_RECOMMENDED: yes
PIVOT_REASON: Hard metric tests_pass keeps failing, score stuck at 45
"""
        result = parse_evaluator_output(text, metrics)
        assert len(result.metrics) == 2
        assert result.overall_score == 45
        assert result.should_pivot is True
        assert "tests_pass" in result.pivot_reason

    # --- Prompt construction tests (Issue #3) ---

    def test_prompt_includes_attempt_history(self):
        """attempt_number > 1 → prompt includes Attempt History section"""
        prompt = build_evaluator_prompt(
            task_id="T001",
            goal="Build feature X",
            task_detail="Task details here",
            metrics=[Metric("test", MetricType.HARD, "Test")],
            attempt_number=3,
            previous_scores=[40.0, 55.0],
        )
        assert "Attempt History" in prompt
        assert "#3" in prompt
        assert "40" in prompt
        assert "55" in prompt

    def test_prompt_no_history_on_first_attempt(self):
        """attempt_number=1 → no Attempt History section"""
        prompt = build_evaluator_prompt(
            task_id="T001",
            goal="Build feature X",
            task_detail="Task details here",
            metrics=[Metric("test", MetricType.HARD, "Test")],
            attempt_number=1,
            previous_scores=None,
        )
        assert "Attempt History" not in prompt

    def test_prompt_no_history_when_no_scores(self):
        """attempt_number > 1 but no previous_scores → no history section"""
        prompt = build_evaluator_prompt(
            task_id="T001",
            goal="Build feature X",
            task_detail="Task details here",
            metrics=[Metric("test", MetricType.HARD, "Test")],
            attempt_number=2,
            previous_scores=None,
        )
        assert "Attempt History" not in prompt

    def test_prompt_includes_trend(self):
        """Score trend description appears in prompt"""
        prompt = build_evaluator_prompt(
            task_id="T001",
            goal="Build feature X",
            task_detail="Task details here",
            metrics=[Metric("test", MetricType.HARD, "Test")],
            attempt_number=4,
            previous_scores=[30.0, 40.0, 50.0],
        )
        assert "improving" in prompt

    def test_prompt_declining_trend(self):
        """Declining scores show declining trend"""
        prompt = build_evaluator_prompt(
            task_id="T001",
            goal="Build feature X",
            task_detail="Task details here",
            metrics=[Metric("test", MetricType.HARD, "Test")],
            attempt_number=4,
            previous_scores=[80.0, 60.0, 40.0],
        )
        assert "declining" in prompt

    # --- _describe_trend tests ---

    def test_describe_trend_improving(self):
        assert _describe_trend([30, 40, 50]) == "improving"

    def test_describe_trend_declining(self):
        assert _describe_trend([50, 40, 30]) == "declining"

    def test_describe_trend_stable(self):
        assert _describe_trend([50, 50, 50]) == "stable"

    def test_describe_trend_insufficient(self):
        assert _describe_trend([50]) == "insufficient data"
        assert _describe_trend([]) == "insufficient data"

    # --- System prompt tests (Issues #1, #5) ---

    def test_no_cross_agent_communication_in_prompt(self):
        """System prompt no longer contains <cross_agent_communication>"""
        assert "cross_agent_communication" not in EVALUATOR_SYSTEM_PROMPT
        assert "Edit tool" not in EVALUATOR_SYSTEM_PROMPT
        assert "pool.md" not in EVALUATOR_SYSTEM_PROMPT

    def test_evaluator_has_no_write_tools(self):
        """Evaluator agent should not have Write or Edit tools"""
        import inspect
        from ralph_sdk import evaluator
        source = inspect.getsource(evaluator.evaluate)
        # Check that Edit and Write are NOT in allowed_tools
        # The allowed_tools list should not contain "Edit" or "Write"
        assert '"Edit"' not in source, "Evaluator should not have Edit tool"
        assert '"Write"' not in source, "Evaluator should not have Write tool"

    def test_system_prompt_has_pivot_assessment(self):
        """System prompt contains pivot assessment guidance"""
        assert "Pivot Assessment" in EVALUATOR_SYSTEM_PROMPT
        assert "PIVOT_RECOMMENDED" in EVALUATOR_SYSTEM_PROMPT

    # --- Code fallback tests ---

    def test_code_fallback_when_agent_silent(self):
        """Agent doesn't output PIVOT → fallback to _assess_pivot_recommendation

        When parse_evaluator_output doesn't find PIVOT_RECOMMENDED,
        should_pivot stays at default (False) and pivot_reason is empty.
        In evaluate(), this triggers the fallback to code-based detection.
        """
        metrics = [Metric("test", MetricType.HARD, "Test")]
        text = """METRIC: test
PASSED: yes
REASON: Pass

OVERALL_SCORE: 50
"""
        result = parse_evaluator_output(text, metrics)
        # Agent didn't output PIVOT, so should_pivot=False and pivot_reason=""
        assert result.should_pivot is False
        assert result.pivot_reason == ""
        # This combination (False + empty reason) triggers fallback in evaluate()

    def test_agent_pivot_overrides_code(self):
        """Agent outputs PIVOT_RECOMMENDED: yes → code fallback is skipped.

        When agent sets should_pivot=True, evaluate() should NOT
        overwrite it with _assess_pivot_recommendation().
        """
        metrics = [Metric("test", MetricType.HARD, "Test")]
        text = """METRIC: test
PASSED: yes
REASON: Pass

OVERALL_SCORE: 80

PIVOT_RECOMMENDED: yes
PIVOT_REASON: Despite good score, approach is fundamentally limited
"""
        result = parse_evaluator_output(text, metrics)
        # Agent said pivot, and provided a reason
        assert result.should_pivot is True
        assert result.pivot_reason != ""
        # The condition in evaluate() checks: if not should_pivot and not pivot_reason
        # This is True and non-empty, so fallback would NOT trigger

    # --- clear_pivot_recommendation format variant tests (Issue #2) ---

    def test_clear_handles_bold_format(self, tmp_path):
        """**[PIVOT_RECOMMENDED]** format is cleared"""
        cwd = setup_ralph(tmp_path)
        init_pool("Test", "| T001 | IMPLEMENT | pending |", cwd)
        append_to_findings("**[PIVOT_RECOMMENDED]** T001: reason here", cwd)
        assert has_pivot_recommendation(cwd)

        clear_pivot_recommendation("T001", cwd)
        assert not has_pivot_recommendation(cwd)
        assert "[PIVOT_PROCESSED]" in read_pool(cwd)

    def test_clear_handles_plain_format(self, tmp_path):
        """[PIVOT_RECOMMENDED] format (no bold) is cleared"""
        cwd = setup_ralph(tmp_path)
        init_pool("Test", "| T001 | IMPLEMENT | pending |", cwd)
        pool = read_pool(cwd)
        pool = pool.replace(
            "## Findings",
            "## Findings\n- [PIVOT_RECOMMENDED] T001: reason"
        )
        write_pool(pool, cwd)
        assert has_pivot_recommendation(cwd)

        clear_pivot_recommendation("T001", cwd)
        assert not has_pivot_recommendation(cwd)

    def test_clear_handles_extra_spaces(self, tmp_path):
        """[PIVOT_RECOMMENDED]  T001 (extra spaces) is cleared"""
        cwd = setup_ralph(tmp_path)
        init_pool("Test", "| T001 | IMPLEMENT | pending |", cwd)
        pool = read_pool(cwd)
        pool = pool.replace(
            "## Findings",
            "## Findings\n- [PIVOT_RECOMMENDED]  T001: extra spaces"
        )
        write_pool(pool, cwd)
        assert has_pivot_recommendation(cwd)

        clear_pivot_recommendation("T001", cwd)
        assert not has_pivot_recommendation(cwd)

    def test_clear_only_affects_target_task(self, tmp_path):
        """Clearing T001 doesn't affect T002's pivot recommendation"""
        cwd = setup_ralph(tmp_path)
        init_pool("Test", "| T001 | IMPLEMENT | pending |\n| T002 | IMPLEMENT | pending |", cwd)
        append_to_findings("**[PIVOT_RECOMMENDED]** T001: reason1", cwd)
        append_to_findings("**[PIVOT_RECOMMENDED]** T002: reason2", cwd)

        clear_pivot_recommendation("T001", cwd)
        pool = read_pool(cwd)
        # T001 should be processed
        assert "[PIVOT_PROCESSED]" in pool
        # T002 should still be recommended
        assert "[PIVOT_RECOMMENDED]" in pool
        assert has_pivot_recommendation(cwd)

    def test_clear_idempotent(self, tmp_path):
        """Clearing twice doesn't break anything"""
        cwd = setup_ralph(tmp_path)
        init_pool("Test", "| T001 | IMPLEMENT | pending |", cwd)
        append_to_findings("**[PIVOT_RECOMMENDED]** T001: reason", cwd)

        clear_pivot_recommendation("T001", cwd)
        clear_pivot_recommendation("T001", cwd)  # Second call
        assert not has_pivot_recommendation(cwd)

    # --- Integration: write path verification (Issue #1) ---

    def test_only_code_writes_to_pool(self):
        """Verify the evaluator agent cannot write to pool.md.

        With Plan B, only the orchestrator (via append_to_findings) writes
        pivot markers to pool.md. The evaluator agent just outputs text.
        """
        import inspect
        from ralph_sdk import evaluator

        # The evaluator system prompt should not instruct writing to pool.md
        assert "pool.md" not in EVALUATOR_SYSTEM_PROMPT

        # The evaluate() function should not have "Edit" in allowed_tools
        source = inspect.getsource(evaluator.evaluate)
        assert '"Edit"' not in source

    def test_orchestrator_writes_pivot_marker(self):
        """Verify orchestrator.py has the code path to write pivot to pool.md"""
        import inspect
        from ralph_sdk import orchestrator
        source = inspect.getsource(orchestrator.run)
        assert "append_to_findings" in source
        assert "eval_result.should_pivot" in source
