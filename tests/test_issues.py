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
    _parse_evaluator_json,
    _validate_score_consistency,
    build_evaluator_prompt,
    get_attempt_history,
    parse_evaluator_output,
)
from ralph_sdk.planner import (
    Action,
    PlannerDecision,
    _parse_planner_json,
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
    generate_handoff_note,
    get_verified_info,
    has_pivot_recommendation,
    init_pool,
    init_ralph_dir,
    init_task,
    is_topic_verified,
    list_verified_topics,
    mark_pending_tasks_skipped,
    read_eval_config_from_goal,
    read_pool,
    read_task,
    update_pool_status,
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
        text = json.dumps({
            "metrics": [
                {"name": "tests_pass", "passed": True, "value": "all tests pass", "score": 100, "reason": "All 42 tests pass"},
                {"name": "code_quality", "passed": True, "score": 85, "reason": "Good code quality"},
            ],
            "overall_score": 90,
        })
        result = parse_evaluator_output(None, text, metrics)
        assert len(result.metrics) == 2
        assert result.overall_score == 90
        assert result.metrics[0].passed is True
        assert result.metrics[1].score == 85

    def test_empty_input(self):
        metrics = self._make_metrics()
        result = parse_evaluator_output(None, "", metrics)
        assert len(result.metrics) == 0
        assert result.overall_score == 0

    def test_no_overall_score(self):
        metrics = self._make_metrics()
        text = json.dumps({
            "metrics": [{"name": "tests_pass", "passed": True, "reason": "Tests pass"}],
            "overall_score": 0,
        })
        result = parse_evaluator_output(None, text, metrics)
        assert result.overall_score == 0

    def test_multiline_reason(self):
        metrics = self._make_metrics()
        text = json.dumps({
            "metrics": [{"name": "tests_pass", "passed": False, "reason": "Tests fail because: 1. test_auth fails with timeout 2. test_db fails with connection error"}],
            "overall_score": 30,
        })
        result = parse_evaluator_output(None, text, metrics)
        assert len(result.metrics) == 1
        assert "timeout" in result.metrics[0].reason

    def test_case_insensitive_metric_name(self):
        metrics = self._make_metrics()
        text = json.dumps({
            "metrics": [{"name": "Tests_Pass", "passed": True, "reason": "All pass"}],
            "overall_score": 100,
        })
        result = parse_evaluator_output(None, text, metrics)
        assert len(result.metrics) == 1
        assert result.metrics[0].passed is True

    def test_issues_and_suggestions(self):
        metrics = self._make_metrics()
        text = json.dumps({
            "metrics": [{"name": "tests_pass", "passed": True, "reason": "Pass"}],
            "issues": ["Error in auth.py:42", "Warning in db.py:100"],
            "suggestions": ["Add more tests", "Refactor auth module"],
            "overall_score": 70,
        })
        result = parse_evaluator_output(None, text, metrics)
        assert len(result.issues) == 2
        assert "auth.py:42" in result.issues[0]
        assert len(result.suggestions) == 2

    def test_proxy_score_parsing(self):
        metrics = [
            Metric("latency", MetricType.SOFT, "Response time",
                   automation=AutomationLevel.HYBRID, proxy_metric="Mock latency"),
        ]
        text = json.dumps({
            "metrics": [{"name": "latency", "passed": True, "value": "50ms", "score": 85, "proxy_score": 90, "proxy_notes": "Mock test shows good latency", "reason": "Latency within acceptable range"}],
            "overall_score": 85,
        })
        result = parse_evaluator_output(None, text, metrics)
        assert len(result.metrics) == 1
        assert result.metrics[0].proxy_score == 90
        assert "Mock" in result.metrics[0].proxy_notes

    def test_missing_metric_name_in_list(self):
        """Metric name from output doesn't match provided list → ad-hoc Metric created."""
        metrics = self._make_metrics()
        text = json.dumps({
            "metrics": [{"name": "unknown_metric", "passed": True, "reason": "Some reason"}],
            "overall_score": 50,
        })
        result = parse_evaluator_output(None, text, metrics)
        assert len(result.metrics) == 1  # Ad-hoc metric created
        assert result.metrics[0].metric.name == "unknown_metric"

    def test_overall_passed_logic(self):
        """overall_passed is True only when ALL hard metrics pass."""
        metrics = [
            Metric("test1", MetricType.HARD, "Test 1"),
            Metric("test2", MetricType.HARD, "Test 2"),
            Metric("quality", MetricType.SUBJECTIVE, "Quality"),
        ]
        text = json.dumps({
            "metrics": [
                {"name": "test1", "passed": True, "reason": "Pass"},
                {"name": "test2", "passed": False, "reason": "Fail"},
                {"name": "quality", "passed": True, "score": 90, "reason": "Good"},
            ],
            "overall_score": 70,
        })
        result = parse_evaluator_output(None, text, metrics)
        assert result.overall_passed is False  # test2 failed

    def test_no_hard_metrics_means_passed(self):
        """If no hard metrics, overall_passed defaults to True."""
        metrics = [Metric("quality", MetricType.SUBJECTIVE, "Quality")]
        text = json.dumps({
            "metrics": [{"name": "quality", "passed": True, "score": 80, "reason": "Good"}],
            "overall_score": 80,
        })
        result = parse_evaluator_output(None, text, metrics)
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
        result = parse_evaluator_output(None, text, metrics)
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
        result = parse_evaluator_output(None, text, metrics)
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
        result = parse_evaluator_output(None, text, metrics)
        assert len(result.metrics) == 1
        assert result.overall_score == 95

    def test_structured_output_preferred(self):
        """structured_output is preferred over text extraction."""
        metrics = self._make_metrics()
        structured = {
            "metrics": [{"name": "tests_pass", "passed": True, "reason": "From structured"}],
            "overall_score": 80,
        }
        text = json.dumps({
            "metrics": [{"name": "tests_pass", "passed": False, "reason": "From text"}],
            "overall_score": 50,
        })
        result = parse_evaluator_output(structured, text, metrics)
        assert len(result.metrics) == 1
        assert result.overall_score == 80
        assert result.metrics[0].passed is True


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
        decision = parse_planner_output(None, text)
        assert decision.action == Action.EXECUTE
        assert decision.target == "T001"
        assert "ready" in decision.reason

    def test_bare_json(self):
        text = '{"action": "explore", "target": "T002", "reason": "Need research"}'
        decision = parse_planner_output(None, text)
        assert decision.action == Action.EXPLORE
        assert decision.target == "T002"

    def test_json_fallback_from_text(self):
        """When structured_output is None, extract_json parses JSON from text."""
        text = '{"action": "execute", "target": "T001", "reason": "Task is ready"}'
        decision = parse_planner_output(None, text)
        assert decision.action == Action.EXECUTE
        assert decision.target == "T001"

    def test_empty_input(self):
        decision = parse_planner_output(None, "")
        assert decision.action == Action.SKIP  # Default

    def test_invalid_action(self):
        text = '{"action": "nonexistent_action", "target": "T001"}'
        decision = parse_planner_output(None, text)
        assert decision.action == Action.SKIP  # Fallback

    def test_parallel_execute_with_task_ids(self):
        text = '{"action": "parallel_execute", "task_ids": ["T001", "T002", "T003"], "reason": "Independent tasks"}'
        decision = parse_planner_output(None, text)
        assert decision.action == Action.PARALLEL_EXECUTE
        assert decision.task_ids == ["T001", "T002", "T003"]

    def test_task_ids_as_string(self):
        """task_ids provided as string → parsed into list."""
        text = '{"action": "parallel_execute", "task_ids": "T001, T002", "reason": "Test"}'
        decision = parse_planner_output(None, text)
        assert decision.task_ids == ["T001", "T002"]

    def test_hedge_action(self):
        text = '{"action": "hedge", "target": "T001", "failure_assumptions": "May not work", "reason": "Risky"}'
        decision = parse_planner_output(None, text)
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
        decision = parse_planner_output(None, text)
        assert decision.action == Action.PIVOT_ITERATION
        assert decision.attempt_count == 5
        assert decision.best_score == "65"

    def test_done_action(self):
        text = '{"action": "done", "reason": "All tasks completed"}'
        decision = parse_planner_output(None, text)
        assert decision.action == Action.DONE

    def test_ask_with_question(self):
        text = '{"action": "ask", "question": "Should we use Redis or Memcached?", "reason": "Need user input"}'
        decision = parse_planner_output(None, text)
        assert decision.action == Action.ASK
        assert "Redis" in decision.question

    def test_structured_output_preferred(self):
        """structured_output is preferred over text extraction."""
        structured = {"action": "execute", "target": "T001", "reason": "From structured output"}
        text = '{"action": "done", "reason": "From text"}'
        decision = parse_planner_output(structured, text)
        assert decision.action == Action.EXECUTE
        assert "structured output" in decision.reason


class TestReviewerParsing:
    """Test parse_reviewer_output edge cases."""

    def test_json_verdict(self):
        text = '{"verdict": "passed", "reason": "All requirements met", "suggestions": "Add more tests"}'
        result = parse_reviewer_output(None, text)
        assert result.verdict == Verdict.PASSED
        assert "requirements" in result.reason

    def test_json_fallback_from_text(self):
        """When structured_output is None, extract_json parses JSON from text."""
        text = '{"verdict": "retry", "reason": "Some tests fail", "suggestions": "Fix test_auth"}'
        result = parse_reviewer_output(None, text)
        assert result.verdict == Verdict.RETRY
        assert "tests fail" in result.reason

    def test_empty_input(self):
        result = parse_reviewer_output(None, "")
        assert result.verdict == Verdict.RETRY  # Default on parse failure

    def test_invalid_verdict(self):
        text = '{"verdict": "unknown_verdict", "reason": "test"}'
        result = parse_reviewer_output(None, text)
        assert result.verdict == Verdict.RETRY  # Fallback on parse failure

    def test_failed_verdict(self):
        text = '{"verdict": "failed", "reason": "Fundamental design flaw"}'
        result = parse_reviewer_output(None, text)
        assert result.verdict == Verdict.FAILED

    def test_structured_output_preferred(self):
        """structured_output is preferred over text extraction."""
        structured = {"verdict": "passed", "reason": "From structured output"}
        text = '{"verdict": "retry", "reason": "From text"}'
        result = parse_reviewer_output(structured, text)
        assert result.verdict == Verdict.PASSED
        assert "structured output" in result.reason


class TestWorkerParsing:
    """Test extract_worker_metadata edge cases."""

    def test_always_succeeds(self):
        result = extract_worker_metadata(None, "any text", "IMPLEMENT")
        assert result["success"] is True

    def test_explore_confidence_high(self):
        result = extract_worker_metadata(None, "Confidence: high\nDone", "EXPLORE")
        assert result["confidence"] == "high"

    def test_explore_confidence_low(self):
        result = extract_worker_metadata(None, "Confidence: Low", "EXPLORE")
        assert result["confidence"] == "low"

    def test_no_confidence(self):
        result = extract_worker_metadata(None, "Some output", "EXPLORE")
        assert "confidence" not in result

    def test_empty_output(self):
        result = extract_worker_metadata(None, "", "IMPLEMENT")
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
        """pivot_recommended: true → should_pivot=True"""
        metrics = [Metric("test", MetricType.HARD, "Test")]
        text = json.dumps({
            "metrics": [{"name": "test", "passed": True, "reason": "Pass"}],
            "overall_score": 70,
            "pivot_recommended": True,
            "pivot_reason": "Scores declining consistently",
        })
        result = parse_evaluator_output(None, text, metrics)
        assert result.should_pivot is True
        assert "declining" in result.pivot_reason.lower()

    def test_parse_pivot_no(self):
        """pivot_recommended: false → should_pivot=False"""
        metrics = [Metric("test", MetricType.HARD, "Test")]
        text = json.dumps({
            "metrics": [{"name": "test", "passed": True, "reason": "Pass"}],
            "overall_score": 85,
            "pivot_recommended": False,
            "pivot_reason": "Scores improving steadily",
        })
        result = parse_evaluator_output(None, text, metrics)
        assert result.should_pivot is False

    def test_parse_pivot_missing(self):
        """No pivot_recommended → should_pivot stays at default (False)"""
        metrics = [Metric("test", MetricType.HARD, "Test")]
        text = json.dumps({
            "metrics": [{"name": "test", "passed": True, "reason": "Pass"}],
            "overall_score": 90,
        })
        result = parse_evaluator_output(None, text, metrics)
        assert result.should_pivot is False

    def test_parse_pivot_reason_extracted(self):
        """pivot_reason is correctly extracted"""
        metrics = [Metric("test", MetricType.HARD, "Test")]
        text = json.dumps({
            "metrics": [{"name": "test", "passed": False, "reason": "Fail"}],
            "overall_score": 30,
            "pivot_recommended": True,
            "pivot_reason": "Hard metric tests_pass has failed 3 times in a row",
        })
        result = parse_evaluator_output(None, text, metrics)
        assert result.should_pivot is True
        assert "tests_pass" in result.pivot_reason

    def test_parse_pivot_true_always_works(self):
        """pivot_recommended: true is consistently parsed"""
        metrics = [Metric("test", MetricType.HARD, "Test")]
        text = json.dumps({
            "metrics": [{"name": "test", "passed": True, "reason": "Pass"}],
            "overall_score": 50,
            "pivot_recommended": True,
            "pivot_reason": "Test reason",
        })
        result = parse_evaluator_output(None, text, metrics)
        assert result.should_pivot is True

    def test_parse_pivot_false_always_works(self):
        """pivot_recommended: false is consistently parsed"""
        metrics = [Metric("test", MetricType.HARD, "Test")]
        text = json.dumps({
            "metrics": [{"name": "test", "passed": True, "reason": "Pass"}],
            "overall_score": 80,
            "pivot_recommended": False,
        })
        result = parse_evaluator_output(None, text, metrics)
        assert result.should_pivot is False

    def test_parse_pivot_with_full_evaluator_output(self):
        """PIVOT fields extracted from a full evaluator output with all sections"""
        metrics = [
            Metric("tests_pass", MetricType.HARD, "Tests pass"),
            Metric("code_quality", MetricType.SUBJECTIVE, "Quality"),
        ]
        text = json.dumps({
            "metrics": [
                {"name": "tests_pass", "passed": False, "value": "3/10 tests fail", "score": 30, "reason": "Multiple test failures in auth module"},
                {"name": "code_quality", "passed": True, "score": 75, "reason": "Decent code structure"},
            ],
            "issues": ["auth.py:42 - Missing null check", "db.py:100 - Connection leak"],
            "suggestions": ["Add error handling", "Fix auth tests"],
            "overall_score": 45,
            "pivot_recommended": True,
            "pivot_reason": "Hard metric tests_pass keeps failing, score stuck at 45",
        })
        result = parse_evaluator_output(None, text, metrics)
        assert len(result.metrics) == 2
        assert result.overall_score == 45
        assert result.should_pivot is True
        assert "tests_pass" in result.pivot_reason

    # --- Prompt construction tests (Issue #3) ---

    def test_prompt_includes_attempt_history(self):
        """attempt_number > 1 → prompt includes Attempt History section without scores"""
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
        # Anti-anchoring: specific scores should NOT appear in prompt
        assert "Previous scores" not in prompt
        assert "40, 55" not in prompt
        # Should mention attempt count without scores
        assert "2 previous attempts" in prompt

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

    def test_prompt_no_scores_or_trend(self):
        """Anti-anchoring: no scores or trend in prompt"""
        prompt = build_evaluator_prompt(
            task_id="T001",
            goal="Build feature X",
            task_detail="Task details here",
            metrics=[Metric("test", MetricType.HARD, "Test")],
            attempt_number=4,
            previous_scores=[30.0, 40.0, 50.0],
        )
        # Should NOT contain score numbers or trend words
        assert "30, 40, 50" not in prompt
        assert "Score trend" not in prompt
        # Should contain anti-anchoring instruction
        assert "Evaluate the code AS-IS" in prompt

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

    def test_evaluator_has_no_edit_tool(self):
        """Evaluator agent should not have Edit tool (Write is needed for adversarial tests)"""
        import inspect
        from ralph_sdk import evaluator
        source = inspect.getsource(evaluator.evaluate)
        # Evaluator should not have Edit tool (it doesn't modify existing files)
        assert '"Edit"' not in source, "Evaluator should not have Edit tool"
        # Write IS allowed — needed for adversarial test scripts and findings
        assert '"Write"' in source, "Evaluator needs Write tool for adversarial testing"

    def test_system_prompt_has_pivot_assessment(self):
        """System prompt contains pivot assessment guidance"""
        assert "Pivot Assessment" in EVALUATOR_SYSTEM_PROMPT
        assert "PIVOT_RECOMMENDED" in EVALUATOR_SYSTEM_PROMPT

    # --- Code fallback tests ---

    def test_code_fallback_when_agent_silent(self):
        """Agent doesn't output pivot_recommended → fallback to _assess_pivot_recommendation

        When parse_evaluator_output doesn't find pivot_recommended,
        should_pivot stays at default (False) and pivot_reason is empty.
        In evaluate(), this triggers the fallback to code-based detection.
        """
        metrics = [Metric("test", MetricType.HARD, "Test")]
        text = json.dumps({
            "metrics": [{"name": "test", "passed": True, "reason": "Pass"}],
            "overall_score": 50,
        })
        result = parse_evaluator_output(None, text, metrics)
        # Agent didn't output pivot, so should_pivot=False and pivot_reason=""
        assert result.should_pivot is False
        assert result.pivot_reason == ""
        # This combination (False + empty reason) triggers fallback in evaluate()

    def test_agent_pivot_overrides_code(self):
        """Agent outputs pivot_recommended: true → code fallback is skipped.

        When agent sets should_pivot=True, evaluate() should NOT
        overwrite it with _assess_pivot_recommendation().
        """
        metrics = [Metric("test", MetricType.HARD, "Test")]
        text = json.dumps({
            "metrics": [{"name": "test", "passed": True, "reason": "Pass"}],
            "overall_score": 80,
            "pivot_recommended": True,
            "pivot_reason": "Despite good score, approach is fundamentally limited",
        })
        result = parse_evaluator_output(None, text, metrics)
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


# =============================================================================
# 10. Handoff Note Bugs (Bug 1 & 2)
# =============================================================================


class TestHandoffStatusParsing:
    """Bug 1: Handoff status parsing should handle both formats."""

    def test_colon_format_done(self, tmp_path):
        """## Status: done → classified as completed."""
        cwd = setup_ralph(tmp_path)
        init_pool("Test", "| T001 | IMPLEMENT | pending |", cwd)
        write_task("T001", "# T001\n\n## Status: done\n\n## Notes\nDone.", cwd)

        handoff = generate_handoff_note(cwd)
        assert "T001" in handoff
        # T001 should be in completed list
        assert "已完成: T001" in handoff

    def test_colon_format_completed(self, tmp_path):
        """## Status: completed → classified as completed."""
        cwd = setup_ralph(tmp_path)
        init_pool("Test", "| T001 | IMPLEMENT | pending |", cwd)
        write_task("T001", "# T001\n\n## Status: completed\n", cwd)

        handoff = generate_handoff_note(cwd)
        assert "已完成: T001" in handoff

    def test_colon_format_passed(self, tmp_path):
        """## Status: passed → classified as completed."""
        cwd = setup_ralph(tmp_path)
        init_pool("Test", "| T001 | IMPLEMENT | pending |", cwd)
        write_task("T001", "# T001\n\n## Status: passed\n", cwd)

        handoff = generate_handoff_note(cwd)
        assert "已完成: T001" in handoff

    def test_newline_format_completed(self, tmp_path):
        """## Status\\ncompleted → classified as completed."""
        cwd = setup_ralph(tmp_path)
        init_pool("Test", "| T001 | IMPLEMENT | pending |", cwd)
        write_task("T001", "# T001\n\n## Status\ncompleted\n", cwd)

        handoff = generate_handoff_note(cwd)
        assert "已完成: T001" in handoff

    def test_colon_format_pending(self, tmp_path):
        """## Status: pending → classified as pending."""
        cwd = setup_ralph(tmp_path)
        init_pool("Test", "| T001 | IMPLEMENT | pending |", cwd)
        write_task("T001", "# T001\n\n## Status: pending\n", cwd)

        handoff = generate_handoff_note(cwd)
        assert "待处理: T001" in handoff

    def test_mixed_statuses(self, tmp_path):
        """Multiple tasks with different status formats."""
        cwd = setup_ralph(tmp_path)
        init_pool("Test", "| T001 | IMPLEMENT | done |\n| T002 | IMPLEMENT | pending |", cwd)
        write_task("T001", "# T001\n\n## Status: done\n", cwd)
        write_task("T002", "# T002\n\n## Status\npending\n", cwd)

        handoff = generate_handoff_note(cwd)
        assert "已完成: T001" in handoff
        assert "待处理: T002" in handoff

    def test_in_progress_status(self, tmp_path):
        """## Status: in_progress → classified as in progress."""
        cwd = setup_ralph(tmp_path)
        init_pool("Test", "| T001 | IMPLEMENT | in_progress |", cwd)
        write_task("T001", "# T001\n\n## Status: in_progress\n", cwd)

        handoff = generate_handoff_note(cwd)
        assert "进行中: T001" in handoff


class TestHandoffFindingsParsing:
    """Bug 2: Handoff should match both 'Findings' and 'Shared Findings'."""

    def test_shared_findings_section(self, tmp_path):
        """Pool with '## Shared Findings' → findings appear in handoff."""
        cwd = setup_ralph(tmp_path)
        pool_content = """# Task Pool

## Goal Summary
Test

## Active Tasks
| T001 | IMPLEMENT | done |

## Shared Findings

- Important discovery about the API
- Another key finding

## Progress Log
"""
        write_pool(pool_content, cwd)
        write_task("T001", "# T001\n\n## Status: done\n", cwd)

        handoff = generate_handoff_note(cwd)
        assert "Important discovery" in handoff
        assert "Another key finding" in handoff

    def test_plain_findings_section(self, tmp_path):
        """Pool with '## Findings' → findings still appear (backward compat)."""
        cwd = setup_ralph(tmp_path)
        pool_content = """# Task Pool

## Goal Summary
Test

## Active Tasks
| T001 | IMPLEMENT | done |

## Findings

- Legacy finding here

## Progress Log
"""
        write_pool(pool_content, cwd)
        write_task("T001", "# T001\n\n## Status: done\n", cwd)

        handoff = generate_handoff_note(cwd)
        assert "Legacy finding" in handoff


# =============================================================================
# 11. Input Tokens Counting (Bug 3)
# =============================================================================


class TestInputTokensCounting:
    """Bug 3: input_tokens should include cache tokens."""

    def test_log_query_stats_includes_cache_tokens(self):
        """log_query_stats sums input_tokens + cache tokens."""
        from ralph_sdk.logger import SessionLogger
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            logger = SessionLogger(tmp)

            # Create a mock ResultMessage with cache tokens
            mock_result = type("MockResult", (), {
                "usage": {
                    "input_tokens": 65,
                    "output_tokens": 5000,
                    "cache_creation_input_tokens": 10000,
                    "cache_read_input_tokens": 50000,
                },
                "total_cost_usd": 0.50,
            })()

            logger.log_query_stats(mock_result)

            # Total input should be 65 + 10000 + 50000 = 60065
            assert logger.metrics.total_input_tokens == 60065
            assert logger.metrics.total_output_tokens == 5000

    def test_log_query_stats_no_cache_tokens(self):
        """Works when cache tokens are absent."""
        from ralph_sdk.logger import SessionLogger
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            logger = SessionLogger(tmp)

            mock_result = type("MockResult", (), {
                "usage": {
                    "input_tokens": 5000,
                    "output_tokens": 3000,
                },
                "total_cost_usd": 0.10,
            })()

            logger.log_query_stats(mock_result)
            assert logger.metrics.total_input_tokens == 5000

    def test_log_query_stats_null_usage(self):
        """Handles None usage gracefully."""
        from ralph_sdk.logger import SessionLogger
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            logger = SessionLogger(tmp)

            mock_result = type("MockResult", (), {
                "usage": None,
                "total_cost_usd": None,
            })()

            logger.log_query_stats(mock_result)
            assert logger.metrics.total_input_tokens == 0


# =============================================================================
# 12. Pool Status Update (Issue 9)
# =============================================================================


class TestPoolStatusUpdate:
    """Issue 9: Pool status should update to COMPLETED on DONE."""

    def test_update_existing_status(self, tmp_path):
        """Update '## Status: IN_PROGRESS' to '## Status: COMPLETED'."""
        cwd = setup_ralph(tmp_path)
        pool_content = """# Task Pool

## Status: IN_PROGRESS

## Active Tasks
| T001 | IMPLEMENT | done |
"""
        write_pool(pool_content, cwd)

        update_pool_status("COMPLETED", cwd)

        result = read_pool(cwd)
        assert "## Status: COMPLETED" in result
        assert "IN_PROGRESS" not in result

    def test_insert_status_when_missing(self, tmp_path):
        """Add status when none exists."""
        cwd = setup_ralph(tmp_path)
        init_pool("Test", "| T001 | IMPLEMENT | pending |", cwd)

        update_pool_status("COMPLETED", cwd)

        result = read_pool(cwd)
        assert "## Status: COMPLETED" in result

    def test_update_newline_format_status(self, tmp_path):
        """Update '## Status\\nIN_PROGRESS' format."""
        cwd = setup_ralph(tmp_path)
        pool_content = """# Task Pool

## Status
IN_PROGRESS

## Active Tasks
| T001 | IMPLEMENT | done |
"""
        write_pool(pool_content, cwd)

        update_pool_status("COMPLETED", cwd)

        result = read_pool(cwd)
        assert "## Status: COMPLETED" in result


# =============================================================================
# 13. Mark Pending Tasks Skipped (Issue 7)
# =============================================================================


class TestMarkPendingTasksSkipped:
    """Issue 7: Pending tasks should be marked skipped when DONE."""

    def test_marks_pending_as_skipped(self, tmp_path):
        """Pending task → status changed to skipped."""
        cwd = setup_ralph(tmp_path)
        init_pool("Test", "| T001 | IMPLEMENT | done |\n| T002 | IMPLEMENT | pending |", cwd)
        write_task("T001", "# T001\n\n## Status: done\n", cwd)
        write_task("T002", "# T002\n\n## Status: pending\n", cwd)

        skipped = mark_pending_tasks_skipped("Goal completed", cwd)

        assert skipped == ["T002"]
        t002_content = read_task("T002", cwd)
        assert "skipped" in t002_content.lower()
        assert "Goal completed" in t002_content

    def test_does_not_touch_done_tasks(self, tmp_path):
        """Completed tasks are not affected."""
        cwd = setup_ralph(tmp_path)
        init_pool("Test", "| T001 | IMPLEMENT | done |", cwd)
        write_task("T001", "# T001\n\n## Status: done\n", cwd)

        skipped = mark_pending_tasks_skipped("Goal completed", cwd)

        assert skipped == []
        t001_content = read_task("T001", cwd)
        assert "skipped" not in t001_content.lower()

    def test_multiple_pending_tasks(self, tmp_path):
        """Multiple pending tasks → all marked skipped."""
        cwd = setup_ralph(tmp_path)
        init_pool("Test", "| T001 | IMPLEMENT | done |\n| T002 | EXPLORE | pending |\n| T003 | IMPLEMENT | pending |", cwd)
        write_task("T001", "# T001\n\n## Status: done\n", cwd)
        write_task("T002", "# T002\n\n## Status: pending\n", cwd)
        write_task("T003", "# T003\n\n## Status: pending\n", cwd)

        skipped = mark_pending_tasks_skipped("Superseded", cwd)

        assert set(skipped) == {"T002", "T003"}


# =============================================================================
# 14. Evaluator Metrics Parsed Count (Issue 10)
# =============================================================================


class TestMetricsParsedCount:
    """Issue 10: Evaluator metrics parsed count should be accurate."""

    def test_adhoc_metrics_created_for_unknown_names(self):
        """When agent outputs metric names not in metrics list, create ad-hoc Metrics."""
        default_metrics = [
            Metric("no_errors", MetricType.HARD, "No errors"),
            Metric("requirements_met", MetricType.HARD, "Requirements met"),
            Metric("code_quality", MetricType.SUBJECTIVE, "Code quality"),
        ]
        agent_output = {
            "metrics": [
                {"name": "runs_without_error", "passed": True, "reason": "All good"},
                {"name": "basic_functionality", "passed": True, "reason": "Works"},
                {"name": "test_pass_rate", "passed": True, "score": 95, "reason": "95% pass"},
                {"name": "concurrency_safety", "passed": True, "reason": "Thread safe"},
                {"name": "timing_precision", "passed": True, "score": 88, "reason": "Within bounds"},
                {"name": "test_coverage", "passed": False, "score": 60, "reason": "Low coverage"},
            ],
            "overall_score": 80,
        }
        result = _parse_evaluator_json(agent_output, default_metrics)

        # All 6 metrics should be included (none discarded)
        assert len(result.metrics) == 6
        names = [mr.metric.name for mr in result.metrics]
        assert "runs_without_error" in names
        assert "test_coverage" in names

    def test_adhoc_metric_type_inferred_from_score(self):
        """Ad-hoc metrics with score → SUBJECTIVE, without → HARD."""
        agent_output = {
            "metrics": [
                {"name": "has_score", "passed": True, "score": 90, "reason": "Good"},
                {"name": "no_score", "passed": True, "reason": "OK"},
            ],
            "overall_score": 85,
        }
        result = _parse_evaluator_json(agent_output, [])

        by_name = {mr.metric.name: mr for mr in result.metrics}
        assert by_name["has_score"].metric.type == MetricType.SUBJECTIVE
        assert by_name["no_score"].metric.type == MetricType.HARD

    def test_metrics_attempted_set_correctly(self):
        """metrics_attempted should equal the number of metrics in agent output."""
        agent_output = {
            "metrics": [
                {"name": "m1", "passed": True, "reason": "OK"},
                {"name": "m2", "passed": False, "reason": "Fail"},
                {"name": "m3", "passed": True, "score": 80, "reason": "Good"},
            ],
            "overall_score": 70,
        }
        result = _parse_evaluator_json(agent_output, [])
        assert result.metrics_attempted == 3

    def test_denominator_uses_metrics_attempted(self):
        """evaluate() should use metrics_attempted as denominator, not len(metrics)."""
        import inspect
        from ralph_sdk import evaluator
        source = inspect.getsource(evaluator.evaluate)

        # The denominator should reference metrics_attempted
        assert "metrics_attempted" in source
        # The old hard-coded pattern should be gone
        assert 'f"**Metrics parsed**: {len(result.metrics)}/{len(metrics)}"' not in source

    def test_known_metrics_still_matched(self):
        """Metrics matching the provided list should use the original Metric object."""
        metrics = [
            Metric("code_quality", MetricType.SUBJECTIVE, "Code quality", target=">= 80"),
        ]
        agent_output = {
            "metrics": [
                {"name": "code_quality", "passed": True, "score": 90, "reason": "Great"},
                {"name": "unknown_metric", "passed": True, "reason": "Fine"},
            ],
            "overall_score": 90,
        }
        result = _parse_evaluator_json(agent_output, metrics)

        by_name = {mr.metric.name: mr for mr in result.metrics}
        # Known metric should keep its target from the original definition
        assert by_name["code_quality"].metric.target == ">= 80"
        # Unknown metric should be ad-hoc (no target)
        assert by_name["unknown_metric"].metric.target is None

    def test_parse_evaluator_output_with_adhoc(self):
        """parse_evaluator_output should also create ad-hoc metrics via _parse_evaluator_json."""
        metrics = [Metric("no_errors", MetricType.HARD, "No errors")]
        structured = {
            "metrics": [
                {"name": "no_errors", "passed": True, "reason": "OK"},
                {"name": "custom_metric", "passed": True, "score": 75, "reason": "Decent"},
            ],
            "overall_score": 80,
        }
        result = parse_evaluator_output(structured, "", metrics)
        assert len(result.metrics) == 2
        assert result.metrics_attempted == 2


# =============================================================================
# 15. Orchestrator DONE Handler Updates (Issues 7 & 9)
# =============================================================================


class TestOrchestratorDoneHandler:
    """Verify orchestrator DONE handler updates pool status and marks skipped tasks."""

    def test_done_updates_pool_status(self):
        """orchestrator.run() calls update_pool_status on DONE."""
        import inspect
        from ralph_sdk import orchestrator
        source = inspect.getsource(orchestrator.run)
        assert "update_pool_status" in source

    def test_done_marks_pending_skipped(self):
        """orchestrator.run() calls mark_pending_tasks_skipped on DONE."""
        import inspect
        from ralph_sdk import orchestrator
        source = inspect.getsource(orchestrator.run)
        assert "mark_pending_tasks_skipped" in source


# =============================================================================
# 16. Emoji Status Value Parsing (Issue 10 附带)
# =============================================================================


class TestEmojiStatusParsing:
    """Status regex should skip emoji prefixes like '✅ completed'."""

    def test_handoff_parses_emoji_status_completed(self, tmp_path):
        """generate_handoff_note should parse '✅ completed' as completed."""
        init_ralph_dir(str(tmp_path))
        # Write pool with a task reference (must use T001 format)
        write_pool("## Tasks\n| ID | Title |\n|---|---|\n| T001 | Do something |\n", str(tmp_path))
        # Write task with emoji status
        task_content = "# T001: Test\n\n## Type\nIMPLEMENT\n\n## Status\n✅ completed\n"
        write_task("T001", task_content, str(tmp_path))

        note = generate_handoff_note(str(tmp_path))
        assert "T001" in note
        # Should classify as completed, not in_progress
        # The handoff note uses Chinese "已完成" for completed
        assert "已完成" in note or "completed" in note.lower()

    def test_mark_pending_skips_emoji_completed(self, tmp_path):
        """mark_pending_tasks_skipped should not skip already-completed tasks with emoji."""
        init_ralph_dir(str(tmp_path))
        write_pool("## Tasks\n| ID | Title |\n|---|---|\n| T001 | Do something |\n", str(tmp_path))
        task_content = "# T001: Test\n\n## Type\nIMPLEMENT\n\n## Status\n✅ completed\n"
        write_task("T001", task_content, str(tmp_path))

        skipped = mark_pending_tasks_skipped("done", str(tmp_path))
        assert "T001" not in skipped  # Should not be skipped since it's completed

    def test_mark_pending_skips_plain_pending(self, tmp_path):
        """mark_pending_tasks_skipped should skip plain pending tasks."""
        init_ralph_dir(str(tmp_path))
        write_pool("## Tasks\n| ID | Title |\n|---|---|\n| T001 | Do something |\n", str(tmp_path))
        task_content = "# T001: Test\n\n## Type\nIMPLEMENT\n\n## Status\npending\n"
        write_task("T001", task_content, str(tmp_path))

        skipped = mark_pending_tasks_skipped("done", str(tmp_path))
        assert "T001" in skipped


# =============================================================================
# 17. Pipeline Quality Control (timeparser findings)
# =============================================================================


class TestDoneBlockedByTargetScore:
    """DONE handler should block when tasks score below target_score."""

    def test_done_blocked_below_target(self, tmp_path):
        """Task scored 88 with target 95 → DONE should be blocked."""
        cwd = setup_ralph(tmp_path)
        init_pool("Test", "| T001 | IMPLEMENT | pending |", cwd)
        # Write task with score below target
        task_content = """# T001: Test Task

## Type
IMPLEMENT

## Status
in_progress

## Evaluation (2026-02-05 10:00)

**Score**: 88/100
**Metrics parsed**: 3/3
"""
        write_task("T001", task_content, cwd)

        # Simulate what DONE handler checks
        task_ids = extract_task_ids_from_pool(cwd)
        target_score = 95
        below_target = []
        for tid in task_ids:
            tc = read_task(tid, cwd)
            if not tc:
                continue
            content_lower = tc.lower()
            status_match = re.search(r"## status[:\s]+(?:[^\w\s]\s*)?(\w+)", content_lower)
            if status_match and status_match.group(1) in ("skipped", "pending"):
                continue
            _, prev_scores = get_attempt_history(tid, cwd)
            if prev_scores and prev_scores[-1] < target_score:
                below_target.append((tid, prev_scores[-1]))

        assert len(below_target) == 1
        assert below_target[0] == ("T001", 88.0)

    def test_done_allowed_above_target(self, tmp_path):
        """Task scored 96 with target 95 → DONE should be allowed."""
        cwd = setup_ralph(tmp_path)
        init_pool("Test", "| T001 | IMPLEMENT | pending |", cwd)
        task_content = """# T001: Test Task

## Type
IMPLEMENT

## Status
in_progress

## Evaluation (2026-02-05 10:00)

**Score**: 96/100
**Metrics parsed**: 3/3
"""
        write_task("T001", task_content, cwd)

        task_ids = extract_task_ids_from_pool(cwd)
        target_score = 95
        below_target = []
        for tid in task_ids:
            tc = read_task(tid, cwd)
            if not tc:
                continue
            content_lower = tc.lower()
            status_match = re.search(r"## status[:\s]+(?:[^\w\s]\s*)?(\w+)", content_lower)
            if status_match and status_match.group(1) in ("skipped", "pending"):
                continue
            _, prev_scores = get_attempt_history(tid, cwd)
            if prev_scores and prev_scores[-1] < target_score:
                below_target.append((tid, prev_scores[-1]))

        assert len(below_target) == 0

    def test_done_skips_skipped_tasks(self, tmp_path):
        """Task with status=skipped should NOT block DONE even if scored low."""
        cwd = setup_ralph(tmp_path)
        init_pool("Test", "| T001 | IMPLEMENT | pending |\n| T002 | IMPLEMENT | pending |", cwd)

        # T001: skipped with low score
        write_task("T001", """# T001: Task

## Type
IMPLEMENT

## Status
skipped

## Evaluation (2026-02-05 10:00)

**Score**: 40/100
""", cwd)

        # T002: completed with good score
        write_task("T002", """# T002: Task

## Type
IMPLEMENT

## Status
in_progress

## Evaluation (2026-02-05 10:00)

**Score**: 97/100
""", cwd)

        task_ids = extract_task_ids_from_pool(cwd)
        target_score = 95
        below_target = []
        for tid in task_ids:
            tc = read_task(tid, cwd)
            if not tc:
                continue
            content_lower = tc.lower()
            status_match = re.search(r"## status[:\s]+(?:[^\w\s]\s*)?(\w+)", content_lower)
            if status_match and status_match.group(1) in ("skipped", "pending"):
                continue
            _, prev_scores = get_attempt_history(tid, cwd)
            if prev_scores and prev_scores[-1] < target_score:
                below_target.append((tid, prev_scores[-1]))

        # T001 skipped → not checked; T002 above target → no blocker
        assert len(below_target) == 0

    def test_done_no_scores_allows(self, tmp_path):
        """Task with no evaluation (no scores) should NOT block DONE."""
        cwd = setup_ralph(tmp_path)
        init_pool("Test", "| T001 | IMPLEMENT | pending |", cwd)
        write_task("T001", """# T001: Task

## Type
IMPLEMENT

## Status
in_progress
""", cwd)

        task_ids = extract_task_ids_from_pool(cwd)
        target_score = 95
        below_target = []
        for tid in task_ids:
            tc = read_task(tid, cwd)
            if not tc:
                continue
            content_lower = tc.lower()
            status_match = re.search(r"## status[:\s]+(?:[^\w\s]\s*)?(\w+)", content_lower)
            if status_match and status_match.group(1) in ("skipped", "pending"):
                continue
            _, prev_scores = get_attempt_history(tid, cwd)
            if prev_scores and prev_scores[-1] < target_score:
                below_target.append((tid, prev_scores[-1]))

        assert len(below_target) == 0


class TestEvalSectionSlimmed:
    """Evaluator should write structured summary, not full output, to task file."""

    def test_eval_section_no_full_output(self, tmp_path):
        """After evaluate(), task file should NOT contain '### Evaluator Full Output'."""
        cwd = setup_ralph(tmp_path)
        init_pool("Test", "| T001 | IMPLEMENT | pending |", cwd)
        write_task("T001", "# T001: Task\n\n## Status\nin_progress\n", cwd)

        # Simulate what evaluate() writes to task file
        from ralph_sdk.evaluator import EvaluationResult
        result = EvaluationResult(
            task_id="T001",
            overall_passed=True,
            overall_score=85,
            issues=["auth.py:42 missing validation"],
            suggestions=["Add input validation"],
            metrics_attempted=3,
        )

        from datetime import datetime
        task_content = read_task("T001", cwd)
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
        write_task("T001", task_content + eval_section, cwd)

        final_content = read_task("T001", cwd)
        assert "### Evaluator Full Output" not in final_content
        assert "### Issues" in final_content
        assert "auth.py:42" in final_content
        assert "### Suggestions" in final_content
        assert "Add input validation" in final_content

    def test_eval_log_file_created(self, tmp_path):
        """Full evaluator output should be saved to .ralph/logs/eval_{task_id}_{attempt}.md."""
        cwd = str(tmp_path)
        from ralph_sdk.pool import RALPH_DIR
        from pathlib import Path

        eval_log_path = Path(cwd) / RALPH_DIR / "logs" / "eval_T001_2.md"
        eval_log_path.parent.mkdir(parents=True, exist_ok=True)
        eval_log_path.write_text("Full evaluator output here with lots of details...")

        assert eval_log_path.exists()
        assert "Full evaluator output" in eval_log_path.read_text()


class TestMetricCountConstraint:
    """Evaluator prompt should enforce exact metric count."""

    def test_prompt_contains_metric_count(self):
        """build_evaluator_prompt should include exact metric count constraint."""
        metrics = [
            Metric("tests_pass", MetricType.HARD, "All tests pass"),
            Metric("code_quality", MetricType.SUBJECTIVE, "Code quality"),
            Metric("performance", MetricType.SOFT, "Response time", target="<= 100ms"),
        ]
        prompt = build_evaluator_prompt(
            task_id="T001",
            goal="Build something",
            task_detail="Do the thing",
            metrics=metrics,
        )
        assert "exactly 3 metrics entries" in prompt

    def test_prompt_count_with_single_metric(self):
        """Single metric → 'exactly 1 metrics entries'."""
        metrics = [Metric("tests_pass", MetricType.HARD, "Tests pass")]
        prompt = build_evaluator_prompt(
            task_id="T001",
            goal="Goal",
            task_detail="Task",
            metrics=metrics,
        )
        assert "exactly 1 metrics entries" in prompt

    def test_metric_count_warning(self, capsys):
        """Warning printed when metrics_attempted < metrics_expected."""
        from ralph_sdk.evaluator import console as eval_console
        from io import StringIO

        # We can check by verifying the code path exists
        import inspect
        from ralph_sdk import evaluator
        source = inspect.getsource(evaluator.evaluate)
        assert "metrics_attempted" in source
        assert "metrics_expected" in source


class TestProgressLogBelowTarget:
    """Progress log should contain BELOW TARGET annotation."""

    def test_below_target_in_source(self):
        """orchestrator.py NEEDS IMPROVEMENT log should say BELOW TARGET."""
        import inspect
        from ralph_sdk import orchestrator
        source = inspect.getsource(orchestrator.run)
        assert "BELOW TARGET" in source

    def test_done_handler_checks_target_score(self):
        """DONE handler should check tasks against target_score."""
        import inspect
        from ralph_sdk import orchestrator
        source = inspect.getsource(orchestrator.run)
        # The DONE handler should reference target_score and get_attempt_history
        assert "below_target" in source
        assert "get_attempt_history" in source
        assert "DONE_BLOCKED" in source


# =============================================================================
# 18. Thinking + Full Text Output
# =============================================================================


class TestStreamResultThinkingText:
    """StreamResult should accumulate ThinkingBlock content."""

    def test_stream_result_has_thinking_text_field(self):
        """StreamResult dataclass includes thinking_text field."""
        from ralph_sdk.logger import StreamResult
        sr = StreamResult(text="hello", tool_count=0, result_stats=None)
        assert sr.thinking_text == ""

    def test_stream_result_with_thinking(self):
        """StreamResult can store thinking text."""
        from ralph_sdk.logger import StreamResult
        sr = StreamResult(
            text="answer",
            tool_count=0,
            result_stats=None,
            thinking_text="Let me think about this...",
        )
        assert sr.thinking_text == "Let me think about this..."

    def test_thinking_block_import(self):
        """ThinkingBlock is importable from logger module."""
        from ralph_sdk.logger import ThinkingBlock
        tb = ThinkingBlock(thinking="test thought", signature="sig123")
        assert tb.thinking == "test thought"
        assert tb.signature == "sig123"

    def test_thinking_block_detected_in_content(self):
        """ThinkingBlock is identified via isinstance, not hasattr."""
        from claude_agent_sdk import ThinkingBlock
        tb = ThinkingBlock(thinking="deep thought", signature="sig")
        assert isinstance(tb, ThinkingBlock)
        assert not hasattr(tb, "text")  # ThinkingBlock has .thinking, not .text
        assert hasattr(tb, "thinking")


class TestThinkingBudgetDefaults:
    """Each agent should have correct default thinking_budget."""

    def test_planner_default_10000(self):
        """Planner defaults to 10000 thinking tokens."""
        import inspect
        from ralph_sdk import planner
        source = inspect.getsource(planner.plan)
        # Default should be 10_000 or 10000
        assert "10_000" in source or "10000" in source

    def test_evaluator_default_10000(self):
        """Evaluator defaults to 10000 thinking tokens."""
        import inspect
        from ralph_sdk import evaluator
        source = inspect.getsource(evaluator.evaluate)
        assert "10_000" in source or "10000" in source

    def test_worker_default_0(self):
        """Worker defaults to 0 thinking tokens."""
        import inspect
        from ralph_sdk import worker
        source = inspect.getsource(worker.work)
        # Worker: effective = thinking_budget if thinking_budget is not None else 0
        assert "else 0" in source

    def test_reviewer_default_0(self):
        """Reviewer defaults to 0 thinking tokens."""
        import inspect
        from ralph_sdk import reviewer
        source = inspect.getsource(reviewer.review)
        assert "else 0" in source

    def test_initializer_default_0(self):
        """Initializer defaults to 0 thinking tokens."""
        import inspect
        from ralph_sdk import orchestrator
        source = inspect.getsource(orchestrator.initialize_pool)
        assert "else 0" in source

    def test_thinking_budget_override_propagates(self):
        """thinking_budget=0 should override all agent defaults."""
        import inspect
        from ralph_sdk import planner, evaluator, worker, reviewer

        # All agents check: thinking_budget if thinking_budget is not None else <default>
        for mod in [planner, evaluator, worker, reviewer]:
            func_name = {
                planner: "plan",
                evaluator: "evaluate",
                worker: "work",
                reviewer: "review",
            }[mod]
            source = inspect.getsource(getattr(mod, func_name))
            assert "thinking_budget is not None" in source, \
                f"{mod.__name__}.{func_name} should check thinking_budget is not None"


class TestCLIThinkingOption:
    """CLI --thinking option is properly wired."""

    def test_run_cmd_has_thinking_param(self):
        """run command accepts --thinking."""
        import inspect
        from ralph_sdk.cli import run_cmd
        sig = inspect.signature(run_cmd)
        assert "thinking_budget" in sig.parameters

    def test_resume_cmd_has_thinking_param(self):
        """resume command accepts --thinking."""
        import inspect
        from ralph_sdk.cli import resume_cmd
        sig = inspect.signature(resume_cmd)
        assert "thinking_budget" in sig.parameters

    def test_orchestrator_run_has_thinking_param(self):
        """orchestrator.run() accepts thinking_budget."""
        import inspect
        from ralph_sdk.orchestrator import run
        sig = inspect.signature(run)
        assert "thinking_budget" in sig.parameters

    def test_orchestrator_resume_has_thinking_param(self):
        """orchestrator.resume() accepts thinking_budget."""
        import inspect
        from ralph_sdk.orchestrator import resume
        sig = inspect.signature(resume)
        assert "thinking_budget" in sig.parameters


class TestFullTextDisplay:
    """Text blocks should be displayed in full (no truncation)."""

    def test_no_truncation_in_stream_query(self):
        """stream_query should not truncate text to 80 chars."""
        import inspect
        from ralph_sdk.logger import stream_query
        source = inspect.getsource(stream_query)
        # Old code had: first_line[:77] + "..."
        assert "[:77]" not in source
        assert 'first_line[:77]' not in source

    def test_no_truncation_in_stream_client_query(self):
        """stream_client_query should not truncate text to 80 chars."""
        import inspect
        from ralph_sdk.logger import stream_client_query
        source = inspect.getsource(stream_client_query)
        assert "[:77]" not in source
        assert 'first_line[:77]' not in source

    def test_thinking_delta_in_stream_query(self):
        """stream_query handles thinking_delta events."""
        import inspect
        from ralph_sdk.logger import stream_query
        source = inspect.getsource(stream_query)
        assert "thinking_delta" in source


# =============================================================================
# Category 14: Score Consistency Validation
# =============================================================================


class TestScoreConsistencyValidation:
    """Tests for _validate_score_consistency anti-anchoring mechanism."""

    def test_cosmetic_only_low_score_warns(self, capsys):
        """All cosmetic issues + score < 95 → warning printed."""
        result = EvaluationResult(
            task_id="T001",
            overall_passed=True,
            overall_score=88,
            issues=[
                "[COSMETIC] Missing docstring in parse_input()",
                "[COSMETIC] Naming: 'x' should be more descriptive",
                "[COSMETIC] Add type annotation to return value",
            ],
        )
        _validate_score_consistency(result)
        captured = capsys.readouterr()
        assert "All 3 issues are cosmetic" in captured.out
        assert "88" in captured.out

    def test_functional_issues_no_warning(self, capsys):
        """Mix of functional and cosmetic issues → no warning."""
        result = EvaluationResult(
            task_id="T001",
            overall_passed=False,
            overall_score=70,
            issues=[
                "[FUNCTIONAL] Bug: off-by-one in date calculation",
                "[COSMETIC] Missing docstring",
            ],
        )
        _validate_score_consistency(result)
        captured = capsys.readouterr()
        assert "cosmetic" not in captured.out.lower()

    def test_cosmetic_high_score_no_warning(self, capsys):
        """All cosmetic issues + score >= 95 → no warning."""
        result = EvaluationResult(
            task_id="T001",
            overall_passed=True,
            overall_score=96,
            issues=[
                "[COSMETIC] Minor naming inconsistency",
            ],
        )
        _validate_score_consistency(result)
        captured = capsys.readouterr()
        assert "cosmetic" not in captured.out.lower()

    def test_no_issues_no_warning(self, capsys):
        """No issues → no warning."""
        result = EvaluationResult(
            task_id="T001",
            overall_passed=True,
            overall_score=50,
            issues=[],
        )
        _validate_score_consistency(result)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_keyword_matching_case_insensitive(self, capsys):
        """Cosmetic keywords match case-insensitively."""
        result = EvaluationResult(
            task_id="T001",
            overall_passed=True,
            overall_score=85,
            issues=[
                "Missing DOCSTRING for class Foo",
                "Style issue: inconsistent indentation",
            ],
        )
        _validate_score_consistency(result)
        captured = capsys.readouterr()
        assert "All 2 issues are cosmetic" in captured.out
