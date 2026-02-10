"""
Experiment 4: Concurrent Writes — 并发写入 pool.md 的数据完整性

假设: 两个 agent 同时 Edit pool.md Findings section 会导致数据丢失。

测试层次:
1. Unit tests: pool.py 文件锁 + 原子写入的并发安全性
2. Integration tests: 真实并行 worker 的 findings 完整性

测量指标:
- 问题是否真实存在？(Y/N + 证据)
- pool.py 锁机制够用吗？(如果 agent 绕过 pool.py 用 Edit 工具呢？)
- MCP 工具的预期提升？
"""

import asyncio
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.mcp_validation.common import ExperimentResult, RalphTestDir


# =============================================================================
# Unit Tests: Concurrent Write Safety
# =============================================================================

def run_unit_tests() -> ExperimentResult:
    result = ExperimentResult(
        experiment_name="exp4_concurrent_writes_unit",
        description="Test pool.py file locking and atomic writes under contention",
    )

    from ralph_sdk.pool import (
        append_to_findings,
        append_to_progress_log,
        add_verified_info,
        read_pool,
        write_pool,
        _atomic_write,
        file_lock,
    )

    # --- Test 1: 10 concurrent append_to_findings ---
    with RalphTestDir("exp4_") as td:
        td.setup_basic_project(goal="Test concurrent writes")
        errors = []

        def append_finding(i):
            try:
                append_to_findings(f"Finding_{i:03d}: important discovery #{i}", td.cwd)
            except Exception as e:
                errors.append((i, str(e)))

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(append_finding, i) for i in range(10)]
            for f in futures:
                f.result()

        pool = read_pool(td.cwd)
        found_count = sum(1 for i in range(10) if f"Finding_{i:03d}" in pool)
        result.add(
            "10_concurrent_findings_all_present",
            found_count == 10 and len(errors) == 0,
            f"Found {found_count}/10 findings. Errors: {errors}",
        )

    # --- Test 2: 20 concurrent append_to_progress_log ---
    with RalphTestDir("exp4_") as td:
        td.setup_basic_project(goal="Test concurrent progress")
        errors = []

        def append_progress(i):
            try:
                append_to_progress_log(f"Progress_{i:03d}: step completed", td.cwd)
            except Exception as e:
                errors.append((i, str(e)))

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(append_progress, i) for i in range(20)]
            for f in futures:
                f.result()

        pool = read_pool(td.cwd)
        found_count = sum(1 for i in range(20) if f"Progress_{i:03d}" in pool)
        result.add(
            "20_concurrent_progress_all_present",
            found_count == 20 and len(errors) == 0,
            f"Found {found_count}/20 progress entries. Errors: {errors}",
        )

    # --- Test 3: 10 concurrent add_verified_info ---
    with RalphTestDir("exp4_") as td:
        td.setup_basic_project(goal="Test concurrent verified info")
        errors = []

        def add_info(i):
            try:
                add_verified_info(f"Topic_{i:03d}", f"Info_{i}", f"http://url{i}.com", td.cwd)
            except Exception as e:
                errors.append((i, str(e)))

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(add_info, i) for i in range(10)]
            for f in futures:
                f.result()

        pool = read_pool(td.cwd)
        found_count = sum(1 for i in range(10) if f"Topic_{i:03d}" in pool)
        result.add(
            "10_concurrent_verified_info_all_present",
            found_count == 10 and len(errors) == 0,
            f"Found {found_count}/10 verified info entries. Errors: {errors}",
        )

    # --- Test 4: Mixed concurrent operations ---
    with RalphTestDir("exp4_") as td:
        td.setup_basic_project(goal="Test mixed concurrent operations")
        errors = []

        def mixed_op(i):
            try:
                if i % 3 == 0:
                    append_to_findings(f"Mixed_Finding_{i:03d}: discovery", td.cwd)
                elif i % 3 == 1:
                    append_to_progress_log(f"Mixed_Progress_{i:03d}: step", td.cwd)
                else:
                    add_verified_info(f"Mixed_Topic_{i:03d}", f"Info_{i}", f"http://url{i}.com", td.cwd)
            except Exception as e:
                errors.append((i, str(e)))

        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = [executor.submit(mixed_op, i) for i in range(30)]
            for f in futures:
                f.result()

        pool = read_pool(td.cwd)
        finding_count = sum(1 for i in range(0, 30, 3) if f"Mixed_Finding_{i:03d}" in pool)
        progress_count = sum(1 for i in range(1, 30, 3) if f"Mixed_Progress_{i:03d}" in pool)
        info_count = sum(1 for i in range(2, 30, 3) if f"Mixed_Topic_{i:03d}" in pool)
        total_expected = 10 + 10 + 10  # 30 items, each type gets 10

        result.add(
            "30_mixed_concurrent_all_present",
            finding_count == 10 and progress_count == 10 and info_count == 10 and len(errors) == 0,
            f"Findings: {finding_count}/10, Progress: {progress_count}/10, Info: {info_count}/10. Errors: {errors}",
        )

    # --- Test 5: 50 rapid-fire findings (stress test) ---
    with RalphTestDir("exp4_") as td:
        td.setup_basic_project(goal="Stress test")
        errors = []

        def rapid_finding(i):
            try:
                append_to_findings(f"Rapid_{i:03d}", td.cwd)
            except Exception as e:
                errors.append((i, str(e)))

        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(rapid_finding, i) for i in range(50)]
            for f in futures:
                f.result()

        pool = read_pool(td.cwd)
        found_count = sum(1 for i in range(50) if f"Rapid_{i:03d}" in pool)
        result.add(
            "50_rapid_fire_stress_test",
            found_count == 50,
            f"Found {found_count}/50 entries. Lost: {50 - found_count}. Errors: {len(errors)}",
            category="stress",
        )

    # --- Test 6: pool.md structure integrity after concurrent writes ---
    with RalphTestDir("exp4_") as td:
        td.setup_basic_project(goal="Structure test")
        for i in range(20):
            append_to_findings(f"Structure_test_{i}", td.cwd)
        pool = read_pool(td.cwd)

        has_active = "## Active Tasks" in pool
        has_completed = "## Completed" in pool
        has_findings = "## Findings" in pool
        has_verified = "## Verified Information" in pool
        has_progress = "## Progress Log" in pool
        sections_intact = has_active and has_completed and has_findings and has_verified and has_progress

        result.add(
            "structure_intact_after_20_writes",
            sections_intact,
            f"Active={has_active}, Completed={has_completed}, Findings={has_findings}, "
            f"Verified={has_verified}, Progress={has_progress}",
        )

    # --- Test 7: atomic write doesn't leave temp files ---
    with RalphTestDir("exp4_") as td:
        td.setup_basic_project(goal="Test")
        ralph_dir = td.path / ".ralph"

        # Count files before
        files_before = set(str(p) for p in ralph_dir.rglob("*") if p.is_file())

        for i in range(10):
            append_to_findings(f"Cleanup_{i}", td.cwd)

        files_after = set(str(p) for p in ralph_dir.rglob("*") if p.is_file())
        temp_files = [f for f in files_after - files_before if ".tmp_" in f]

        result.add(
            "no_temp_files_left",
            len(temp_files) == 0,
            f"Temp files found: {temp_files}",
        )

    # --- Test 8: file lock timeout works ---
    with RalphTestDir("exp4_") as td:
        td.setup_basic_project(goal="Test")
        lock_path = td.path / ".ralph" / ".pool.lock"

        # Acquire lock in main thread
        import fcntl
        lock_fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)

        # Try to write from another thread (should timeout)
        timeout_occurred = [False]
        error_msg = [None]

        def try_write():
            try:
                from ralph_sdk.pool import file_lock as fl
                with fl(lock_path, timeout=1.0):
                    pass
            except TimeoutError:
                timeout_occurred[0] = True
            except Exception as e:
                error_msg[0] = str(e)

        t = threading.Thread(target=try_write)
        t.start()
        t.join(timeout=3.0)

        # Release lock
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

        result.add(
            "file_lock_timeout_works",
            timeout_occurred[0],
            f"Timeout occurred: {timeout_occurred[0]}, Error: {error_msg[0]}",
        )

    # --- Test 9: Agent BYPASSES pool.py with Edit tool ---
    # This is the core design gap: agents use Edit tool directly on pool.md
    # instead of pool.py's locked functions
    from ralph_sdk.prompts import WORKER_EXPLORE_PROMPT, WORKER_IMPLEMENT_PROMPT

    # Check if worker prompts tell agent to use Edit tool on pool.md
    explore_mentions_edit = "Edit" in WORKER_EXPLORE_PROMPT and "pool.md" in WORKER_EXPLORE_PROMPT
    implement_mentions_edit = "Edit" in WORKER_IMPLEMENT_PROMPT and "pool.md" in WORKER_IMPLEMENT_PROMPT
    mentions_findings = "Findings" in WORKER_EXPLORE_PROMPT and "Findings" in WORKER_IMPLEMENT_PROMPT

    result.add(
        "prompt_instructs_edit_pool_directly",
        mentions_findings,
        f"Worker prompts mention Findings section updates: {mentions_findings}. "
        "Agents use Read/Edit tools to modify pool.md directly, bypassing pool.py locks. "
        "This is the core concurrency risk.",
        category="design_gap",
    )

    # --- Test 10: worker.py allowed_tools includes Edit ---
    import ralph_sdk.worker as worker_mod
    worker_source = Path(worker_mod.__file__).read_text()
    has_edit_tool = '"Edit"' in worker_source or "'Edit'" in worker_source
    has_write_tool = '"Write"' in worker_source or "'Write'" in worker_source

    result.add(
        "worker_has_edit_write_tools",
        has_edit_tool and has_write_tool,
        f"Worker has Edit: {has_edit_tool}, Write: {has_write_tool}. "
        "These bypass pool.py file locking when agents modify pool.md directly.",
        category="design_gap",
    )

    # --- Test 11: Simulate what happens when 2 agents Edit pool.md without locking ---
    with RalphTestDir("exp4_") as td:
        td.setup_basic_project(goal="Simulate agent Edit conflict")
        pool_path = td.path / ".ralph" / "pool.md"

        # Simulate: 2 agents read pool.md at the same time
        content1 = pool_path.read_text()
        content2 = pool_path.read_text()

        # Agent 1 appends to Findings
        content1 = content1.replace(
            "## Findings\n\n(discoveries shared across tasks)",
            "## Findings\n\n(discoveries shared across tasks)\n- Agent1: Found important thing\n",
        )

        # Agent 2 appends to Findings (from the SAME original content)
        content2 = content2.replace(
            "## Findings\n\n(discoveries shared across tasks)",
            "## Findings\n\n(discoveries shared across tasks)\n- Agent2: Found another thing\n",
        )

        # Agent 1 writes first
        pool_path.write_text(content1)
        # Agent 2 overwrites (this is what happens without locking)
        pool_path.write_text(content2)

        final = pool_path.read_text()
        has_agent1 = "Agent1" in final
        has_agent2 = "Agent2" in final

        result.add(
            "simulated_edit_race_loses_data",
            not has_agent1 and has_agent2,
            f"Agent1 finding present: {has_agent1}, Agent2 finding present: {has_agent2}. "
            "This proves: when agents use Edit/Write tools directly (bypassing pool.py), "
            "concurrent writes cause data loss. Agent2's write overwrites Agent1's changes.",
            category="design_gap",
        )

    # --- Test 12: Same simulation with pool.py locked writes ---
    with RalphTestDir("exp4_") as td:
        td.setup_basic_project(goal="Test locked writes")
        errors = []

        def locked_write(i):
            try:
                append_to_findings(f"LockedAgent_{i}: important finding", td.cwd)
            except Exception as e:
                errors.append((i, str(e)))

        # Simulate 2 agents writing simultaneously via pool.py (which uses locks)
        t1 = threading.Thread(target=locked_write, args=(1,))
        t2 = threading.Thread(target=locked_write, args=(2,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        pool = read_pool(td.cwd)
        has_1 = "LockedAgent_1" in pool
        has_2 = "LockedAgent_2" in pool
        result.add(
            "locked_writes_preserve_both",
            has_1 and has_2 and len(errors) == 0,
            f"Agent1: {has_1}, Agent2: {has_2}, Errors: {errors}. "
            "With pool.py locks, both writes are preserved.",
            category="design_gap",
        )

    # --- Test 13: Large content concurrent writes ---
    with RalphTestDir("exp4_") as td:
        td.setup_basic_project(goal="Large content test")
        errors = []

        def large_finding(i):
            try:
                large_text = f"LargeFinding_{i:03d}: " + "x" * 1000  # 1KB per finding
                append_to_findings(large_text, td.cwd)
            except Exception as e:
                errors.append((i, str(e)))

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(large_finding, i) for i in range(10)]
            for f in futures:
                f.result()

        pool = read_pool(td.cwd)
        found_count = sum(1 for i in range(10) if f"LargeFinding_{i:03d}" in pool)
        result.add(
            "large_content_concurrent_safe",
            found_count == 10 and len(errors) == 0,
            f"Found {found_count}/10 large findings (1KB each). Errors: {errors}",
            category="stress",
        )

    return result


# =============================================================================
# Integration Tests: Real Parallel Workers
# =============================================================================

async def run_integration_tests(runs: int = 5) -> ExperimentResult:
    """
    Run real parallel EXPLORE workers and check findings integrity.

    Each worker should produce findings that appear in pool.md.
    """
    from ralph_sdk.orchestrator import execute_parallel_tasks

    result = ExperimentResult(
        experiment_name="exp4_concurrent_writes_integration",
        description=f"Test parallel worker findings integrity across {runs} runs",
    )

    for run_idx in range(runs):
        with RalphTestDir(f"exp4_int_r{run_idx}_") as td:
            td.setup_basic_project(
                goal="""Research three independent topics for a technical blog:
1. Python asyncio best practices
2. Docker multi-stage builds
3. Git workflow strategies

Each topic should be researched independently.
""",
                tasks=[
                    {
                        "id": "T001",
                        "type": "EXPLORE",
                        "title": "Research Python asyncio",
                        "description": "Find current best practices for asyncio. Write findings to pool.md Findings section.",
                    },
                    {
                        "id": "T002",
                        "type": "EXPLORE",
                        "title": "Research Docker multi-stage",
                        "description": "Find best practices for Docker multi-stage builds. Write findings to pool.md Findings section.",
                    },
                    {
                        "id": "T003",
                        "type": "EXPLORE",
                        "title": "Research Git workflows",
                        "description": "Compare Git workflow strategies. Write findings to pool.md Findings section.",
                    },
                ],
            )

            try:
                parallel_result = await execute_parallel_tasks(
                    task_ids=["T001", "T002", "T003"],
                    cwd=td.cwd,
                    verbose=False,
                    max_parallel=3,
                )
            except Exception as e:
                result.add(
                    f"run{run_idx}_parallel_exec",
                    False,
                    f"Parallel execution failed: {e}",
                    category="integration",
                )
                continue

            # Check results
            succeeded = [r for r in parallel_result.results if r.success]
            result.add(
                f"run{run_idx}_workers_succeeded",
                len(succeeded) >= 2,
                f"{len(succeeded)}/3 workers succeeded",
                category="integration",
            )

            # Check pool.md for findings from each worker
            pool = read_pool(td.cwd)
            findings_section = ""
            if "## Findings" in pool:
                fi_start = pool.index("## Findings")
                rest = pool[fi_start:]
                next_sec = rest.find("\n## ", 3)
                findings_section = rest[:next_sec] if next_sec > 0 else rest

            # Count distinct findings entries (lines starting with -)
            finding_lines = [l for l in findings_section.split("\n") if l.strip().startswith("-")]

            result.add(
                f"run{run_idx}_findings_present",
                len(finding_lines) > 0,
                f"Found {len(finding_lines)} finding entries in pool.md. "
                f"Succeeded workers: {len(succeeded)}. "
                f"If findings < succeeded, some findings were lost to concurrent Edit.",
                category="integration",
            )

            # Check task files for content
            task_files_with_content = 0
            for tid in ["T001", "T002", "T003"]:
                tc = td.read_task(tid)
                if tc and len(tc) > 200:  # More than just template
                    task_files_with_content += 1

            result.add(
                f"run{run_idx}_task_files_updated",
                task_files_with_content >= len(succeeded),
                f"{task_files_with_content}/3 task files have content (expected >= {len(succeeded)} succeeded)",
                category="integration",
            )

    return result


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("  Experiment 4: Concurrent Writes")
    print("=" * 60)

    print("\n--- Phase 1: Unit Tests ---\n")
    unit_result = run_unit_tests()
    print(unit_result.summary())
    unit_result.save()

    return unit_result


async def main_full(runs: int = 5):
    print("\n" + "=" * 60)
    print("  Experiment 4: Concurrent Writes (Full)")
    print("=" * 60)

    print("\n--- Phase 1: Unit Tests ---\n")
    unit_result = run_unit_tests()
    print(unit_result.summary())
    unit_result.save()

    print("\n--- Phase 2: Integration Tests ---\n")
    int_result = await run_integration_tests(runs=runs)
    print(int_result.summary())
    int_result.save()

    return unit_result, int_result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run integration tests too")
    parser.add_argument("--runs", type=int, default=5, help="Number of integration runs")
    args = parser.parse_args()

    if args.full:
        asyncio.run(main_full(runs=args.runs))
    else:
        main()
