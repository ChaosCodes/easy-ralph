"""
Experiment 2: Atomic Task Creation — Planner CREATE 一致性验证

假设: Planner 在 CREATE action 时会遗漏创建 task file 或更新 pool.md。

测试层次:
1. Unit tests: pool.py 任务创建函数正确性、不一致检测
2. Integration tests: 真实 Planner CREATE 的一致性

测量指标:
- 问题是否真实存在？(Y/N + 证据)
- Agent 自己能做到什么程度？(0-100%)
- MCP 工具的预期提升？
"""

import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.mcp_validation.common import ExperimentResult, RalphTestDir


# =============================================================================
# Unit Tests: pool.py Task Creation Functions
# =============================================================================

def run_unit_tests() -> ExperimentResult:
    result = ExperimentResult(
        experiment_name="exp2_atomic_task_unit",
        description="Test pool.py task creation and consistency checking",
    )

    from ralph_sdk.pool import (
        init_task,
        write_pool,
        read_pool,
        read_task,
        task_exists,
        list_tasks,
        extract_task_ids_from_pool,
        ensure_task_files_exist,
        write_task,
    )

    # --- Test 1: init_task creates file correctly ---
    with RalphTestDir("exp2_") as td:
        td.setup_basic_project(goal="Test")
        init_task("T010", "EXPLORE", "Test Task", "Description here", td.cwd)
        exists = task_exists("T010", td.cwd)
        content = read_task("T010", td.cwd)
        result.add(
            "init_task_creates_file",
            exists and "Test Task" in content and "EXPLORE" in content,
            f"exists={exists}, has title={'Test Task' in content}, has type={'EXPLORE' in content}",
        )

    # --- Test 2: init_task IMPLEMENT type ---
    with RalphTestDir("exp2_") as td:
        td.setup_basic_project(goal="Test")
        init_task("T020", "IMPLEMENT", "Impl Task", "Build something", td.cwd)
        content = read_task("T020", td.cwd)
        result.add(
            "init_task_implement_type",
            "IMPLEMENT" in content and "Acceptance Criteria" in content,
            f"Has IMPLEMENT and Acceptance Criteria: {'IMPLEMENT' in content and 'Acceptance Criteria' in content}",
        )

    # --- Test 3: extract_task_ids_from_pool ---
    with RalphTestDir("exp2_") as td:
        td.setup_basic_project(
            goal="Test",
            tasks=[
                {"id": "T001", "title": "Task 1"},
                {"id": "T002", "title": "Task 2"},
                {"id": "T003", "title": "Task 3"},
            ],
        )
        ids = extract_task_ids_from_pool(td.cwd)
        result.add(
            "extract_ids_finds_all",
            set(ids) >= {"T001", "T002", "T003"},
            f"Found: {ids}",
        )

    # --- Test 4: detect inconsistency — task in pool but no file ---
    with RalphTestDir("exp2_") as td:
        td.setup_basic_project(goal="Test")
        # Manually add task ID to pool.md without creating file
        pool = read_pool(td.cwd)
        pool = pool.replace(
            "## Active Tasks\n\n(no tasks)",
            "## Active Tasks\n\n| ID | Type | Status | Title |\n|---|---|---|---|\n| T001 | EXPLORE | pending | Ghost task |",
        )
        write_pool(pool, td.cwd)

        ids_in_pool = extract_task_ids_from_pool(td.cwd)
        files_on_disk = td.list_task_files()
        in_pool_no_file = [tid for tid in ids_in_pool if tid not in files_on_disk]

        result.add(
            "detect_pool_without_file",
            len(in_pool_no_file) > 0 and "T001" in in_pool_no_file,
            f"Pool IDs: {ids_in_pool}, Files: {files_on_disk}, Missing: {in_pool_no_file}",
            category="consistency",
        )

    # --- Test 5: detect inconsistency — file exists but not in pool ---
    with RalphTestDir("exp2_") as td:
        td.setup_basic_project(goal="Test")
        # Create task file without adding to pool
        init_task("T099", "EXPLORE", "Orphan Task", "Not in pool", td.cwd)
        ids_in_pool = extract_task_ids_from_pool(td.cwd)
        files_on_disk = td.list_task_files()
        on_disk_not_in_pool = [f for f in files_on_disk if f not in ids_in_pool]

        result.add(
            "detect_file_without_pool_entry",
            len(on_disk_not_in_pool) > 0 and "T099" in on_disk_not_in_pool,
            f"Pool IDs: {ids_in_pool}, Files: {files_on_disk}, Orphans: {on_disk_not_in_pool}",
            category="consistency",
        )

    # --- Test 6: ensure_task_files_exist fixes missing files ---
    with RalphTestDir("exp2_") as td:
        td.setup_basic_project(goal="Test")
        pool = read_pool(td.cwd)
        pool = pool.replace(
            "## Active Tasks\n\n(no tasks)",
            "## Active Tasks\n\n| ID | Type | Status | Title |\n|---|---|---|---|\n"
            "| T001 | EXPLORE | pending | Task 1 |\n"
            "| T002 | IMPLEMENT | pending | Task 2 |\n"
            "| T003 | EXPLORE | pending | Task 3 |",
        )
        write_pool(pool, td.cwd)

        # Before: no task files
        files_before = td.list_task_files()
        # Run ensure
        created = ensure_task_files_exist(td.cwd)
        files_after = td.list_task_files()

        result.add(
            "ensure_creates_missing_files",
            set(created) == {"T001", "T002", "T003"},
            f"Created: {created}, Before: {files_before}, After: {files_after}",
        )

    # --- Test 7: ensure_task_files_exist is idempotent ---
    with RalphTestDir("exp2_") as td:
        td.setup_basic_project(
            goal="Test",
            tasks=[{"id": "T001", "title": "Task 1"}],
        )
        # Files already exist (created by setup_basic_project)
        created = ensure_task_files_exist(td.cwd)
        result.add(
            "ensure_idempotent_no_recreate",
            len(created) == 0,
            f"Should create nothing, created: {created}",
        )

    # --- Test 8: many tasks at once ---
    with RalphTestDir("exp2_") as td:
        td.setup_basic_project(goal="Test")
        task_lines = ["| ID | Type | Status | Title |", "|---|---|---|---|"]
        for i in range(1, 16):  # 15 tasks
            task_lines.append(f"| T{i:03d} | EXPLORE | pending | Task {i} |")
        pool = read_pool(td.cwd)
        pool = pool.replace("## Active Tasks\n\n(no tasks)", "## Active Tasks\n\n" + "\n".join(task_lines))
        write_pool(pool, td.cwd)
        created = ensure_task_files_exist(td.cwd)
        result.add(
            "ensure_handles_15_tasks",
            len(created) == 15,
            f"Expected 15 created, got {len(created)}: {created}",
        )

    # --- Test 9: pool.md task table format robustness ---
    # Test different table formats that Planner might produce
    formats_to_test = [
        # Standard format
        ("standard", "| T001 | EXPLORE | pending | Task 1 |"),
        # Extra spaces
        ("extra_spaces", "|  T001  |  EXPLORE  |  pending  |  Task 1  |"),
        # No trailing pipe
        ("no_trailing_pipe", "| T001 | EXPLORE | pending | Task 1"),
        # Markdown bold
        ("bold_id", "| **T001** | EXPLORE | pending | Task 1 |"),
        # Mixed case
        ("mixed_case", "| T001 | explore | Pending | task 1 |"),
        # With link
        ("with_link", "| [T001](tasks/T001.md) | EXPLORE | pending | Task 1 |"),
    ]

    for fmt_name, fmt_line in formats_to_test:
        with RalphTestDir(f"exp2_fmt_{fmt_name}_") as td:
            td.setup_basic_project(goal="Test")
            pool = read_pool(td.cwd)
            pool = pool.replace(
                "## Active Tasks\n\n(no tasks)",
                f"## Active Tasks\n\n| ID | Type | Status | Title |\n|---|---|---|---|\n{fmt_line}",
            )
            write_pool(pool, td.cwd)
            ids = extract_task_ids_from_pool(td.cwd)
            result.add(
                f"table_format_{fmt_name}",
                "T001" in ids,
                f"Format '{fmt_name}': extracted {ids}",
                category="robustness",
            )

    # --- Test 10: task ID format edge cases ---
    edge_ids = [
        ("T001", True, "standard 3-digit"),
        ("T999", True, "max 3-digit"),
        ("T1000", True, "4-digit"),
        ("T00001", True, "5-digit"),
    ]
    for tid, should_match, desc in edge_ids:
        with RalphTestDir(f"exp2_id_{tid}_") as td:
            td.setup_basic_project(goal="Test")
            pool = read_pool(td.cwd)
            pool = pool.replace(
                "## Active Tasks\n\n(no tasks)",
                f"## Active Tasks\n\n{tid}: Test task",
            )
            write_pool(pool, td.cwd)
            ids = extract_task_ids_from_pool(td.cwd)
            found = tid in ids
            result.add(
                f"id_format_{tid}",
                found == should_match,
                f"ID '{tid}' ({desc}): found={found}, expected={should_match}",
                category="robustness",
            )

    # --- Test 11: Planner prompt explicitly requires both pool.md + task file ---
    from ralph_sdk.prompts import PLANNER_SYSTEM_PROMPT
    mentions_both = (
        "pool.md" in PLANNER_SYSTEM_PROMPT
        and "task" in PLANNER_SYSTEM_PROMPT.lower()
        and ("NEVER" in PLANNER_SYSTEM_PROMPT or "MUST" in PLANNER_SYSTEM_PROMPT)
    )
    has_sync_rule = "NEVER add a task to pool.md without creating its task file" in PLANNER_SYSTEM_PROMPT
    result.add(
        "planner_prompt_has_sync_rule",
        has_sync_rule,
        f"Planner prompt has explicit sync rule: {has_sync_rule}",
        category="design_gap",
    )

    # --- Test 12: orchestrator has ensure_task_files_exist safety net ---
    import ralph_sdk.orchestrator as orch_mod
    orch_source = Path(orch_mod.__file__).read_text()
    uses_ensure = "ensure_task_files_exist" in orch_source
    result.add(
        "orchestrator_has_safety_net",
        uses_ensure,
        f"Orchestrator calls ensure_task_files_exist: {uses_ensure}. "
        "This is a code-level safety net but adds latency and creates minimal task files.",
        category="design_gap",
    )

    return result


# =============================================================================
# Integration Tests: Real Planner CREATE Actions
# =============================================================================

async def run_integration_tests(runs: int = 10) -> ExperimentResult:
    """
    Run real Planner in scenarios requiring CREATE, check consistency.

    Scenarios:
    A. Pool state that clearly needs new tasks
    B. Pool with completed EXPLORE that should trigger IMPLEMENT creation
    """
    from ralph_sdk.planner import plan, Action

    result = ExperimentResult(
        experiment_name="exp2_atomic_task_integration",
        description=f"Test Planner CREATE consistency across {runs} runs per scenario",
    )

    # Scenario A: Fresh project needing task decomposition
    for run_idx in range(runs):
        with RalphTestDir(f"exp2_intA_r{run_idx}_") as td:
            td.setup_basic_project(
                goal="""Build a REST API with three endpoints:
1. GET /users - list users
2. POST /users - create user
3. DELETE /users/:id - delete user

Each endpoint needs its own implementation and tests.
""",
                tasks=[
                    {
                        "id": "T001",
                        "type": "EXPLORE",
                        "title": "Research API framework options",
                        "status": "completed",
                        "description": "Research done. Recommend FastAPI.",
                    },
                ],
            )

            # Update T001 status to completed in task file
            from ralph_sdk.pool import write_task, read_task
            task_content = read_task("T001", td.cwd)
            task_content = task_content.replace("## Status\npending", "## Status\ncompleted")
            task_content += "\n\n## Findings\nRecommend FastAPI for the REST API. Need to create tasks for each endpoint.\n"
            write_task("T001", task_content, td.cwd)

            try:
                decision = await plan(td.cwd, verbose=False)
            except Exception as e:
                result.add(
                    f"scenA_run{run_idx}_planner",
                    False,
                    f"Planner failed: {e}",
                    category="integration",
                )
                continue

            # Check what action Planner chose
            action = decision.action.value

            if decision.action in (Action.CREATE, Action.DECOMPOSE):
                # Planner decided to create tasks — check consistency
                pool = td.read_pool()
                pool_ids = set(re.findall(r'\bT\d{3,}\b', pool))
                file_ids = set(td.list_task_files())

                # IDs in pool but no file
                pool_only = pool_ids - file_ids - {"T001"}  # T001 already exists
                # IDs with file but not in pool
                file_only = file_ids - pool_ids - {"T001"}

                consistent = len(pool_only) == 0 and len(file_only) == 0
                result.add(
                    f"scenA_run{run_idx}_consistency",
                    consistent,
                    f"Action={action}, Pool IDs={pool_ids}, File IDs={file_ids}, "
                    f"Pool-only={pool_only}, File-only={file_only}",
                    category="integration",
                )
            else:
                result.add(
                    f"scenA_run{run_idx}_action",
                    True,
                    f"Planner chose {action} instead of CREATE (may be valid). Reason: {decision.reason[:100]}",
                    category="integration",
                )

    # Scenario B: HEDGE action requiring new EXPLORE tasks
    for run_idx in range(runs):
        with RalphTestDir(f"exp2_intB_r{run_idx}_") as td:
            td.setup_basic_project(
                goal="""Optimize database query performance.
Current queries take 500ms, target is <50ms.
""",
                tasks=[
                    {
                        "id": "T001",
                        "type": "IMPLEMENT",
                        "title": "Add database indexing",
                        "status": "completed",
                        "description": "Add indexes to commonly queried columns.",
                    },
                ],
                pool_extra="""
- [2026-02-09 10:00] **[PIVOT_RECOMMENDED]** T001: 索引已添加但查询仍需 300ms，需要换方向
""",
            )

            # Update T001 to completed with evaluation showing needs improvement
            from ralph_sdk.pool import write_task, read_task
            task_content = read_task("T001", td.cwd)
            task_content = task_content.replace("## Status\npending", "## Status\ncompleted")
            task_content += """

## Evaluation (2026-02-09 10:00)

**Score**: 40/100

### Evaluator Full Output
Indexes added but query still takes 300ms. Need different approach.
"""
            write_task("T001", task_content, td.cwd)

            try:
                decision = await plan(td.cwd, verbose=False)
            except Exception as e:
                result.add(
                    f"scenB_run{run_idx}_planner",
                    False,
                    f"Planner failed: {e}",
                    category="integration",
                )
                continue

            action = decision.action.value
            is_pivot = decision.action in (
                Action.HEDGE, Action.PIVOT_RESEARCH, Action.PIVOT_ITERATION,
                Action.PIVOT_WAIT, Action.CREATE,
            )

            if is_pivot:
                pool = td.read_pool()
                pool_ids = set(re.findall(r'\bT\d{3,}\b', pool))
                file_ids = set(td.list_task_files())
                pool_only = pool_ids - file_ids - {"T001"}
                file_only = file_ids - pool_ids - {"T001"}
                consistent = len(pool_only) == 0

                result.add(
                    f"scenB_run{run_idx}_consistency",
                    consistent,
                    f"Action={action}, Pool-only (no file)={pool_only}, File-only={file_only}",
                    category="integration",
                )
            else:
                result.add(
                    f"scenB_run{run_idx}_action",
                    True,
                    f"Planner chose {action}. Reason: {decision.reason[:100]}",
                    category="integration",
                )

    return result


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("  Experiment 2: Atomic Task Creation")
    print("=" * 60)

    print("\n--- Phase 1: Unit Tests ---\n")
    unit_result = run_unit_tests()
    print(unit_result.summary())
    unit_result.save()

    return unit_result


async def main_full(runs: int = 10):
    print("\n" + "=" * 60)
    print("  Experiment 2: Atomic Task Creation (Full)")
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
    parser.add_argument("--runs", type=int, default=10, help="Number of integration runs per scenario")
    args = parser.parse_args()

    if args.full:
        asyncio.run(main_full(runs=args.runs))
    else:
        main()
