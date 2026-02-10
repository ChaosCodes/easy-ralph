"""
Experiment 3: Pivot Signal Detection — Planner 检测 [PIVOT_RECOMMENDED] 可靠性

假设: Planner 手动解析 [PIVOT_RECOMMENDED] 标记不可靠。

测试层次:
1. Unit tests: pool.py pivot 函数正确性 + 格式变体
2. Integration tests: 真实 Planner 对各种 pivot 信号的响应

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
# Unit Tests: pool.py Pivot Functions
# =============================================================================

def run_unit_tests() -> ExperimentResult:
    result = ExperimentResult(
        experiment_name="exp3_pivot_signals_unit",
        description="Test pool.py pivot detection and processing functions",
    )

    from ralph_sdk.pool import (
        has_pivot_recommendation,
        clear_pivot_recommendation,
        append_to_findings,
        read_pool,
        write_pool,
    )

    # --- Test 1: has_pivot_recommendation basic ---
    with RalphTestDir("exp3_") as td:
        td.setup_basic_project(goal="Test")
        result.add(
            "no_pivot_initially",
            not has_pivot_recommendation(td.cwd),
            "",
        )

    # --- Test 2: detect after append ---
    with RalphTestDir("exp3_") as td:
        td.setup_basic_project(goal="Test")
        append_to_findings("**[PIVOT_RECOMMENDED]** T001: score declining", td.cwd)
        result.add(
            "detect_after_append",
            has_pivot_recommendation(td.cwd),
            "",
        )

    # --- Test 3: detect various formatting ---
    formats = [
        ("plain", "[PIVOT_RECOMMENDED] T001: reason"),
        ("bold", "**[PIVOT_RECOMMENDED]** T001: reason"),
        ("bold_no_space", "**[PIVOT_RECOMMENDED]**T001: reason"),
        ("with_timestamp", "- [2026-02-09 10:00] **[PIVOT_RECOMMENDED]** T001: declining scores"),
        ("extra_spaces", "[PIVOT_RECOMMENDED]  T001:  reason  with  spaces"),
        ("multiline", "[PIVOT_RECOMMENDED] T001: line1\n  continuation of reason"),
    ]

    for fmt_name, fmt_text in formats:
        with RalphTestDir(f"exp3_fmt_{fmt_name}_") as td:
            td.setup_basic_project(goal="Test")
            append_to_findings(fmt_text, td.cwd)
            detected = has_pivot_recommendation(td.cwd)
            result.add(
                f"detect_format_{fmt_name}",
                detected,
                f"Format '{fmt_name}': detected={detected}. Text: {fmt_text[:60]}",
                category="format_robustness",
            )

    # --- Test 4: clear_pivot_recommendation ---
    with RalphTestDir("exp3_") as td:
        td.setup_basic_project(goal="Test")
        append_to_findings("**[PIVOT_RECOMMENDED]** T001: test reason", td.cwd)
        assert has_pivot_recommendation(td.cwd), "Should have pivot before clear"
        clear_pivot_recommendation("T001", td.cwd)
        pool = read_pool(td.cwd)
        result.add(
            "clear_marks_as_processed",
            "[PIVOT_PROCESSED]" in pool and "[PIVOT_RECOMMENDED]" not in pool,
            f"Has PROCESSED: {'[PIVOT_PROCESSED]' in pool}, Has RECOMMENDED: {'[PIVOT_RECOMMENDED]' in pool}",
        )

    # --- Test 5: clear handles bold formatting ---
    with RalphTestDir("exp3_") as td:
        td.setup_basic_project(goal="Test")
        append_to_findings("**[PIVOT_RECOMMENDED]** T002: bold test", td.cwd)
        clear_pivot_recommendation("T002", td.cwd)
        pool = read_pool(td.cwd)
        result.add(
            "clear_handles_bold",
            "[PIVOT_PROCESSED]" in pool and "[PIVOT_RECOMMENDED]" not in pool,
            f"Pool excerpt: {pool[pool.find('PIVOT'):pool.find('PIVOT')+80] if 'PIVOT' in pool else 'no PIVOT found'}",
        )

    # --- Test 6: clear only specific task ---
    with RalphTestDir("exp3_") as td:
        td.setup_basic_project(goal="Test")
        append_to_findings("**[PIVOT_RECOMMENDED]** T001: task 1 issue", td.cwd)
        append_to_findings("**[PIVOT_RECOMMENDED]** T002: task 2 issue", td.cwd)
        clear_pivot_recommendation("T001", td.cwd)
        pool = read_pool(td.cwd)
        has_t1_recommended = bool(re.search(r'\[PIVOT_RECOMMENDED\].*T001', pool))
        has_t2_recommended = bool(re.search(r'\[PIVOT_RECOMMENDED\].*T002', pool))
        result.add(
            "clear_only_specific_task",
            not has_t1_recommended and has_t2_recommended,
            f"T001 still recommended: {has_t1_recommended}, T002 still recommended: {has_t2_recommended}",
        )

    # --- Test 7: multiple pivots detected ---
    with RalphTestDir("exp3_") as td:
        td.setup_basic_project(goal="Test")
        append_to_findings("[PIVOT_RECOMMENDED] T001: issue 1", td.cwd)
        append_to_findings("[PIVOT_RECOMMENDED] T002: issue 2", td.cwd)
        append_to_findings("[PIVOT_RECOMMENDED] T003: issue 3", td.cwd)
        result.add(
            "detect_multiple_pivots",
            has_pivot_recommendation(td.cwd),
            "",
        )
        pool = read_pool(td.cwd)
        count = pool.count("[PIVOT_RECOMMENDED]")
        result.add(
            "multiple_pivot_count",
            count == 3,
            f"Expected 3, found {count}",
        )

    # --- Test 8: PROCESSED not detected as pending ---
    with RalphTestDir("exp3_") as td:
        td.setup_basic_project(goal="Test")
        append_to_findings("[PIVOT_RECOMMENDED] T001: issue", td.cwd)
        clear_pivot_recommendation("T001", td.cwd)
        result.add(
            "processed_not_detected",
            not has_pivot_recommendation(td.cwd),
            "",
        )

    # --- Test 9: mix of RECOMMENDED and PROCESSED ---
    with RalphTestDir("exp3_") as td:
        td.setup_basic_project(goal="Test")
        append_to_findings("[PIVOT_RECOMMENDED] T001: done", td.cwd)
        append_to_findings("[PIVOT_RECOMMENDED] T002: pending", td.cwd)
        clear_pivot_recommendation("T001", td.cwd)
        pool = read_pool(td.cwd)
        has_rec = "[PIVOT_RECOMMENDED]" in pool
        has_proc = "[PIVOT_PROCESSED]" in pool
        result.add(
            "mix_recommended_and_processed",
            has_rec and has_proc,
            f"Has RECOMMENDED: {has_rec} (from T002), Has PROCESSED: {has_proc} (from T001)",
        )

    # --- Test 10: clear non-existent task is safe ---
    with RalphTestDir("exp3_") as td:
        td.setup_basic_project(goal="Test")
        append_to_findings("[PIVOT_RECOMMENDED] T001: issue", td.cwd)
        pool_before = read_pool(td.cwd)
        clear_pivot_recommendation("T999", td.cwd)
        pool_after = read_pool(td.cwd)
        # T001's recommendation should still be there
        result.add(
            "clear_nonexistent_safe",
            "[PIVOT_RECOMMENDED]" in pool_after,
            "",
        )

    # --- Test 11: pivot signal in different pool.md sections ---
    with RalphTestDir("exp3_") as td:
        td.setup_basic_project(goal="Test")
        # Add pivot signal outside Findings section (in Progress Log)
        pool = read_pool(td.cwd)
        pool += "\n### 2026-02-09 10:00\n[PIVOT_RECOMMENDED] T001: in progress log\n"
        write_pool(pool, td.cwd)
        result.add(
            "detect_outside_findings",
            has_pivot_recommendation(td.cwd),
            "has_pivot_recommendation searches entire pool.md, not just Findings section",
        )

    # --- Test 12: pool.md without Findings section ---
    with RalphTestDir("exp3_") as td:
        td.setup_basic_project(goal="Test")
        pool = read_pool(td.cwd)
        # Remove Findings section
        pool = re.sub(r'## Findings.*?(?=## |\Z)', '', pool, flags=re.DOTALL)
        write_pool(pool, td.cwd)
        # append_to_findings should create the section
        append_to_findings("[PIVOT_RECOMMENDED] T001: auto-created section", td.cwd)
        pool_after = read_pool(td.cwd)
        result.add(
            "append_creates_findings_section",
            "[PIVOT_RECOMMENDED]" in pool_after,
            f"Findings section created and pivot signal present",
        )

    # --- Test 13: Planner prompt has explicit pivot detection instructions ---
    from ralph_sdk.prompts import PLANNER_SYSTEM_PROMPT
    has_reading_section = "<reading_evaluator_signals>" in PLANNER_SYSTEM_PROMPT
    has_pivot_instruction = "PIVOT_RECOMMENDED" in PLANNER_SYSTEM_PROMPT
    has_must_respond = "MUST respond" in PLANNER_SYSTEM_PROMPT or "必须" in PLANNER_SYSTEM_PROMPT
    has_never_done = "NEVER choose DONE while" in PLANNER_SYSTEM_PROMPT

    result.add(
        "planner_prompt_has_pivot_instructions",
        has_reading_section and has_pivot_instruction,
        f"Has reading section: {has_reading_section}, mentions PIVOT_RECOMMENDED: {has_pivot_instruction}",
        category="design_gap",
    )
    result.add(
        "planner_prompt_requires_response",
        has_must_respond,
        f"Has MUST respond: {has_must_respond}. Without MCP, Planner must parse markdown to find this.",
        category="design_gap",
    )
    result.add(
        "planner_prompt_blocks_done",
        has_never_done,
        f"Has NEVER DONE rule: {has_never_done}. Orchestrator also has code-level check.",
        category="design_gap",
    )

    # --- Test 14: orchestrator has code-level DONE blocking ---
    import ralph_sdk.orchestrator as orch_mod
    orch_source = Path(orch_mod.__file__).read_text()
    blocks_done = "PIVOT_RECOMMENDED" in orch_source
    result.add(
        "orchestrator_blocks_done_on_pivot",
        blocks_done,
        f"Orchestrator checks for PIVOT_RECOMMENDED before allowing DONE: {blocks_done}. "
        "This is a code-level safety net for when Planner misses the signal.",
        category="design_gap",
    )

    # --- Test 15: pivot signal with unicode/special chars ---
    with RalphTestDir("exp3_") as td:
        td.setup_basic_project(goal="Test")
        append_to_findings(
            "**[PIVOT_RECOMMENDED]** T001: 分数连续下降 (40→35→30)，建议转向",
            td.cwd,
        )
        detected = has_pivot_recommendation(td.cwd)
        result.add(
            "detect_with_unicode",
            detected,
            f"Pivot with Chinese chars + arrows detected: {detected}",
            category="format_robustness",
        )

    # --- Test 16: empty pool.md ---
    with RalphTestDir("exp3_") as td:
        td.setup_basic_project(goal="Test")
        write_pool("", td.cwd)
        result.add(
            "empty_pool_no_error",
            not has_pivot_recommendation(td.cwd),
            "Should return False for empty pool without error",
        )

    return result


# =============================================================================
# Integration Tests: Real Planner Response to Pivot Signals
# =============================================================================

async def run_integration_tests(runs: int = 10) -> ExperimentResult:
    """
    Run real Planner with various pivot signals, check response.

    Scenarios:
    A. Single clear [PIVOT_RECOMMENDED] — Planner should respond with HEDGE/PIVOT
    B. Pivot buried in long Findings — Planner might miss it
    C. No pivot signal, all tasks done — Planner should DONE
    D. DONE attempted while pivot pending — should be blocked
    """
    from ralph_sdk.planner import plan, Action

    result = ExperimentResult(
        experiment_name="exp3_pivot_signals_integration",
        description=f"Test Planner pivot signal detection across {runs} runs per scenario",
    )

    PIVOT_ACTIONS = {Action.HEDGE, Action.PIVOT_RESEARCH, Action.PIVOT_ITERATION, Action.PIVOT_WAIT}

    # Scenario A: Single clear pivot signal
    for run_idx in range(runs):
        with RalphTestDir(f"exp3_intA_r{run_idx}_") as td:
            td.setup_basic_project(
                goal="Optimize image processing pipeline for speed.",
                tasks=[
                    {
                        "id": "T001",
                        "type": "IMPLEMENT",
                        "title": "Optimize with OpenCV",
                        "status": "completed",
                        "description": "Use OpenCV for image processing optimization.",
                    },
                ],
                pool_extra="""
- [2026-02-09 10:00] **[PIVOT_RECOMMENDED]** T001: 分数连续下降 (40→35→30), 当前 OpenCV 方案性能不达标
""",
            )
            # Make T001 look completed with poor score
            from ralph_sdk.pool import write_task, read_task
            task_content = read_task("T001", td.cwd)
            task_content = task_content.replace("## Status\npending", "## Status\ncompleted")
            task_content += "\n\n## Evaluation (2026-02-09)\n**Score**: 30/100\n"
            write_task("T001", task_content, td.cwd)

            try:
                decision = await plan(td.cwd, verbose=False)
            except Exception as e:
                result.add(f"scenA_run{run_idx}", False, f"Planner error: {e}", "integration")
                continue

            detected = decision.action in PIVOT_ACTIONS
            result.add(
                f"scenA_run{run_idx}_detects_pivot",
                detected,
                f"Action={decision.action.value}, Reason={decision.reason[:100]}",
                category="integration",
            )

            # Check if Planner marked as processed
            pool = td.read_pool()
            marked_processed = "[PIVOT_PROCESSED]" in pool
            result.add(
                f"scenA_run{run_idx}_marks_processed",
                marked_processed,
                f"Marked as PIVOT_PROCESSED: {marked_processed}",
                category="integration",
            )

    # Scenario B: Pivot buried in long Findings
    for run_idx in range(runs):
        with RalphTestDir(f"exp3_intB_r{run_idx}_") as td:
            long_findings = "\n".join([
                f"- [2026-02-0{i+1} 10:00] Finding {i}: Some detailed analysis about performance characteristics "
                f"that goes on for a while to add noise and make the pivot signal harder to spot."
                for i in range(8)
            ])
            td.setup_basic_project(
                goal="Build ML model for text classification.",
                tasks=[
                    {"id": "T001", "type": "IMPLEMENT", "title": "Train BERT model", "status": "completed"},
                    {"id": "T002", "type": "EXPLORE", "title": "Research alternatives", "status": "pending"},
                ],
                pool_extra=long_findings + "\n- [2026-02-09 15:00] **[PIVOT_RECOMMENDED]** T001: accuracy stuck at 60%\n",
            )
            from ralph_sdk.pool import write_task, read_task
            task_content = read_task("T001", td.cwd)
            task_content = task_content.replace("## Status\npending", "## Status\ncompleted")
            task_content += "\n\n## Evaluation\n**Score**: 55/100\nAccuracy stuck.\n"
            write_task("T001", task_content, td.cwd)

            try:
                decision = await plan(td.cwd, verbose=False)
            except Exception as e:
                result.add(f"scenB_run{run_idx}", False, f"Error: {e}", "integration")
                continue

            detected = decision.action in PIVOT_ACTIONS
            result.add(
                f"scenB_run{run_idx}_buried_pivot_detected",
                detected,
                f"Action={decision.action.value}. Pivot was buried under {len(long_findings.splitlines())} other findings.",
                category="integration",
            )

    # Scenario C: No pivot, all done — Planner should DONE
    for run_idx in range(runs):
        with RalphTestDir(f"exp3_intC_r{run_idx}_") as td:
            td.setup_basic_project(
                goal="Add logging to the application.",
                tasks=[
                    {"id": "T001", "type": "IMPLEMENT", "title": "Add logging", "status": "completed"},
                ],
            )
            from ralph_sdk.pool import write_task, read_task
            task_content = read_task("T001", td.cwd)
            task_content = task_content.replace("## Status\npending", "## Status\ncompleted")
            task_content += "\n\n## Evaluation\n**Score**: 95/100\nAll requirements met.\n"
            write_task("T001", task_content, td.cwd)

            try:
                decision = await plan(td.cwd, verbose=False)
            except Exception as e:
                result.add(f"scenC_run{run_idx}", False, f"Error: {e}", "integration")
                continue

            is_done = decision.action == Action.DONE
            result.add(
                f"scenC_run{run_idx}_done_without_pivot",
                is_done,
                f"Action={decision.action.value}. Should be DONE when no pivot signals exist.",
                category="integration",
            )

    # Scenario D: Multiple pivot signals for different tasks
    for run_idx in range(runs):
        with RalphTestDir(f"exp3_intD_r{run_idx}_") as td:
            td.setup_basic_project(
                goal="Build two independent features: caching and rate limiting.",
                tasks=[
                    {"id": "T001", "type": "IMPLEMENT", "title": "Add caching", "status": "completed"},
                    {"id": "T002", "type": "IMPLEMENT", "title": "Add rate limiting", "status": "completed"},
                ],
                pool_extra=(
                    "- [2026-02-09 10:00] **[PIVOT_RECOMMENDED]** T001: Redis approach failed\n"
                    "- [2026-02-09 10:01] **[PIVOT_RECOMMENDED]** T002: Token bucket overflow issues\n"
                ),
            )
            from ralph_sdk.pool import write_task, read_task
            for tid in ["T001", "T002"]:
                tc = read_task(tid, td.cwd)
                tc = tc.replace("## Status\npending", "## Status\ncompleted")
                tc += "\n\n## Evaluation\n**Score**: 35/100\n"
                write_task(tid, tc, td.cwd)

            try:
                decision = await plan(td.cwd, verbose=False)
            except Exception as e:
                result.add(f"scenD_run{run_idx}", False, f"Error: {e}", "integration")
                continue

            detected = decision.action in PIVOT_ACTIONS
            not_done = decision.action != Action.DONE
            result.add(
                f"scenD_run{run_idx}_multi_pivot_response",
                detected and not_done,
                f"Action={decision.action.value}, Target={decision.target}. "
                f"Should not DONE with 2 pending pivots.",
                category="integration",
            )

    return result


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("  Experiment 3: Pivot Signal Detection")
    print("=" * 60)

    print("\n--- Phase 1: Unit Tests ---\n")
    unit_result = run_unit_tests()
    print(unit_result.summary())
    unit_result.save()

    return unit_result


async def main_full(runs: int = 10):
    print("\n" + "=" * 60)
    print("  Experiment 3: Pivot Signal Detection (Full)")
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
    parser.add_argument("--runs", type=int, default=10, help="Number of runs per scenario")
    args = parser.parse_args()

    if args.full:
        asyncio.run(main_full(runs=args.runs))
    else:
        main()
