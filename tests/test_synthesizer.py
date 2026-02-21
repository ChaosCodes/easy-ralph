"""
Tests for the synthesizer module.

Tests KB read/write, findings rendering, prompt building, output parsing,
and pool helper functions.
No Claude SDK calls needed — we test the pure logic.
"""

import re
from pathlib import Path

import pytest

from ralph_sdk.pool import (
    SYNTHESIS_KB_TEMPLATE,
    count_completed_tasks_with_results,
    get_completed_task_summaries,
    init_pool,
    init_ralph_dir,
    init_task,
    read_pool,
    read_synthesis_kb,
    render_findings_from_kb,
    update_kb_experiment_status,
    update_pool_findings_from_kb,
    write_synthesis_kb,
    write_task,
)
from ralph_sdk.prompts import SYNTHESIZER_SYSTEM_PROMPT, build_synthesizer_prompt
from ralph_sdk.synthesizer import (
    Insight,
    ProposedExperiment,
    SynthesisResult,
    auto_promote_constraints,
    extract_kb_markdown,
    parse_kb_to_result,
)


@pytest.fixture
def ralph_dir(tmp_path):
    """Create a temporary .ralph/ directory with pool.md."""
    cwd = str(tmp_path)
    init_ralph_dir(cwd)
    init_pool(
        goal_summary="Improve benchmark score on cl-bench",
        initial_tasks=(
            "| ID | Type | Status | Title |\n"
            "|---|---|---|---|\n"
            "| T001 | EXPLORE | completed | Baseline analysis |\n"
            "| T002 | IMPLEMENT | completed | Dual agent approach |\n"
            "| T003 | IMPLEMENT | pending | New approach |"
        ),
        cwd=cwd,
    )
    return cwd


def _create_completed_task(cwd, task_id, title, execution_log, findings=""):
    """Helper to create a completed task file with execution log."""
    content = f"""# {task_id}: {title}

## Type
IMPLEMENT

## Status
completed

## Created
2024-01-26 10:00

## Description
Test task

## Execution Log

{execution_log}

## Findings

{findings}
"""
    write_task(task_id, content, cwd)


def _create_pending_task(cwd, task_id, title):
    """Helper to create a pending task file."""
    init_task(task_id, "IMPLEMENT", title, "Pending task", cwd)


SAMPLE_KB = """# Synthesis KB
> Last updated: 2026-02-21 08:35 | Synthesis round: 7

## Strategic Summary
- **Current best (non-circular)**: 52.5% (category routing)
- **Target**: >60%
- **Gap root cause**: No non-circular routing signal
- **Top action**: E001 Stage 3 Isolation ($0, expected +5pp)

## Active Insights

### I001 [P1, high, 9x confirmed]
9/9 post-gen modifications always degrade performance under all-or-nothing grading.
→ Hard constraint: only first-pass generation or unmodified selection.
First: T014 | Evidence: T007,T014,T017,T025,T027,T028,T031,T034 | Updated: T034

### I002 [P2, high, 6x confirmed]
Capability additions crack always-fail items; architecture variations do not.
→ New experiments must pass the "new capability test."
First: T030 | Evidence: T030,T038,T041,T042,T046 | Updated: T046

### I003 [P3, medium]
EVE validation paradox: negative correlation with success yet +12.5pp vs no-validation.
→ Stage 3 isolation experiment needed.
First: T019 | Updated: T031

## Superseded
- ~~I005~~: "Oracle ceiling 70%" → superseded by I002 (now 82.5%). Reason: CV+styles.

## Experiments

### Proposed
| ID | Priority | Name | Tests | Expected Gain | Cost | Times Proposed |
|----|----------|------|-------|---------------|------|----------------|
| E001 | P1 | Stage 3 Isolation | I003 | +5pp | $0 | 4 |
| E002 | P2 | Temperature=0 Variance | I004 | +3pp | $5 | 3 |
| E003 | P3 | Question-Structure Classifier | I006 | +2pp | $10 | 2 |

### Executing
| ID | Task | Name |
|----|------|------|
| E004 | T049 | Temperature=0 Dual Agent |

### Completed
| ID | Task | Name | Result |
|----|------|------|--------|
| E010 | T044 | Constraint Full-40 Standalone | 12.8%, FALSIFIED |

### Abandoned
| ID | Reason |
|----|--------|
| E020 | Document preprocessing — LLM tokenizers handle markdown fine |
"""


# ---------------------------------------------------------------------------
# Tests for Synthesis KB Read/Write
# ---------------------------------------------------------------------------


class TestSynthesisKBReadWrite:
    def test_read_nonexistent_returns_template(self, ralph_dir):
        kb = read_synthesis_kb(ralph_dir)
        assert "# Synthesis KB" in kb
        assert "Synthesis round: 0" in kb
        assert "(none yet)" in kb

    def test_write_and_read_roundtrip(self, ralph_dir):
        write_synthesis_kb(SAMPLE_KB, ralph_dir)
        kb = read_synthesis_kb(ralph_dir)
        assert kb == SAMPLE_KB
        assert "52.5% (category routing)" in kb
        assert "I001" in kb

    def test_template_structure(self):
        """Ensure template has all required sections."""
        for section in [
            "## Strategic Summary",
            "## Active Insights",
            "## Superseded",
            "## Experiments",
            "### Proposed",
            "### Executing",
            "### Completed",
            "### Abandoned",
        ]:
            assert section in SYNTHESIS_KB_TEMPLATE


# ---------------------------------------------------------------------------
# Tests for render_findings_from_kb
# ---------------------------------------------------------------------------


class TestRenderFindingsFromKB:
    def test_renders_strategic_summary(self):
        rendered = render_findings_from_kb(SAMPLE_KB)
        assert "Strategic Summary" in rendered
        assert "52.5%" in rendered
        assert ">60%" in rendered

    def test_renders_top_insights(self):
        rendered = render_findings_from_kb(SAMPLE_KB)
        assert "Key Insights" in rendered
        assert "I001" in rendered
        assert "I002" in rendered
        assert "I003" in rendered

    def test_limits_to_5_insights(self):
        """Should not render more than 5 insights."""
        # Add 8 insights to KB
        many_insights = "## Active Insights\n\n"
        for i in range(1, 9):
            many_insights += f"### I{i:03d} [P{min(i, 3)}, medium]\nInsight {i}.\n→ Action {i}.\n\n"
        kb = f"# Synthesis KB\n> Last updated: now\n\n## Strategic Summary\nTest\n\n{many_insights}\n## Superseded\n## Experiments\n### Proposed\n(none yet)\n### Executing\n(none yet)\n### Completed\n(none yet)\n### Abandoned\n(none yet)\n"

        rendered = render_findings_from_kb(kb)
        # Count insight references (I001..I008)
        insight_refs = re.findall(r"I\d{3}", rendered)
        assert len(insight_refs) <= 5

    def test_renders_experiments(self):
        rendered = render_findings_from_kb(SAMPLE_KB)
        assert "E001" in rendered
        # Executing experiment should be marked
        assert "executing" in rendered.lower()

    def test_output_under_2kb(self):
        rendered = render_findings_from_kb(SAMPLE_KB)
        assert len(rendered) < 2048

    def test_includes_kb_reference(self):
        rendered = render_findings_from_kb(SAMPLE_KB)
        assert "synthesis_kb.md" in rendered

    def test_empty_kb_renders_gracefully(self):
        rendered = render_findings_from_kb(SYNTHESIS_KB_TEMPLATE)
        assert "synthesis_kb.md" in rendered
        # Should not crash on empty template


# ---------------------------------------------------------------------------
# Tests for update_pool_findings_from_kb
# ---------------------------------------------------------------------------


class TestUpdatePoolFindingsFromKB:
    def test_replaces_findings_section(self, ralph_dir):
        write_synthesis_kb(SAMPLE_KB, ralph_dir)
        update_pool_findings_from_kb(ralph_dir)
        pool = read_pool(ralph_dir)
        assert "Synthesis KB" in pool
        assert "52.5%" in pool

    def test_preserves_pivot_markers(self, ralph_dir):
        # Inject a PIVOT_RECOMMENDED marker into findings
        pool = read_pool(ralph_dir)
        pool = pool.replace(
            "## Findings\n\n(discoveries shared across tasks)",
            "## Findings\n\n**[PIVOT_RECOMMENDED]** T001: score declining\n(discoveries shared across tasks)",
        )
        from ralph_sdk.pool import write_pool
        write_pool(pool, ralph_dir)

        write_synthesis_kb(SAMPLE_KB, ralph_dir)
        update_pool_findings_from_kb(ralph_dir)

        pool_after = read_pool(ralph_dir)
        assert "[PIVOT_RECOMMENDED]" in pool_after
        assert "Synthesis KB" in pool_after


# ---------------------------------------------------------------------------
# Tests for KB experiment status transitions
# ---------------------------------------------------------------------------


class TestKBExperimentStatus:
    def test_proposed_to_executing(self, ralph_dir):
        write_synthesis_kb(SAMPLE_KB, ralph_dir)
        update_kb_experiment_status(
            experiment_id="E001",
            from_section="Proposed",
            to_section="Executing",
            extra_cols="| T050 | Stage 3 Isolation",
            cwd=ralph_dir,
        )
        kb = read_synthesis_kb(ralph_dir)
        # E001 should no longer be in Proposed
        proposed_match = re.search(r"### Proposed\n(.*?)(?=### )", kb, re.DOTALL)
        if proposed_match:
            assert "E001" not in proposed_match.group(1)
        # E001 should be in Executing
        executing_match = re.search(r"### Executing\n(.*?)(?=### )", kb, re.DOTALL)
        assert executing_match
        assert "E001" in executing_match.group(1)
        assert "T050" in executing_match.group(1)

    def test_executing_to_completed(self, ralph_dir):
        write_synthesis_kb(SAMPLE_KB, ralph_dir)
        update_kb_experiment_status(
            experiment_id="E004",
            from_section="Executing",
            to_section="Completed",
            extra_cols="| 55%, CONFIRMED",
            cwd=ralph_dir,
        )
        kb = read_synthesis_kb(ralph_dir)
        # E004 should no longer be in Executing
        executing_match = re.search(r"### Executing\n(.*?)(?=### )", kb, re.DOTALL)
        if executing_match:
            assert "E004" not in executing_match.group(1)
        # E004 should be in Completed
        completed_match = re.search(r"### Completed\n(.*?)(?=### )", kb, re.DOTALL)
        assert completed_match
        assert "E004" in completed_match.group(1)

    def test_nonexistent_experiment_noop(self, ralph_dir):
        write_synthesis_kb(SAMPLE_KB, ralph_dir)
        original = read_synthesis_kb(ralph_dir)
        update_kb_experiment_status(
            experiment_id="E999",
            from_section="Proposed",
            to_section="Executing",
            cwd=ralph_dir,
        )
        after = read_synthesis_kb(ralph_dir)
        assert original == after


# ---------------------------------------------------------------------------
# Tests for extract_kb_markdown
# ---------------------------------------------------------------------------


class TestExtractKBMarkdown:
    def test_raw_markdown(self):
        text = "Some preamble\n\n# Synthesis KB\n> Last updated: now\n\n## Strategic Summary\nTest"
        result = extract_kb_markdown(text)
        assert result.startswith("# Synthesis KB")
        assert "Strategic Summary" in result

    def test_fenced_markdown(self):
        text = "Here is the KB:\n\n```markdown\n# Synthesis KB\n> Last updated: now\n\n## Strategic Summary\nTest\n```\n\nDone."
        result = extract_kb_markdown(text)
        assert result.startswith("# Synthesis KB")
        assert "Strategic Summary" in result

    def test_fallback_returns_text(self):
        text = "No KB here, just random text"
        result = extract_kb_markdown(text)
        assert result == text.strip()


# ---------------------------------------------------------------------------
# Tests for parse_kb_to_result
# ---------------------------------------------------------------------------


class TestParseKBToResult:
    def test_parses_insights(self):
        result = parse_kb_to_result(SAMPLE_KB)
        assert len(result.insights) == 3
        assert result.insights[0].confidence == "high"
        assert result.insights[2].confidence == "medium"

    def test_parses_proposed_experiments(self):
        result = parse_kb_to_result(SAMPLE_KB)
        assert len(result.proposed_experiments) == 3
        assert result.proposed_experiments[0].name == "Stage 3 Isolation"

    def test_p1_is_diagnostic(self):
        result = parse_kb_to_result(SAMPLE_KB)
        assert result.proposed_experiments[0].is_diagnostic is True

    def test_stores_kb_content(self):
        result = parse_kb_to_result(SAMPLE_KB)
        assert result.kb_content == SAMPLE_KB


# ---------------------------------------------------------------------------
# Tests for auto_promote_constraints
# ---------------------------------------------------------------------------


class TestAutoPromoteConstraints:
    def test_promotes_high_confidence_negative(self, ralph_dir):
        auto_promote_constraints(SAMPLE_KB, ralph_dir)
        pool = read_pool(ralph_dir)
        # I001 has "9/9" and high confidence — should be promoted
        assert "Hard Constraints" in pool
        assert "Auto-promoted from synthesis" in pool

    def test_skips_medium_confidence(self, ralph_dir):
        kb_medium = """# Synthesis KB
> Last updated: now

## Active Insights

### I099 [P1, medium, 2x confirmed]
Something always fails.
→ Don't do it.

## Superseded
## Experiments
### Proposed
(none yet)
### Executing
(none yet)
### Completed
(none yet)
### Abandoned
(none yet)
"""
        auto_promote_constraints(kb_medium, ralph_dir)
        pool = read_pool(ralph_dir)
        assert "Hard Constraints" not in pool


# ---------------------------------------------------------------------------
# Tests for count_completed_tasks_with_results
# ---------------------------------------------------------------------------


class TestCountCompletedTasksWithResults:
    def test_no_tasks(self, ralph_dir):
        assert count_completed_tasks_with_results(ralph_dir) == 0

    def test_completed_with_log(self, ralph_dir):
        _create_completed_task(
            ralph_dir, "T001", "Task 1",
            execution_log="Step 1: Ran baseline\nStep 2: Got score 42.5%",
        )
        assert count_completed_tasks_with_results(ralph_dir) == 1

    def test_completed_without_log(self, ralph_dir):
        """Completed task with empty/placeholder log should not count."""
        content = """# T001: Task 1

## Type
IMPLEMENT

## Status
completed

## Execution Log

(execution details will be recorded here)
"""
        write_task("T001", content, ralph_dir)
        assert count_completed_tasks_with_results(ralph_dir) == 0

    def test_pending_task_not_counted(self, ralph_dir):
        _create_pending_task(ralph_dir, "T001", "Task 1")
        assert count_completed_tasks_with_results(ralph_dir) == 0

    def test_multiple_tasks_mixed(self, ralph_dir):
        _create_completed_task(
            ralph_dir, "T001", "Task 1",
            execution_log="Ran experiment, got 42.5%",
        )
        _create_completed_task(
            ralph_dir, "T002", "Task 2",
            execution_log="Tried dual agent, got 35%",
        )
        _create_pending_task(ralph_dir, "T003", "Task 3")
        assert count_completed_tasks_with_results(ralph_dir) == 2


# ---------------------------------------------------------------------------
# Tests for get_completed_task_summaries
# ---------------------------------------------------------------------------


class TestGetCompletedTaskSummaries:
    def test_empty(self, ralph_dir):
        assert get_completed_task_summaries(ralph_dir) == []

    def test_returns_summaries(self, ralph_dir):
        _create_completed_task(
            ralph_dir, "T001", "Baseline",
            execution_log="Step 1: Got 42.5%",
            findings="EVE method works best on short docs",
        )
        summaries = get_completed_task_summaries(ralph_dir)
        assert len(summaries) == 1
        assert summaries[0]["task_id"] == "T001"
        assert summaries[0]["status"] == "completed"
        assert "42.5%" in summaries[0]["execution_log"]
        assert "EVE method" in summaries[0]["findings"]


# ---------------------------------------------------------------------------
# Tests for build_synthesizer_prompt
# ---------------------------------------------------------------------------


class TestBuildSynthesizerPrompt:
    def test_includes_kb_content(self):
        prompt = build_synthesizer_prompt(
            goal="Improve cl-bench score",
            pool="## Findings\n- EVE scored 42.5%",
            task_summaries=[
                {
                    "task_id": "T001",
                    "status": "completed",
                    "execution_log": "Ran EVE, scored 42.5%",
                    "findings": "Works on short docs",
                },
            ],
            kb_content=SAMPLE_KB,
        )
        assert "Improve cl-bench score" in prompt
        assert "Knowledge Base" in prompt
        assert "52.5% (category routing)" in prompt
        assert "T001" in prompt

    def test_recent_tasks_only(self):
        """With >5 tasks, only last 5 should be in detail."""
        tasks = [
            {
                "task_id": f"T{i:03d}",
                "status": "completed",
                "execution_log": f"Task {i} log",
                "findings": "",
            }
            for i in range(1, 9)
        ]
        prompt = build_synthesizer_prompt(
            goal="goal",
            pool="pool",
            task_summaries=tasks,
            kb_content="# Synthesis KB\n",
        )
        # Recent 5 (T004-T008) should be in detail
        assert "T008" in prompt
        assert "T004" in prompt
        # Older count mentioned
        assert "3 older" in prompt

    def test_empty_kb(self):
        prompt = build_synthesizer_prompt(
            goal="goal",
            pool="pool",
            task_summaries=[],
            kb_content="",
        )
        assert "Knowledge Base" in prompt


# ---------------------------------------------------------------------------
# Tests for SYNTHESIZER_SYSTEM_PROMPT content
# ---------------------------------------------------------------------------


class TestSynthesizerPrompt:
    def test_has_convergence_rules(self):
        assert "Convergence Rules" in SYNTHESIZER_SYSTEM_PROMPT
        assert "Max 10 active insights" in SYNTHESIZER_SYSTEM_PROMPT
        assert "Max 3 proposed experiments" in SYNTHESIZER_SYSTEM_PROMPT

    def test_has_kb_awareness(self):
        assert "MEMORY" in SYNTHESIZER_SYSTEM_PROMPT
        assert "Knowledge Base" in SYNTHESIZER_SYSTEM_PROMPT
        assert "NOT to re-analyze everything" in SYNTHESIZER_SYSTEM_PROMPT

    def test_has_strategic_convergence(self):
        assert "Strategic Convergence" in SYNTHESIZER_SYSTEM_PROMPT
        assert "MOST IMPORTANT" in SYNTHESIZER_SYSTEM_PROMPT
        assert "AT MOST 1 blind spot" in SYNTHESIZER_SYSTEM_PROMPT

    def test_has_methodology_references(self):
        assert "Schulman" in SYNTHESIZER_SYSTEM_PROMPT
        assert "Karpathy" in SYNTHESIZER_SYSTEM_PROMPT
        assert "Sasha Rush" in SYNTHESIZER_SYSTEM_PROMPT

    def test_output_is_markdown_not_json(self):
        """New prompt should ask for markdown, not JSON."""
        assert "Output the COMPLETE updated synthesis_kb.md" in SYNTHESIZER_SYSTEM_PROMPT
        assert "Do NOT output JSON" in SYNTHESIZER_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Tests for SynthesisResult dataclass
# ---------------------------------------------------------------------------


class TestSynthesisResult:
    def test_default_empty(self):
        result = SynthesisResult()
        assert result.insights == []
        assert result.proposed_experiments == []
        assert result.kb_content == ""
        assert result.result_stats is None

    def test_with_data(self):
        result = SynthesisResult(
            insights=[
                Insight(
                    observation="obs",
                    implication="imp",
                    confidence="high",
                ),
            ],
            proposed_experiments=[
                ProposedExperiment(
                    name="exp1",
                    insight_tested="obs",
                    method="do something",
                    is_diagnostic=True,
                ),
            ],
        )
        assert len(result.insights) == 1
        assert result.insights[0].confidence == "high"
        assert len(result.proposed_experiments) == 1
        assert result.proposed_experiments[0].is_diagnostic is True
