"""
Tests for MCP tools module.

Tests the MCP tool functions directly (no Claude SDK needed).
Each tool is an async function — we use asyncio.run() to call them.
"""

import asyncio
from pathlib import Path

import pytest

from ralph_sdk.mcp_tools import (
    ALL_TOOL_NAMES,
    PLANNER_TOOL_NAMES,
    WORKER_TOOL_NAMES,
    _set_cwd,
    create_ralph_mcp_server,
    get_tool_names,
    ralph_add_verified as _ralph_add_verified,
    ralph_append_finding as _ralph_append_finding,
    ralph_check_verified as _ralph_check_verified,
    ralph_create_task as _ralph_create_task,
    ralph_get_pivot_signals as _ralph_get_pivot_signals,
    ralph_list_verified as _ralph_list_verified,
    ralph_log_progress as _ralph_log_progress,
    ralph_mark_pivot_processed as _ralph_mark_pivot_processed,
)

# SdkMcpTool objects are not callable — access .handler for direct testing
ralph_check_verified = _ralph_check_verified.handler
ralph_add_verified = _ralph_add_verified.handler
ralph_list_verified = _ralph_list_verified.handler
ralph_append_finding = _ralph_append_finding.handler
ralph_create_task = _ralph_create_task.handler
ralph_get_pivot_signals = _ralph_get_pivot_signals.handler
ralph_mark_pivot_processed = _ralph_mark_pivot_processed.handler
ralph_log_progress = _ralph_log_progress.handler
from ralph_sdk.pool import (
    append_to_findings,
    init_pool,
    init_ralph_dir,
    read_pool,
    read_task,
)


def _run(coro):
    """Run an async function synchronously."""
    return asyncio.run(coro)


@pytest.fixture
def ralph_dir(tmp_path):
    """Create a temporary .ralph/ directory with pool.md."""
    cwd = str(tmp_path)
    init_ralph_dir(cwd)
    init_pool(
        goal_summary="Test project",
        initial_tasks="| ID | Type | Status | Title |\n|---|---|---|---|\n| T001 | EXPLORE | pending | Initial exploration |",
        cwd=cwd,
    )
    _set_cwd(cwd)
    return cwd


# =============================================================================
# Server Factory Tests
# =============================================================================


class TestServerFactory:
    def test_create_server_all(self, ralph_dir):
        server = create_ralph_mcp_server(ralph_dir, role="all")
        assert server["type"] == "sdk"
        assert server["name"] == "ralph"
        assert server["instance"] is not None

    def test_create_server_worker(self, ralph_dir):
        server = create_ralph_mcp_server(ralph_dir, role="worker")
        assert server["type"] == "sdk"

    def test_create_server_planner(self, ralph_dir):
        server = create_ralph_mcp_server(ralph_dir, role="planner")
        assert server["type"] == "sdk"

    def test_tool_names_worker(self):
        names = get_tool_names("worker")
        assert "ralph_check_verified" in names
        assert "ralph_add_verified" in names
        assert "ralph_list_verified" in names
        assert "ralph_append_finding" in names
        assert "ralph_log_progress" in names
        assert "ralph_create_task" not in names
        assert "ralph_get_pivot_signals" not in names

    def test_tool_names_planner(self):
        names = get_tool_names("planner")
        assert "ralph_create_task" in names
        assert "ralph_get_pivot_signals" in names
        assert "ralph_mark_pivot_processed" in names
        assert "ralph_append_finding" in names
        assert "ralph_list_verified" in names
        assert "ralph_check_verified" not in names
        assert "ralph_log_progress" not in names

    def test_tool_names_all(self):
        names = get_tool_names("all")
        assert len(names) == 8
        assert "ralph_check_verified" in names
        assert "ralph_create_task" in names

    def test_tool_name_lists_consistent(self):
        assert len(WORKER_TOOL_NAMES) == 5
        assert len(PLANNER_TOOL_NAMES) == 5
        assert len(ALL_TOOL_NAMES) == 8


# =============================================================================
# Verified Info Cache Tests
# =============================================================================


class TestVerifiedInfoTools:
    def test_check_verified_not_found(self, ralph_dir):
        result = _run(ralph_check_verified({"topic": "nonexistent topic"}))
        text = result["content"][0]["text"]
        assert "NOT FOUND" in text

    def test_add_then_check_verified(self, ralph_dir):
        result = _run(ralph_add_verified({
            "topic": "transformers API",
            "finding": "Use AutoModelForCausalLM.from_pretrained() in v4.38+",
            "source_url": "https://huggingface.co/docs",
        }))
        text = result["content"][0]["text"]
        assert "cached" in text.lower()

        result = _run(ralph_check_verified({"topic": "transformers API"}))
        text = result["content"][0]["text"]
        assert "ALREADY VERIFIED" in text
        assert "from_pretrained" in text

    def test_list_verified_empty(self, ralph_dir):
        result = _run(ralph_list_verified({}))
        text = result["content"][0]["text"]
        assert "No topics verified" in text

    def test_list_verified_with_entries(self, ralph_dir):
        _run(ralph_add_verified({
            "topic": "topic A",
            "finding": "finding A",
            "source_url": "https://a.com",
        }))
        _run(ralph_add_verified({
            "topic": "topic B",
            "finding": "finding B",
            "source_url": "https://b.com",
        }))

        result = _run(ralph_list_verified({}))
        text = result["content"][0]["text"]
        assert "topic A" in text
        assert "topic B" in text
        assert "2" in text

    def test_check_verified_case_insensitive(self, ralph_dir):
        _run(ralph_add_verified({
            "topic": "Python API",
            "finding": "version 3.12",
            "source_url": "https://python.org",
        }))

        result = _run(ralph_check_verified({"topic": "python api"}))
        text = result["content"][0]["text"]
        assert "ALREADY VERIFIED" in text


# =============================================================================
# Findings Append Tests
# =============================================================================


class TestFindingsAppend:
    def test_append_finding(self, ralph_dir):
        result = _run(ralph_append_finding({
            "finding": "Redis v7 supports ACL2"
        }))
        text = result["content"][0]["text"]
        assert "appended" in text.lower()

        pool = read_pool(ralph_dir)
        assert "Redis v7 supports ACL2" in pool

    def test_append_multiple_findings(self, ralph_dir):
        _run(ralph_append_finding({"finding": "Finding 1"}))
        _run(ralph_append_finding({"finding": "Finding 2"}))
        _run(ralph_append_finding({"finding": "Finding 3"}))

        pool = read_pool(ralph_dir)
        assert "Finding 1" in pool
        assert "Finding 2" in pool
        assert "Finding 3" in pool


# =============================================================================
# Atomic Task Creation Tests
# =============================================================================


class TestAtomicTaskCreation:
    def test_create_task(self, ralph_dir):
        result = _run(ralph_create_task({
            "task_id": "T002",
            "task_type": "IMPLEMENT",
            "title": "Add caching layer",
            "description": "Implement Redis-based caching for API responses",
        }))
        text = result["content"][0]["text"]
        assert "T002" in text
        assert "atomically" in text.lower()

        task_content = read_task("T002", ralph_dir)
        assert task_content != ""
        assert "Add caching layer" in task_content
        assert "Redis-based caching" in task_content

        pool = read_pool(ralph_dir)
        assert "T002" in pool
        assert "IMPLEMENT" in pool
        assert "Add caching layer" in pool

    def test_create_explore_task(self, ralph_dir):
        _run(ralph_create_task({
            "task_id": "T003",
            "task_type": "EXPLORE",
            "title": "Research caching options",
            "description": "Compare Redis, Memcached, and in-memory caching",
        }))

        task_content = read_task("T003", ralph_dir)
        assert "EXPLORE" in task_content
        assert "Research caching options" in task_content

    def test_create_multiple_tasks(self, ralph_dir):
        for i in range(2, 5):
            _run(ralph_create_task({
                "task_id": f"T00{i}",
                "task_type": "EXPLORE",
                "title": f"Task {i}",
                "description": f"Description {i}",
            }))

        pool = read_pool(ralph_dir)
        assert "T002" in pool
        assert "T003" in pool
        assert "T004" in pool


# =============================================================================
# Pivot Signal Tests
# =============================================================================


class TestPivotSignals:
    def test_no_pivot_signals(self, ralph_dir):
        result = _run(ralph_get_pivot_signals({}))
        text = result["content"][0]["text"]
        assert "No pending pivot" in text

    def test_detect_pivot_signal(self, ralph_dir):
        append_to_findings(
            "**[PIVOT_RECOMMENDED]** T001: scores declining (40->35->30)",
            ralph_dir,
        )

        result = _run(ralph_get_pivot_signals({}))
        text = result["content"][0]["text"]
        assert "PENDING PIVOT" in text
        assert "T001" in text
        assert "declining" in text

    def test_mark_pivot_processed(self, ralph_dir):
        append_to_findings(
            "**[PIVOT_RECOMMENDED]** T001: scores declining",
            ralph_dir,
        )

        result = _run(ralph_mark_pivot_processed({"task_id": "T001"}))
        text = result["content"][0]["text"]
        assert "PIVOT_PROCESSED" in text

        result = _run(ralph_get_pivot_signals({}))
        text = result["content"][0]["text"]
        assert "No pending pivot" in text

    def test_multiple_pivot_signals(self, ralph_dir):
        append_to_findings(
            "**[PIVOT_RECOMMENDED]** T001: reason 1",
            ralph_dir,
        )
        append_to_findings(
            "**[PIVOT_RECOMMENDED]** T002: reason 2",
            ralph_dir,
        )

        result = _run(ralph_get_pivot_signals({}))
        text = result["content"][0]["text"]
        assert "2" in text
        assert "T001" in text
        assert "T002" in text

        # Mark only T001 as processed
        _run(ralph_mark_pivot_processed({"task_id": "T001"}))

        result = _run(ralph_get_pivot_signals({}))
        text = result["content"][0]["text"]
        assert "T002" in text
        pool = read_pool(ralph_dir)
        assert "[PIVOT_PROCESSED]" in pool


# =============================================================================
# Progress Logging Tests
# =============================================================================


class TestProgressLogging:
    def test_log_progress(self, ralph_dir):
        result = _run(ralph_log_progress({
            "entry": "Completed API integration"
        }))
        text = result["content"][0]["text"]
        assert "logged" in text.lower()

        pool = read_pool(ralph_dir)
        assert "Completed API integration" in pool


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    def test_full_verified_info_workflow(self, ralph_dir):
        """Simulate the full verified info workflow: check -> search -> cache -> recheck."""
        result = _run(ralph_check_verified({"topic": "PyTorch latest version"}))
        assert "NOT FOUND" in result["content"][0]["text"]

        _run(ralph_add_verified({
            "topic": "PyTorch latest version",
            "finding": "PyTorch 2.3.0 released 2024-04",
            "source_url": "https://pytorch.org/blog",
        }))

        result = _run(ralph_check_verified({"topic": "PyTorch latest version"}))
        assert "ALREADY VERIFIED" in result["content"][0]["text"]
        assert "2.3.0" in result["content"][0]["text"]

    def test_full_planner_workflow(self, ralph_dir):
        """Simulate planner: check pivots -> create tasks -> append findings."""
        result = _run(ralph_get_pivot_signals({}))
        assert "No pending" in result["content"][0]["text"]

        _run(ralph_create_task({
            "task_id": "T002",
            "task_type": "EXPLORE",
            "title": "Research alternatives",
            "description": "Explore alternative approaches",
        }))

        _run(ralph_append_finding({
            "finding": "Alternative approach X is viable"
        }))

        pool = read_pool(ralph_dir)
        assert "T002" in pool
        assert "Research alternatives" in pool
        assert "Alternative approach X" in pool

    def test_concurrent_findings_safe(self, ralph_dir):
        """Multiple concurrent append_finding calls should all succeed."""

        async def run_concurrent():
            tasks = [
                ralph_append_finding({"finding": f"Concurrent finding {i}"})
                for i in range(10)
            ]
            await asyncio.gather(*tasks)

        _run(run_concurrent())

        pool = read_pool(ralph_dir)
        for i in range(10):
            assert f"Concurrent finding {i}" in pool


# =============================================================================
# Prompt Integration Tests
# =============================================================================


class TestPromptIntegration:
    def test_worker_prompt_has_mcp_instructions(self):
        from ralph_sdk.prompts import WORKER_EXPLORE_PROMPT, WORKER_IMPLEMENT_PROMPT
        assert "ralph_check_verified" in WORKER_EXPLORE_PROMPT
        assert "ralph_add_verified" in WORKER_EXPLORE_PROMPT
        assert "ralph_append_finding" in WORKER_EXPLORE_PROMPT
        assert "ralph_check_verified" in WORKER_IMPLEMENT_PROMPT
        assert "ralph_add_verified" in WORKER_IMPLEMENT_PROMPT

    def test_planner_prompt_has_mcp_instructions(self):
        from ralph_sdk.prompts import PLANNER_SYSTEM_PROMPT
        assert "ralph_create_task" in PLANNER_SYSTEM_PROMPT
        assert "ralph_get_pivot_signals" in PLANNER_SYSTEM_PROMPT
        assert "ralph_mark_pivot_processed" in PLANNER_SYSTEM_PROMPT
        assert "ralph_append_finding" in PLANNER_SYSTEM_PROMPT
