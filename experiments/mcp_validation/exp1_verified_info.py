"""
Experiment 1: Verified Info — 重复搜索问题验证

假设: 多个 worker 处理相关任务时，会对同一话题重复 WebSearch。

测试层次:
1. Unit tests: pool.py verified info 函数正确性
2. Integration tests: 真实 worker 是否重复搜索同一话题

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
# Unit Tests: pool.py Verified Info Functions
# =============================================================================

def run_unit_tests() -> ExperimentResult:
    result = ExperimentResult(
        experiment_name="exp1_verified_info_unit",
        description="Test pool.py verified info functions",
    )

    from ralph_sdk.pool import (
        add_verified_info,
        get_verified_info,
        is_topic_verified,
        list_verified_topics,
        read_pool,
    )

    # --- Test 1: add + get basic ---
    with RalphTestDir("exp1_") as td:
        td.setup_basic_project(goal="Test verified info")
        add_verified_info("requests library", "latest version is 2.31.0", "https://pypi.org/project/requests/", td.cwd)
        info = get_verified_info("requests library", td.cwd)
        result.add(
            "add_then_get_basic",
            info is not None and "2.31.0" in info,
            f"Expected finding with '2.31.0', got: {info}",
        )

    # --- Test 2: get non-existent topic ---
    with RalphTestDir("exp1_") as td:
        td.setup_basic_project(goal="Test verified info")
        info = get_verified_info("nonexistent topic", td.cwd)
        result.add(
            "get_nonexistent_returns_none",
            info is None,
            f"Expected None, got: {info}",
        )

    # --- Test 3: is_topic_verified ---
    with RalphTestDir("exp1_") as td:
        td.setup_basic_project(goal="Test verified info")
        result.add(
            "is_topic_verified_false_initially",
            not is_topic_verified("requests", td.cwd),
            "",
        )
        add_verified_info("requests", "v2.31", "https://example.com", td.cwd)
        result.add(
            "is_topic_verified_true_after_add",
            is_topic_verified("requests", td.cwd),
            "",
        )

    # --- Test 4: list_verified_topics ---
    with RalphTestDir("exp1_") as td:
        td.setup_basic_project(goal="Test")
        result.add(
            "list_topics_empty_initially",
            len(list_verified_topics(td.cwd)) == 0,
            f"Got: {list_verified_topics(td.cwd)}",
        )
        add_verified_info("topic A", "info A", "http://a.com", td.cwd)
        add_verified_info("topic B", "info B", "http://b.com", td.cwd)
        add_verified_info("topic C", "info C", "http://c.com", td.cwd)
        topics = list_verified_topics(td.cwd)
        result.add(
            "list_topics_three_after_add",
            len(topics) == 3,
            f"Expected 3 topics, got {len(topics)}: {topics}",
        )

    # --- Test 5: case-insensitive matching ---
    with RalphTestDir("exp1_") as td:
        td.setup_basic_project(goal="Test")
        add_verified_info("Python Requests", "v2.31", "http://x.com", td.cwd)
        result.add(
            "get_case_insensitive",
            get_verified_info("python requests", td.cwd) is not None,
            "",
        )

    # --- Test 6: multiple adds don't corrupt pool.md ---
    with RalphTestDir("exp1_") as td:
        td.setup_basic_project(goal="Test")
        for i in range(10):
            add_verified_info(f"topic_{i}", f"finding_{i}", f"http://url{i}.com", td.cwd)
        pool = read_pool(td.cwd)
        # Check all entries present
        all_present = all(f"topic_{i}" in pool for i in range(10))
        # Check pool structure intact
        has_sections = "## Active Tasks" in pool and "## Progress Log" in pool
        result.add(
            "ten_adds_no_corruption",
            all_present and has_sections,
            f"all_present={all_present}, has_sections={has_sections}",
        )

    # --- Test 7: add to pool without Verified Information section ---
    with RalphTestDir("exp1_") as td:
        td.setup_basic_project(goal="Test")
        # Remove the Verified Information section
        pool = read_pool(td.cwd)
        pool = pool.replace("## Verified Information (时效性验证缓存)", "")
        pool = pool.replace("<!--\nFormat:", "").replace("-->", "")
        pool = pool.replace("Workers should check here before searching to avoid duplicate queries.", "")
        from ralph_sdk.pool import write_pool
        write_pool(pool, td.cwd)

        # Should still work — creates the section
        add_verified_info("new topic", "finding", "http://x.com", td.cwd)
        pool_after = read_pool(td.cwd)
        result.add(
            "add_creates_section_if_missing",
            "new topic" in pool_after and "Verified Information" in pool_after,
            f"topic in pool: {'new topic' in pool_after}, section: {'Verified Information' in pool_after}",
        )

    # --- Test 8: (none yet) placeholder gets removed ---
    with RalphTestDir("exp1_") as td:
        td.setup_basic_project(goal="Test")
        pool_before = read_pool(td.cwd)
        has_none_yet_before = "(none yet)" in pool_before
        add_verified_info("test topic", "finding", "http://x.com", td.cwd)
        pool_after = read_pool(td.cwd)
        # After adding, "(none yet)" in the Verified section should be gone
        # But "(none yet)" in Completed section is fine
        vi_section = ""
        if "## Verified Information" in pool_after:
            vi_start = pool_after.index("## Verified Information")
            rest = pool_after[vi_start:]
            next_section = rest.find("\n## ", 3)
            vi_section = rest[:next_section] if next_section > 0 else rest
        result.add(
            "none_yet_removed_in_verified_section",
            "(none yet)" not in vi_section,
            f"Verified section still has '(none yet)': {vi_section[:200]}",
        )

    # --- Test 9: partial topic match ---
    with RalphTestDir("exp1_") as td:
        td.setup_basic_project(goal="Test")
        add_verified_info("Python requests library version", "2.31", "http://x.com", td.cwd)
        # Exact match
        result.add(
            "exact_topic_match",
            get_verified_info("Python requests library version", td.cwd) is not None,
            "",
        )
        # The function uses re.escape(topic) which means it does exact substring match
        # "requests" alone should NOT match "Python requests library version"
        # because re.escape builds a pattern for exact match
        partial = get_verified_info("requests", td.cwd)
        result.add(
            "partial_topic_no_match",
            partial is None,
            f"Partial match returned: {partial}",
        )

    # --- Test 10: worker.py imports verified info functions ---
    with RalphTestDir("exp1_") as td:
        import ralph_sdk.worker as worker_mod
        has_add = hasattr(worker_mod, 'add_verified_info') or 'add_verified_info' in dir(worker_mod)
        has_get = hasattr(worker_mod, 'get_verified_info') or 'get_verified_info' in dir(worker_mod)
        has_is = hasattr(worker_mod, 'is_topic_verified') or 'is_topic_verified' in dir(worker_mod)

        # Check import statements in worker.py source
        worker_source = Path(worker_mod.__file__).read_text()
        imports_add = "add_verified_info" in worker_source
        imports_get = "get_verified_info" in worker_source
        imports_is = "is_topic_verified" in worker_source

        # Check if they're actually CALLED (not just imported)
        # Strip entire import blocks (including multi-line from...import)
        import re as _re
        # Remove all `from X import (...)` blocks and `import X` lines
        cleaned = _re.sub(
            r'^from\s+\S+\s+import\s*\(.*?\)',
            '',
            worker_source,
            flags=_re.MULTILINE | _re.DOTALL,
        )
        cleaned = _re.sub(r'^(?:from|import)\s+.*$', '', cleaned, flags=_re.MULTILINE)
        called_add = "add_verified_info" in cleaned
        called_get = "get_verified_info" in cleaned
        called_is = "is_topic_verified" in cleaned

        result.add(
            "worker_imports_verified_functions",
            imports_add and imports_get and imports_is,
            f"imports: add={imports_add}, get={imports_get}, is={imports_is}",
        )
        result.add(
            "worker_never_calls_verified_functions",
            not called_add and not called_get and not called_is,
            f"calls: add={called_add}, get={called_get}, is={called_is}. "
            "This confirms the design gap: functions imported but never called by worker code.",
            category="design_gap",
        )

    # --- Test 11: prompt mentions verified info but no mechanism to call it ---
    with RalphTestDir("exp1_") as td:
        from ralph_sdk.prompts import WORKER_EXPLORE_PROMPT, WORKER_IMPLEMENT_PROMPT
        explore_mentions = "verified" in WORKER_EXPLORE_PROMPT.lower() or "验证" in WORKER_EXPLORE_PROMPT
        implement_mentions = "verified" in WORKER_IMPLEMENT_PROMPT.lower() or "验证" in WORKER_IMPLEMENT_PROMPT
        result.add(
            "explore_prompt_mentions_verified_info",
            explore_mentions,
            f"Explore prompt mentions verified info: {explore_mentions}",
        )
        result.add(
            "implement_prompt_mentions_verified_info",
            implement_mentions,
            f"Implement prompt mentions verified info: {implement_mentions}",
        )

    # --- Test 12: agent has no tool to call pool.py functions ---
    with RalphTestDir("exp1_") as td:
        # Check worker's allowed tools
        worker_source = Path(worker_mod.__file__).read_text()
        # Find the base_tools definition
        base_tools_match = re.search(r'base_tools\s*=\s*\[(.*?)\]', worker_source, re.DOTALL)
        if base_tools_match:
            tools_str = base_tools_match.group(1)
            has_custom_tool = "ralph_" in tools_str.lower() or "mcp" in tools_str.lower()
            result.add(
                "worker_has_no_custom_mcp_tools",
                not has_custom_tool,
                f"Worker tools: {tools_str.strip()}. No custom MCP tools available.",
                category="design_gap",
            )
        else:
            result.add(
                "worker_has_no_custom_mcp_tools",
                True,
                "Could not parse base_tools, but no MCP tools expected.",
                category="design_gap",
            )

    return result


# =============================================================================
# Integration Tests: Real Workers + WebSearch Duplication
# =============================================================================

async def run_integration_tests(runs: int = 3) -> ExperimentResult:
    """
    Run real workers on related EXPLORE tasks and measure WebSearch duplication.

    Design:
    - Create 3 EXPLORE tasks all about the same time-sensitive topic
    - Run workers sequentially on each task
    - Count WebSearch calls per topic in tool_calls.jsonl
    """
    from ralph_sdk.worker import work

    result = ExperimentResult(
        experiment_name="exp1_verified_info_integration",
        description=f"Test duplicate WebSearch across {runs} runs with 3 related tasks each",
    )

    TEMPORAL_TOPIC = "claude-agent-sdk Python library"

    for run_idx in range(runs):
        with RalphTestDir(f"exp1_int_r{run_idx}_") as td:
            td.setup_basic_project(
                goal=f"""Research the {TEMPORAL_TOPIC} for building AI agents.
Focus on: current version, API patterns, and best practices.

## Temporal Topics (需验证的时效性话题)
- [ ] {TEMPORAL_TOPIC} latest version and API
- [ ] Claude API pricing and limits
""",
                tasks=[
                    {
                        "id": "T001",
                        "type": "EXPLORE",
                        "title": f"Research {TEMPORAL_TOPIC} API basics",
                        "description": f"Find the latest version of {TEMPORAL_TOPIC} and its basic API. Use WebSearch to verify current information.",
                    },
                    {
                        "id": "T002",
                        "type": "EXPLORE",
                        "title": f"Research {TEMPORAL_TOPIC} advanced usage",
                        "description": f"Investigate advanced patterns in {TEMPORAL_TOPIC} - streaming, tool use, structured output. Must WebSearch to verify current API.",
                    },
                    {
                        "id": "T003",
                        "type": "EXPLORE",
                        "title": f"Research {TEMPORAL_TOPIC} error handling",
                        "description": f"How to handle errors and retries with {TEMPORAL_TOPIC}. WebSearch for current best practices and known issues.",
                    },
                ],
            )

            # Run 3 workers sequentially
            total_websearch = 0
            for task_id in ["T001", "T002", "T003"]:
                try:
                    await work(task_id, "EXPLORE", td.cwd, verbose=False)
                except Exception as e:
                    result.add(
                        f"run{run_idx}_worker_{task_id}",
                        False,
                        f"Worker failed: {e}",
                        category="integration",
                    )
                    continue

            # Count WebSearch calls
            ws_calls = td.get_tool_calls("WebSearch")
            total_websearch = len(ws_calls)

            # Extract search queries
            queries = [c.get("input", {}).get("query", "") for c in ws_calls]

            # Check for duplicate/overlapping queries
            # Simple heuristic: queries that share >50% of words
            def word_overlap(q1: str, q2: str) -> float:
                w1 = set(q1.lower().split())
                w2 = set(q2.lower().split())
                if not w1 or not w2:
                    return 0
                return len(w1 & w2) / min(len(w1), len(w2))

            duplicate_pairs = []
            for i in range(len(queries)):
                for j in range(i + 1, len(queries)):
                    overlap = word_overlap(queries[i], queries[j])
                    if overlap > 0.5:
                        duplicate_pairs.append((queries[i], queries[j], overlap))

            # Check if verified info was written to pool.md
            pool = td.read_pool()
            has_verified = "[Verified" in pool

            result.add(
                f"run{run_idx}_total_websearch_calls",
                True,  # informational
                f"Total WebSearch calls: {total_websearch}, queries: {queries[:5]}",
                category="integration",
            )
            result.add(
                f"run{run_idx}_duplicate_searches_found",
                len(duplicate_pairs) > 0,
                f"Found {len(duplicate_pairs)} duplicate pairs (>50% word overlap): "
                + "; ".join(f"'{p[0][:40]}' vs '{p[1][:40]}' ({p[2]:.0%})" for p in duplicate_pairs[:3]),
                category="integration",
            )
            result.add(
                f"run{run_idx}_verified_info_written",
                has_verified,
                f"Pool has [Verified] entries: {has_verified}. "
                "If False, workers don't write verified info → cache never populated.",
                category="integration",
            )

    return result


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("  Experiment 1: Verified Info Duplicate Search")
    print("=" * 60)

    # Phase 1: Unit tests (no API)
    print("\n--- Phase 1: Unit Tests ---\n")
    unit_result = run_unit_tests()
    print(unit_result.summary())
    unit_result.save()

    return unit_result


async def main_full(runs: int = 3):
    """Run both unit and integration tests."""
    print("\n" + "=" * 60)
    print("  Experiment 1: Verified Info Duplicate Search (Full)")
    print("=" * 60)

    # Phase 1: Unit tests
    print("\n--- Phase 1: Unit Tests ---\n")
    unit_result = run_unit_tests()
    print(unit_result.summary())
    unit_result.save()

    # Phase 2: Integration tests
    print("\n--- Phase 2: Integration Tests ---\n")
    int_result = await run_integration_tests(runs=runs)
    print(int_result.summary())
    int_result.save()

    return unit_result, int_result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run integration tests too (requires API)")
    parser.add_argument("--runs", type=int, default=3, help="Number of integration test runs")
    args = parser.parse_args()

    if args.full:
        asyncio.run(main_full(runs=args.runs))
    else:
        main()
