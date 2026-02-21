"""
Synthesizer agent: extracts insights from accumulated experimental data.

Uses a Knowledge Base (synthesis_kb.md) that is snapshot-replaced each round,
replacing the old append-to-findings model. Pool.md Findings are rendered from KB.

Inspired by:
- Schulman's periodic "关键洞察" extraction
- Karpathy's "change one variable at a time"
- Sasha Rush's synthetic experiments for hypothesis isolation
"""

import re
from dataclasses import dataclass, field

from claude_agent_sdk import ClaudeAgentOptions
from rich.console import Console

from .logger import stream_query
from .pool import (
    append_hard_constraint,
    get_completed_task_summaries,
    read_goal,
    read_pool,
    read_synthesis_kb,
    update_pool_findings_from_kb,
    write_synthesis_kb,
)
from .prompts import SYNTHESIZER_SYSTEM_PROMPT, build_synthesizer_prompt

console = Console()


@dataclass
class Insight:
    """A single synthesized insight."""
    observation: str
    implication: str = ""
    confidence: str = "medium"


@dataclass
class ProposedExperiment:
    """A proposed experiment from synthesis."""
    name: str
    insight_tested: str = ""
    method: str = ""
    is_diagnostic: bool = False
    expected_outcome: str = ""
    differs_from_tried: str = ""


@dataclass
class SynthesisResult:
    """Result of running the synthesizer."""
    insights: list[Insight] = field(default_factory=list)
    proposed_experiments: list[ProposedExperiment] = field(default_factory=list)
    kb_content: str = ""
    result_stats: object | None = None


def extract_kb_markdown(text: str) -> str:
    """Extract KB markdown from LLM output.

    Handles both raw markdown and ```markdown``` wrapped output.
    """
    # Try to extract from markdown code fence
    fence_match = re.search(
        r"```(?:markdown)?\s*\n(# Synthesis KB.*?)```",
        text,
        re.DOTALL,
    )
    if fence_match:
        return fence_match.group(1).strip()

    # Look for raw markdown starting with "# Synthesis KB"
    kb_start = text.find("# Synthesis KB")
    if kb_start != -1:
        return text[kb_start:].strip()

    # Fallback: return full text (best effort)
    return text.strip()


def parse_kb_to_result(kb_content: str) -> SynthesisResult:
    """Parse KB markdown into SynthesisResult for orchestrator consumption."""
    insights = []
    proposed_experiments = []

    # Extract insights from ## Active Insights
    insights_match = re.search(
        r"## Active Insights\s*\n(.*?)(?=\n## |\Z)",
        kb_content,
        re.DOTALL,
    )
    if insights_match:
        body = insights_match.group(1)
        # Parse ### IXXX blocks
        blocks = re.split(r"(?=### I\d+)", body)
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            # Extract header: ### I001 [P1, high, 9x confirmed]
            header_match = re.match(
                r"### (I\d+)\s*\[([^\]]*)\]",
                block,
            )
            if not header_match:
                continue

            insight_id = header_match.group(1)
            meta = header_match.group(2)

            # Parse confidence from meta
            confidence = "medium"
            if "high" in meta:
                confidence = "high"
            elif "low" in meta:
                confidence = "low"
            elif "speculative" in meta:
                confidence = "speculative"

            # Extract observation (lines after header, before →)
            lines = block.split("\n")
            obs_lines = []
            impl_lines = []
            for line in lines[1:]:
                stripped = line.strip()
                if stripped.startswith("→"):
                    impl_lines.append(stripped[1:].strip())
                elif stripped.startswith("First:") or stripped.startswith("Evidence:"):
                    continue
                elif stripped:
                    obs_lines.append(stripped)

            insights.append(Insight(
                observation=" ".join(obs_lines) if obs_lines else insight_id,
                implication=" ".join(impl_lines) if impl_lines else "",
                confidence=confidence,
            ))

    # Extract proposed experiments from ### Proposed table
    proposed_match = re.search(
        r"### Proposed\s*\n(.*?)(?=\n### |\n## |\Z)",
        kb_content,
        re.DOTALL,
    )
    if proposed_match:
        body = proposed_match.group(1)
        # Parse table rows
        rows = [r for r in body.split("\n") if r.strip().startswith("|")]
        for row in rows:
            cols = [c.strip() for c in row.split("|")]
            cols = [c for c in cols if c]  # Remove empty from leading/trailing |
            if len(cols) >= 3 and re.match(r"E\d+", cols[0]):
                name = cols[2] if len(cols) > 2 else cols[0]
                tests = cols[3] if len(cols) > 3 else ""
                is_p1 = len(cols) > 1 and "P1" in cols[1]
                proposed_experiments.append(ProposedExperiment(
                    name=name,
                    insight_tested=tests,
                    is_diagnostic=is_p1,  # Treat P1 as highest priority → auto-create
                ))

    return SynthesisResult(
        insights=insights,
        proposed_experiments=proposed_experiments,
        kb_content=kb_content,
    )


def auto_promote_constraints(kb_content: str, cwd: str) -> None:
    """Auto-promote high-confidence insights with strong negative signals to hard constraints."""
    _HARD_CONSTRAINT_KEYWORDS = [
        "never", "always degrades", "always worse", "always fails",
        "0/", "9/9", "10/10", "100%", "0%",
        "consistently worse", "no improvement", "all failed",
    ]

    insights_match = re.search(
        r"## Active Insights\s*\n(.*?)(?=\n## |\Z)",
        kb_content,
        re.DOTALL,
    )
    if not insights_match:
        return

    blocks = re.split(r"(?=### I\d+)", insights_match.group(1))
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        # Only promote high confidence
        if "high" not in block.lower():
            continue
        block_lower = block.lower()
        if any(kw in block_lower for kw in _HARD_CONSTRAINT_KEYWORDS):
            # Extract first two lines as the constraint text
            lines = [l.strip() for l in block.split("\n") if l.strip()]
            constraint_text = " ".join(lines[:2])
            # Extract implication
            impl = ""
            for line in lines:
                if line.startswith("→"):
                    impl = line
                    break
            append_hard_constraint(
                f"[Auto-promoted from synthesis] {constraint_text} {impl}",
                cwd,
            )


async def synthesize(
    cwd: str = ".",
    verbose: bool = False,
    thinking_budget: int | None = None,
) -> SynthesisResult:
    """
    Run the synthesizer to extract insights from accumulated experimental data.

    Uses KB snapshot model: reads current KB, sends to LLM with new data,
    writes updated KB, renders findings to pool.md.
    """
    goal = read_goal(cwd)
    pool = read_pool(cwd)
    task_summaries = get_completed_task_summaries(cwd)

    if not task_summaries:
        return SynthesisResult()

    kb_content = read_synthesis_kb(cwd)

    prompt = build_synthesizer_prompt(goal, pool, task_summaries, kb_content)

    options = ClaudeAgentOptions(
        system_prompt=SYNTHESIZER_SYSTEM_PROMPT,
        allowed_tools=["Read", "Glob", "Grep"],
        permission_mode="acceptEdits",
        max_turns=5,
        cwd=cwd,
        # No output_format — LLM outputs raw markdown
    )

    # Synthesizer benefits from extended thinking for deep analysis
    effective = thinking_budget if thinking_budget is not None else 10_000
    if effective > 0:
        options.max_thinking_tokens = effective

    sr = await stream_query(
        prompt=prompt,
        options=options,
        agent_name="synthesizer",
        emoji="🔬",
        cwd=cwd,
        verbose=verbose,
    )

    # Extract markdown output
    new_kb = extract_kb_markdown(sr.text)

    # Validate: must contain expected structure
    if "# Synthesis KB" not in new_kb:
        console.print("[yellow]⚠ Synthesizer output doesn't look like valid KB, keeping previous[/yellow]")
        return SynthesisResult(result_stats=sr.result_stats)

    # Write KB (snapshot replace)
    write_synthesis_kb(new_kb, cwd)

    # Render Findings section in pool.md from KB
    update_pool_findings_from_kb(cwd)

    # Auto-promote high-confidence negative insights to hard constraints
    auto_promote_constraints(new_kb, cwd)

    # Parse KB into result for orchestrator
    result = parse_kb_to_result(new_kb)
    result.result_stats = sr.result_stats

    return result
