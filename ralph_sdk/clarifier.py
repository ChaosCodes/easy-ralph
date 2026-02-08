"""
Requirements clarification module.

Clarifies user requirements through two-phase interactive Q&A:
- Phase 1: Agent generates questions as JSON
- Phase 2: Our code presents questions via Rich prompts
- Phase 3: Answers fed back to agent for final output

AskUserQuestion doesn't work through the SDK (interactive form can't render
in piped stdin/stdout mode). Instead we use a two-phase approach.
"""

import json
import re
from claude_agent_sdk import ClaudeAgentOptions
from rich.console import Console
from rich.panel import Panel
from .interactive import ask_user_interactive, CYAN, BOLD, DIM, YELLOW, GREEN, GRAY, RESET

from .metrics import (
    AutomationLevel,
    EvalConfig,
    MetricDefinition,
    MetricsConfig,
    MetricType,
    TaskCategory,
    get_default_metrics,
)
from .logger import log_tool_call, stream_query
from .pool import init_ralph_dir, write_goal
from .prompts import CLARIFIER_SYSTEM_PROMPT, CLARIFIER_V2_SYSTEM_PROMPT, CLARIFIER_V2_EXPLORE_PROMPT

console = Console()


# =============================================================================
# Two-Phase Q&A: Agent generates questions, Rich presents them
# =============================================================================

def _ask_user_rich(questions_json: list[dict]) -> dict[str, str]:
    """Present questions to user via interactive terminal selector.

    Uses arrow-key navigation with highlight bar, direct typing for free text,
    and collapsed confirmation view. Falls back to simple input() on non-Unix.

    Args:
        questions_json: List of question dicts with format:
            [{"question": "...", "options": ["A", "B", "C"]}, ...]

    Returns:
        Dict mapping question text to user's answer.
    """
    return ask_user_interactive(questions_json)


def _format_answers_for_prompt(answers: dict[str, str]) -> str:
    """Format collected answers as text for feeding back to agent."""
    lines = []
    for q, a in answers.items():
        lines.append(f"Q: {q}")
        lines.append(f"A: {a}")
    return "\n".join(lines)


# =============================================================================
# Parsing Functions
# =============================================================================

def parse_metrics(text: str) -> list[dict]:
    """Parse AI-generated metrics from text."""
    metrics = []
    blocks = re.split(r'\n?METRIC:\s*', text)

    for block in blocks[1:]:
        metric = {}

        name_match = re.match(r'^(\S+)', block)
        if name_match:
            metric['name'] = name_match.group(1).strip()

        type_match = re.search(r'TYPE:\s*(\w+)', block, re.IGNORECASE)
        if type_match:
            metric['type'] = type_match.group(1).lower()

        target_match = re.search(r'TARGET:\s*(.+?)(?=\n|$)', block, re.IGNORECASE)
        if target_match:
            metric['target'] = target_match.group(1).strip()

        why_match = re.search(r'WHY:\s*(.+?)(?=\n|$)', block, re.IGNORECASE)
        if why_match:
            metric['why'] = why_match.group(1).strip()

        measure_match = re.search(r'MEASURE:\s*(.+?)(?=\n|$)', block, re.IGNORECASE)
        if measure_match:
            metric['how_to_measure'] = measure_match.group(1).strip()

        auto_match = re.search(r'AUTOMATION:\s*(\w+)', block, re.IGNORECASE)
        if auto_match:
            metric['automation'] = auto_match.group(1).lower()

        proxy_match = re.search(r'PROXY:\s*(.+?)(?=\n|$)', block, re.IGNORECASE)
        if proxy_match:
            metric['proxy_metric'] = proxy_match.group(1).strip()

        batch_match = re.search(r'BATCH:\s*(.+?)(?=\n|$)', block, re.IGNORECASE)
        if batch_match:
            metric['batch_suggestion'] = batch_match.group(1).strip()

        if metric.get('name'):
            metrics.append(metric)

    return metrics


def _extract_json(text: str) -> dict | None:
    """Extract a JSON object from text."""
    fence_match = re.search(r"```(?:json)?\s*\n(\{.*?\})\s*\n```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass

    return None


# =============================================================================
# Base Metrics (always included)
# =============================================================================

BASE_METRICS = [
    {
        "name": "runs_without_error",
        "type": "hard",
        "target": "pass",
        "why": "基础要求：代码能正常运行",
        "automation": "auto",
    },
    {
        "name": "basic_functionality",
        "type": "hard",
        "target": "pass",
        "why": "基础要求：核心功能正常",
        "automation": "auto",
    },
]


# =============================================================================
# Main Clarification Functions
# =============================================================================

async def clarify_metrics(goal_description: str, cwd: str = ".", verbose: bool = False) -> tuple[MetricsConfig, EvalConfig, dict]:
    """
    Clarify success metrics through two-phase interactive Q&A.

    Phase 1: Agent generates questions as JSON
    Phase 2: Our code presents questions via Rich prompts
    Phase 3: Agent generates metrics config based on answers

    Args:
        goal_description: The goal description
        cwd: Working directory
        verbose: Show detailed tool calls and thinking

    Returns:
        Tuple of (MetricsConfig, EvalConfig, all_answers)
    """
    console.print(f"\n[bold cyan]Success Metrics Configuration[/bold cyan]\n")

    # Phase 1: Agent generates questions
    questions_prompt = f"""帮用户配置项目的成功指标。

项目描述：
{goal_description}

请生成需要问用户的问题，输出 JSON 格式：
```json
{{
  "questions": [
    {{
      "question": "问题文本",
      "options": ["选项1", "选项2", "选项3"]
    }}
  ]
}}
```

必须包含的问题：
1. 项目用途：生产部署 / 研究实验 / 学习探索 / 原型验证
2. 评估模式：全自动(有benchmark) / 半自动(代理指标+人工确认) / 人工为主(真实环境测试)

根据项目特点，再加 1-2 个针对性的技术约束问题（如延迟、准确率、成本等）。

只输出 JSON，不要有其他文字。
"""

    sr = await stream_query(
        prompt=questions_prompt,
        options=ClaudeAgentOptions(
            system_prompt="你是一个帮助配置项目评估指标的助手。只输出 JSON 格式的问题列表。",
            max_turns=1,
            cwd=cwd,
        ),
        agent_name="clarifier",
        emoji="📝",
        cwd=cwd,
        verbose=verbose,
        status_message="Generating metrics questions...",
    )
    questions_text = sr.text

    # Parse questions and present to user
    questions_json = _extract_json(questions_text)
    questions_list = questions_json.get("questions", []) if questions_json else []

    if not questions_list:
        # Fallback: use hardcoded questions
        questions_list = [
            {"question": "项目用途是什么？", "options": ["生产部署", "研究实验", "学习探索", "原型验证"]},
            {"question": "评估模式偏好？", "options": ["全自动(有benchmark)", "半自动(代理指标+人工确认)", "人工为主(真实环境测试)"]},
        ]

    # Phase 2: Present questions via Rich prompts
    answers = _ask_user_rich(questions_list)
    answers_text = _format_answers_for_prompt(answers)

    if verbose:
        console.print(f"[dim]Collected {len(answers)} answers[/dim]")

    # Phase 3: Agent generates metrics based on answers
    metrics_prompt = f"""根据以下信息生成评估指标配置。

项目描述：
{goal_description}

用户回答：
{answers_text}

请输出 JSON 格式的评估指标配置：
```json
{{
  "purpose": "项目用途",
  "eval_mode": "评估模式",
  "test_frequency": "测试频率（如适用）",
  "batch_preference": "测试安排偏好（如适用）",
  "category": "algorithm|web|api|cli|library|general",
  "metrics": [
    {{
      "name": "metric_name_in_snake_case",
      "type": "hard|soft|subjective",
      "target": "目标值",
      "why": "为什么重要",
      "automation": "auto|manual|hybrid",
      "proxy_metric": "代理指标（如适用）",
      "batch_suggestion": "批量测试建议（如适用）"
    }}
  ]
}}
```

注意：
- metrics 数组应包含 2-4 个指标
- 根据用户的回答选择合适的 category、type、automation 等
- 只输出 JSON，不要有其他文字
"""

    sr = await stream_query(
        prompt=metrics_prompt,
        options=ClaudeAgentOptions(
            system_prompt="你是一个帮助配置项目评估指标的助手。根据用户的回答生成结构化的指标配置。只输出 JSON。",
            max_turns=1,
            cwd=cwd,
        ),
        agent_name="clarifier",
        emoji="📝",
        cwd=cwd,
        verbose=verbose,
        status_message="Generating metrics config...",
    )
    result_text = sr.text

    # Parse JSON output
    all_answers = {"goal": goal_description}
    json_obj = _extract_json(result_text)

    if json_obj:
        all_answers["purpose"] = json_obj.get("purpose", "")
        all_answers["eval_mode"] = json_obj.get("eval_mode", "")
        if json_obj.get("test_frequency"):
            all_answers["test_frequency"] = json_obj["test_frequency"]
        if json_obj.get("batch_preference"):
            all_answers["batch_preference"] = json_obj["batch_preference"]
        dynamic_metrics = json_obj.get("metrics", [])
    else:
        # Fallback: try to parse metrics from text format
        all_answers["purpose"] = "原型验证"
        all_answers["eval_mode"] = "全自动"
        dynamic_metrics = parse_metrics(result_text)

    # Build EvalConfig
    eval_mode = all_answers.get("eval_mode", "全自动")
    eval_config = EvalConfig(mode=eval_mode)
    if all_answers.get("test_frequency"):
        eval_config.test_frequency = all_answers["test_frequency"]
    if all_answers.get("batch_preference"):
        eval_config.batch_preference = all_answers["batch_preference"]

    # Combine base + dynamic metrics
    all_metrics = BASE_METRICS + dynamic_metrics

    # Display metrics as vertical cards
    print(f"\n  {BOLD}{GREEN}生成的评估指标{RESET}\n")

    for m in all_metrics:
        name = m.get('name', '')
        mtype = m.get('type', '')
        auto_label = {"auto": "自动", "manual": "人工", "hybrid": "混合"}.get(
            m.get("automation", "auto"), "自动"
        )
        auto_color = {"auto": GREEN, "manual": YELLOW, "hybrid": CYAN}.get(
            m.get("automation", "auto"), GREEN
        )
        target = m.get('target', '')
        why = m.get('why', '')

        tags = f"{DIM}{mtype} · {auto_color}{auto_label}{RESET}"
        print(f"  {CYAN}╭{RESET} {BOLD}{CYAN}{name}{RESET}  {tags}")
        if target:
            print(f"  {CYAN}│{RESET} 目标: {YELLOW}{target}{RESET}")
        if why:
            print(f"  {CYAN}│{RESET} {DIM}{why}{RESET}")
        print(f"  {CYAN}╰{RESET}\n")

    # Convert to MetricsConfig
    category_str = json_obj.get("category", "general") if json_obj else "general"
    try:
        category = TaskCategory(category_str)
    except ValueError:
        category = TaskCategory.GENERAL
    metrics_config = MetricsConfig(category=category, eval_config=eval_config)

    for m in all_metrics:
        metric_type = MetricType(m.get("type", "soft"))
        automation = AutomationLevel(m.get("automation", "auto"))

        metric_def = MetricDefinition(
            name=m["name"],
            type=metric_type,
            description=m.get("why", ""),
            target=m.get("target"),
            automation=automation,
            proxy_metric=m.get("proxy_metric"),
            batch_suggestion=m.get("batch_suggestion"),
        )

        if metric_type == MetricType.HARD:
            metrics_config.hard_constraints.append(metric_def)
        elif metric_type == MetricType.SOFT:
            metrics_config.soft_targets.append(metric_def)
        else:
            metrics_config.subjective_criteria.append(metric_def)

    console.print("\n[green]✓ Metrics configured[/green]")
    return metrics_config, eval_config, all_answers


def generate_goal_md(
    initial_prompt: str,
    summary_text: str,
    metrics_config: MetricsConfig,
    eval_config: EvalConfig,
    qa_text: str = "",
    answers_text: str = "",
) -> str:
    """Generate complete goal.md content."""
    lines = ["# Goal", ""]

    # Original request
    lines.append("## Original Request")
    lines.append(initial_prompt)
    lines.append("")

    # Q&A section (if any)
    if qa_text and answers_text:
        lines.append("## Clarification")
        lines.append("")
        lines.append("### Questions")
        lines.append(qa_text)
        lines.append("")
        lines.append("### Answers")
        lines.append(answers_text)
        lines.append("")

    # Clarified description
    lines.append("## Clarified Description")
    lines.append(summary_text)
    lines.append("")

    # Evaluation Mode section
    if eval_config.needs_user_testing():
        lines.append("## Evaluation Mode")
        lines.append("")
        lines.append(f"- **测试模式**: {eval_config.mode}")
        if eval_config.test_frequency:
            lines.append(f"- **测试频率**: {eval_config.test_frequency}")
        if eval_config.batch_preference:
            lines.append(f"- **测试安排**: {eval_config.batch_preference}")
        lines.append("")

    # Success Metrics
    lines.append("## Success Metrics")
    lines.append("")

    # Hard constraints
    if metrics_config.hard_constraints:
        lines.append("### Hard Constraints (must pass)")
        for m in metrics_config.hard_constraints:
            auto_tag = "[auto]" if m.automation == AutomationLevel.AUTO else "[manual]"
            lines.append(f"- [ ] **{m.name}** {auto_tag}: {m.target or 'pass'} - {m.description}")
        lines.append("")

    # Soft targets
    if metrics_config.soft_targets:
        lines.append("### Performance Targets")
        lines.append("| Metric | Target | Automation | Proxy |")
        lines.append("|--------|--------|------------|-------|")
        for m in metrics_config.soft_targets:
            proxy = m.proxy_metric or "-"
            lines.append(f"| {m.name} | {m.target or 'N/A'} | {m.automation.value} | {proxy} |")
        lines.append("")

    # Subjective criteria
    if metrics_config.subjective_criteria:
        lines.append("### Quality Criteria (AI-evaluated)")
        for m in metrics_config.subjective_criteria:
            lines.append(f"- **{m.name}**: {m.description}")
        lines.append("")

    # Manual testing instructions
    manual_metrics = [
        m for m in metrics_config.all_metrics()
        if m.automation in (AutomationLevel.MANUAL, AutomationLevel.HYBRID)
    ]
    if manual_metrics:
        lines.append("### Manual Testing Instructions")
        for m in manual_metrics:
            if m.batch_suggestion:
                lines.append(f"- **{m.name}**: {m.batch_suggestion}")
        lines.append("")

    return "\n".join(lines)


async def clarify_requirements(initial_prompt: str, cwd: str = ".", verbose: bool = False) -> str:
    """
    Clarify requirements through two-phase interactive Q&A.

    Phase 1: Agent explores codebase and generates questions as JSON
    Phase 2: Our code presents questions via Rich prompts
    Phase 3: Agent generates clarified requirements using answers

    Args:
        initial_prompt: The user's initial feature request
        cwd: Working directory
        verbose: Show detailed tool calls and thinking

    Returns:
        The clarified goal content (also written to goal.md)
    """
    init_ralph_dir(cwd)

    console.print(Panel(f"[bold]Feature Request:[/bold]\n{initial_prompt}", title="Input"))
    console.print("\n[yellow]Analyzing requirements...[/yellow]\n")

    # Phase 1: Agent explores codebase and generates questions
    explore_prompt = f"""User's feature request:
{initial_prompt}

请按以下流程操作：
1. 探索代码库，了解项目结构、技术栈、现有模式
2. 根据探索结果，生成需要问用户的澄清问题

最后输出 JSON 格式的问题列表：
```json
{{
  "codebase_context": "对代码库的简要理解",
  "questions": [
    {{
      "question": "问题文本",
      "options": ["选项1", "选项2", "选项3"]
    }}
  ]
}}
```

问题应该关注：需求范围、目标用户、核心功能、技术约束。
生成 2-4 个问题，每个问题 2-4 个选项。
确保 JSON 是输出的最后一部分。
"""

    sr = await stream_query(
        prompt=explore_prompt,
        options=ClaudeAgentOptions(
            system_prompt=CLARIFIER_SYSTEM_PROMPT,
            allowed_tools=[
                "Read", "Glob", "Grep", "LSP",
                "WebFetch", "WebSearch",
            ],
            max_turns=15,
            cwd=cwd,
        ),
        agent_name="clarifier",
        emoji="🔍",
        cwd=cwd,
        verbose=verbose,
        show_tools=True,
    )
    explore_text = sr.text

    # Parse questions and present to user
    questions_json = _extract_json(explore_text)
    codebase_context = questions_json.get("codebase_context", "") if questions_json else ""
    questions_list = questions_json.get("questions", []) if questions_json else []

    if not questions_list:
        # Fallback: use generic questions
        questions_list = [
            {"question": "这个功能的目标用户是谁？", "options": ["开发者", "终端用户", "运维人员", "所有人"]},
            {"question": "核心需求是什么？", "options": ["新功能", "性能优化", "Bug修复", "重构"]},
        ]

    # Phase 2: Present questions via Rich prompts
    console.print("\n[bold cyan]Clarification Questions[/bold cyan]")
    answers = _ask_user_rich(questions_list)
    answers_text = _format_answers_for_prompt(answers)

    # Phase 3: Agent generates clarified requirements using answers
    clarify_prompt = f"""User's feature request:
{initial_prompt}

代码库上下文：
{codebase_context}

用户对澄清问题的回答：
{answers_text}

请根据以上信息，生成 clarified requirements（markdown 格式）。

输出要求：
- Clear, detailed description of what needs to be built
- Scope (what's included)
- Non-goals (what's explicitly NOT included)
- Important context from codebase exploration
- Temporal Topics (需验证的时效性话题)
"""

    sr = await stream_query(
        prompt=clarify_prompt,
        options=ClaudeAgentOptions(
            system_prompt=CLARIFIER_SYSTEM_PROMPT,
            max_turns=3,
            cwd=cwd,
        ),
        agent_name="clarifier",
        emoji="📝",
        cwd=cwd,
        verbose=verbose,
        status_message="Generating clarified requirements...",
    )
    summary_text = sr.text

    console.print(Panel(summary_text, title="Clarified Requirements"))

    # Phase 2: Clarify success metrics
    metrics_config, eval_config, _ = await clarify_metrics(summary_text, cwd, verbose=verbose)

    # Build goal.md content
    goal_content = generate_goal_md(
        initial_prompt=initial_prompt,
        summary_text=summary_text,
        metrics_config=metrics_config,
        eval_config=eval_config,
    )

    # Write to goal.md
    write_goal(goal_content, cwd)
    console.print("\n[green]✓ Goal saved to .ralph/goal.md[/green]")

    return goal_content


async def quick_clarify(initial_prompt: str, cwd: str = ".") -> str:
    """
    Quick clarification without interactive Q&A.
    Useful for simple, well-defined requests.

    Args:
        initial_prompt: The user's initial feature request
        cwd: Working directory

    Returns:
        The goal content (also written to goal.md)
    """
    init_ralph_dir(cwd)

    # Use general default metrics (category detection moved to agent prompt)
    category = TaskCategory.GENERAL
    metrics_config = get_default_metrics(category)
    eval_config = EvalConfig(mode="全自动")

    goal_content = generate_goal_md(
        initial_prompt=initial_prompt,
        summary_text=initial_prompt,
        metrics_config=metrics_config,
        eval_config=eval_config,
    )

    write_goal(goal_content, cwd)
    console.print(f"\n[green]✓ Goal saved to .ralph/goal.md[/green]")
    console.print(f"[dim]Task category: {category.value}, using default metrics[/dim]")

    return goal_content


# =============================================================================
# Clarifier v2: Explore and Propose Mode
# =============================================================================

PROPOSAL_PARSE_PROMPT = """
从以下探索结果中提取结构化信息。

探索结果：
---
{exploration_result}
---

请提取并输出以下 JSON 格式：
```json
{
  "understanding": "对用户需求的一句话理解",
  "proposals": [
    {
      "name": "方案名称",
      "summary": "一句话概述",
      "pros": ["优点1", "优点2"],
      "cons": ["缺点1", "缺点2"],
      "complexity": "低|中|高",
      "risk": "主要风险"
    }
  ],
  "recommendation": {
    "name": "推荐的方案名称",
    "reasons": ["原因1", "原因2", "原因3"]
  },
  "temporal_topics": ["需要验证的时效性话题1", "话题2"]
}
```

只输出 JSON，不要有其他文字。
"""


async def explore_and_propose(initial_prompt: str, cwd: str = ".", verbose: bool = False) -> str:
    """
    Clarifier v2: Explore possible approaches and propose options to user.

    Phase 1: Agent explores codebase and generates proposals as JSON
    Phase 2: Our code presents proposals via Rich prompts
    Phase 3: Agent generates clarified goal based on user's choice

    Args:
        initial_prompt: The user's initial (possibly vague) request
        cwd: Working directory
        verbose: Show detailed tool calls and thinking

    Returns:
        The clarified goal content (also written to goal.md)
    """
    init_ralph_dir(cwd)

    console.print(Panel(
        f"[bold]用户需求:[/bold]\n{initial_prompt}",
        title="[cyan]Clarifier v2: 探索+提议模式[/cyan]",
        border_style="cyan",
    ))

    console.print("\n[yellow]深度探索中...[/yellow]")
    console.print("[dim]Agent 正在研究可能的实现方案...[/dim]\n")

    # Phase 1: Agent explores and generates proposals as JSON
    explore_prompt = CLARIFIER_V2_EXPLORE_PROMPT.format(user_request=initial_prompt) + """

完成探索后，输出 JSON 格式的方案提议：
```json
{
  "understanding": "对用户需求的一句话理解",
  "proposals": [
    {
      "name": "方案名称",
      "summary": "一句话概述",
      "pros": ["优点1", "优点2"],
      "cons": ["缺点1", "缺点2"]
    }
  ],
  "follow_up_questions": [
    {
      "question": "需要进一步了解的问题",
      "options": ["选项1", "选项2", "选项3"]
    }
  ]
}
```

确保 JSON 是输出的最后一部分。
"""

    sr = await stream_query(
        prompt=explore_prompt,
        options=ClaudeAgentOptions(
            system_prompt=CLARIFIER_V2_SYSTEM_PROMPT,
            allowed_tools=[
                "Read", "Glob", "Grep", "LSP",
                "WebFetch", "WebSearch", "Task",
            ],
            max_turns=25,
            cwd=cwd,
        ),
        agent_name="clarifier_v2",
        emoji="🔍",
        cwd=cwd,
        verbose=verbose,
        show_tools=True,
    )
    explore_text = sr.text

    # Parse proposals and present to user
    proposals_json = _extract_json(explore_text)

    if proposals_json and proposals_json.get("proposals"):
        understanding = proposals_json.get("understanding", "")
        proposals = proposals_json["proposals"]

        if understanding:
            console.print(f"\n[bold]理解:[/bold] {understanding}\n")

        # Present proposals as a question
        proposal_question = {
            "question": "请选择一个实现方案：",
            "options": [
                {"label": f"{p['name']}: {p['summary']}"} for p in proposals
            ],
        }
        answers = _ask_user_rich([proposal_question])

        # Also ask follow-up questions if any
        follow_ups = proposals_json.get("follow_up_questions", [])
        if follow_ups:
            console.print("\n[bold cyan]Follow-up Questions[/bold cyan]")
            follow_up_answers = _ask_user_rich(follow_ups)
            answers.update(follow_up_answers)
    else:
        # Fallback: generic question
        answers = _ask_user_rich([
            {"question": "你对这个需求有什么具体的偏好？", "options": ["简单实现", "完整方案", "最佳实践"]},
        ])

    answers_text = _format_answers_for_prompt(answers)

    # Phase 3: Agent generates clarified goal based on user's choice
    goal_prompt = f"""用户需求：
{initial_prompt}

探索结果和方案：
{explore_text[:3000]}

用户的选择和回答：
{answers_text}

请根据以上信息，生成明确的目标描述（markdown 格式），包含：
- Clarified Description
- Scope
- Non-goals
- Technical Approach
- Risks and Mitigations
"""

    sr = await stream_query(
        prompt=goal_prompt,
        options=ClaudeAgentOptions(
            system_prompt=CLARIFIER_V2_SYSTEM_PROMPT,
            max_turns=3,
            cwd=cwd,
        ),
        agent_name="clarifier_v2",
        emoji="📝",
        cwd=cwd,
        verbose=verbose,
        status_message="Generating goal summary...",
    )
    summary_text = sr.text

    console.print(Panel(summary_text, title="明确后的目标", border_style="green"))

    # Configure metrics
    metrics_config, eval_config, _ = await clarify_metrics(summary_text, cwd, verbose=verbose)

    # Build goal.md
    goal_content = generate_goal_md(
        initial_prompt=initial_prompt,
        summary_text=summary_text,
        metrics_config=metrics_config,
        eval_config=eval_config,
    )

    # Write to goal.md
    write_goal(goal_content, cwd)
    console.print("\n[green]✓ Goal saved to .ralph/goal.md[/green]")

    return goal_content


async def clarify_requirements_v2(
    initial_prompt: str,
    cwd: str = ".",
    mode: str = "auto",
    verbose: bool = False,
) -> str:
    """
    Unified clarification entry point that chooses the best mode.

    Args:
        initial_prompt: The user's initial request
        cwd: Working directory
        mode: "auto" | "ask" | "explore"
            - auto: Automatically choose based on request clarity
            - ask: Use traditional Q&A mode
            - explore: Use explore+propose mode
        verbose: Show detailed tool calls and thinking

    Returns:
        The clarified goal content
    """
    if mode == "ask":
        return await clarify_requirements(initial_prompt, cwd, verbose=verbose)
    elif mode == "explore":
        return await explore_and_propose(initial_prompt, cwd, verbose=verbose)
    else:
        # Auto mode: detect based on keywords
        vague_indicators = [
            "研究", "探索", "看看", "想想",
            "可能", "也许", "不确定",
            "怎么做", "什么方法", "有什么",
            "research", "explore", "investigate",
            "could", "might", "maybe",
            "how to", "what if", "possibilities",
        ]

        is_vague = any(indicator in initial_prompt.lower() for indicator in vague_indicators)

        if is_vague:
            console.print("[dim]检测到模糊需求，使用探索+提议模式[/dim]")
            return await explore_and_propose(initial_prompt, cwd, verbose=verbose)
        else:
            console.print("[dim]需求相对明确，使用传统 Q&A 模式[/dim]")
            return await clarify_requirements(initial_prompt, cwd, verbose=verbose)
