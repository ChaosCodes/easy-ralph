"""
Requirements clarification module.

Clarifies user requirements through interactive Q&A and outputs to goal.md.
Includes dynamic metrics clarification with AI-generated questions.
"""

import re
from typing import Optional

import questionary
from questionary import Style
from claude_code_sdk import AssistantMessage, ClaudeCodeOptions, query
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .metrics import (
    AutomationLevel,
    EvalConfig,
    MetricDefinition,
    MetricsConfig,
    MetricType,
    TaskCategory,
    detect_category,
    get_default_metrics,
)
from .pool import init_ralph_dir, write_goal
from .prompts import CLARIFIER_SYSTEM_PROMPT, CLARIFIER_V2_SYSTEM_PROMPT, CLARIFIER_V2_EXPLORE_PROMPT

console = Console()

# Custom style for questionary
custom_style = Style([
    ('qmark', 'fg:cyan bold'),
    ('question', 'fg:white bold'),
    ('answer', 'fg:cyan'),
    ('pointer', 'fg:cyan bold'),
    ('highlighted', 'fg:cyan bold'),
    ('selected', 'fg:green'),
])


# =============================================================================
# Base Questions (always asked)
# =============================================================================

Q_PURPOSE = {
    "question": "这个项目的主要用途是什么？",
    "options": [
        "生产部署 (给真实用户用)",
        "研究实验 (发论文、验证想法)",
        "学习探索 (学习新技术)",
        "原型验证 (快速验证可行性)",
    ],
}

Q_EVAL_MODE = {
    "question": "评估/测试主要由谁来做？",
    "options": [
        "全自动 (有现成 benchmark/测试集)",
        "半自动 (有代理指标，但最终需人工确认)",
        "人工为主 (需要在真实环境测试)",
    ],
}

Q_TEST_FREQUENCY = {
    "question": "你大概多久能测试一次？",
    "options": [
        "实时 (我会一直盯着，随时可以测)",
        "每小时 (我会定期来看)",
        "每天 (晚上/第二天来看结果)",
        "更久 (需要安排专门时间测试)",
    ],
}

Q_BATCH_PREFERENCE = {
    "question": "希望怎么安排测试？",
    "options": [
        "一个一个测 (Agent 出一个方案，我测完再继续)",
        "批量测 (Agent 先出多个方案，我一起测)",
        "自动筛选 (Agent 用代理指标筛选，只让我测最有希望的)",
    ],
}


# =============================================================================
# AI Prompts for Dynamic Generation
# =============================================================================

DYNAMIC_QUESTION_PROMPT = """
用户正在描述他们想要构建的项目。请根据用户的描述，生成 2-3 个针对性的选择题，帮助澄清项目的关键约束和评估指标。

## 用户描述
{goal}

## 用途
{purpose}

## 要求
1. 每个问题必须是选择题，有 3-4 个选项
2. 问题要针对这个具体场景，不要太通用
3. 关注对评估指标有影响的因素（延迟、准确性、成本等）
4. 选项要具体，最好有数字范围
5. 不要问"你担心什么问题"这种泛泛的问题，要问具体的技术约束

## 输出格式（严格按此格式）

QUESTION: <问题文字>
A: <选项A>
B: <选项B>
C: <选项C>
D: <选项D（可选）>

QUESTION: <下一个问题>
...

只输出问题，不要有其他解释。
"""

METRIC_GENERATION_PROMPT = """
根据用户的项目描述和回答，生成具体的评估指标。

## 用户描述
{goal}

## 用途
{purpose}

## 用户回答
{answers}

## 评估模式
{eval_mode}

## 要求
生成 2-4 个具体的评估指标，每个指标包括：
- 名称（英文，snake_case，如 response_latency）
- 类型（hard = 必须达到 / soft = 目标值 / subjective = 主观评估）
- 目标值（具体数字，如 <= 50ms, >= 90%）
- 为什么重要（一句话，针对这个具体项目）
- 如何测量（具体方法，要可执行）
- 自动化程度（auto = 可自动测试 / manual = 需要人工测试 / hybrid = 可用代理指标自动测，最终需人工确认）

如果是 manual 或 hybrid 类型的指标，还需要提供：
- 代理指标（proxy_metric）：一个可以自动测试的近似指标
- 批量测试建议（batch_suggestion）：如何让用户高效批量测试

## 输出格式（严格按此格式）

METRIC: <英文名称>
TYPE: <hard|soft|subjective>
TARGET: <目标值>
WHY: <为什么重要>
MEASURE: <如何测量>
AUTOMATION: <auto|manual|hybrid>
PROXY: <代理指标，如果 AUTOMATION 不是 auto>
BATCH: <批量测试建议，如果 AUTOMATION 不是 auto>

METRIC: <下一个指标>
...

只输出指标，不要有其他解释。
"""


# =============================================================================
# Parsing Functions
# =============================================================================

def parse_dynamic_questions(text: str) -> list[dict]:
    """Parse AI-generated questions from text."""
    questions = []
    blocks = re.split(r'\n?QUESTION:\s*', text)

    for block in blocks[1:]:
        lines = block.strip().split('\n')
        if not lines:
            continue

        question = {"question": lines[0].strip(), "options": []}

        for line in lines[1:]:
            line = line.strip()
            match = re.match(r'^([A-D])[\.:]\s*(.+)$', line)
            if match:
                value = match.group(2).strip()
                if value:
                    question["options"].append(value)

        if question["question"] and len(question["options"]) >= 2:
            questions.append(question)

    return questions


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


# =============================================================================
# Interactive Q&A with questionary
# =============================================================================

async def ask_select(question: str, options: list[str], allow_custom: bool = True) -> str:
    """Ask a selection question with optional custom input."""
    if allow_custom:
        choices = options + ["[自己输入]"]
    else:
        choices = options

    answer = await questionary.select(
        question,
        choices=choices,
        style=custom_style,
        use_shortcuts=False,
        use_indicator=True,
    ).ask_async()

    if answer == "[自己输入]":
        answer = await questionary.text(
            "请输入你的回答:",
            style=custom_style,
        ).ask_async()

    return answer or ""


# =============================================================================
# AI Generation Functions
# =============================================================================

async def generate_dynamic_questions(goal: str, purpose: str) -> list[dict]:
    """Use AI to generate context-specific questions."""
    console.print("\n[dim]分析需求，生成针对性问题...[/dim]")

    prompt = DYNAMIC_QUESTION_PROMPT.format(goal=goal, purpose=purpose)

    result_text = ""
    async for message in query(
        prompt=prompt,
        options=ClaudeCodeOptions(
            system_prompt="你是一个帮助澄清项目需求的助手。只输出要求的格式，不要有多余解释。",
            allowed_tools=[],
            max_turns=1,
        ),
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    result_text += block.text

    return parse_dynamic_questions(result_text)


async def generate_metrics(
    goal: str,
    purpose: str,
    answers: dict,
    eval_mode: str,
) -> list[dict]:
    """Use AI to generate metrics based on answers."""
    console.print("\n[dim]根据回答生成评估指标...[/dim]")

    answers_text = "\n".join([f"- {k}: {v}" for k, v in answers.items()])

    prompt = METRIC_GENERATION_PROMPT.format(
        goal=goal,
        purpose=purpose,
        answers=answers_text,
        eval_mode=eval_mode,
    )

    result_text = ""
    async for message in query(
        prompt=prompt,
        options=ClaudeCodeOptions(
            system_prompt="你是一个帮助定义评估指标的助手。只输出要求的格式，不要有多余解释。",
            allowed_tools=[],
            max_turns=1,
        ),
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    result_text += block.text

    return parse_metrics(result_text)


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

async def clarify_metrics(goal_description: str) -> tuple[MetricsConfig, EvalConfig, dict]:
    """
    Clarify success metrics through dynamic Q&A.

    Args:
        goal_description: The goal description

    Returns:
        Tuple of (MetricsConfig, EvalConfig, all_answers)
    """
    all_answers = {"goal": goal_description}

    console.print(f"\n[bold cyan]Success Metrics Configuration[/bold cyan]\n")

    # Q1: Purpose
    purpose = await ask_select(Q_PURPOSE["question"], Q_PURPOSE["options"], allow_custom=False)
    all_answers["purpose"] = purpose

    # Q2: Evaluation mode
    console.print()
    eval_mode = await ask_select(Q_EVAL_MODE["question"], Q_EVAL_MODE["options"], allow_custom=False)
    all_answers["eval_mode"] = eval_mode

    # Build EvalConfig
    eval_config = EvalConfig(mode=eval_mode)

    # If manual/hybrid, ask follow-up questions
    if "人工" in eval_mode or "半自动" in eval_mode:
        console.print()
        test_freq = await ask_select(Q_TEST_FREQUENCY["question"], Q_TEST_FREQUENCY["options"], allow_custom=True)
        all_answers["test_frequency"] = test_freq
        eval_config.test_frequency = test_freq

        console.print()
        batch_pref = await ask_select(Q_BATCH_PREFERENCE["question"], Q_BATCH_PREFERENCE["options"], allow_custom=True)
        all_answers["batch_preference"] = batch_pref
        eval_config.batch_preference = batch_pref

    # Generate dynamic questions
    dynamic_questions = await generate_dynamic_questions(goal_description, purpose)

    if dynamic_questions:
        for q in dynamic_questions:
            console.print()
            answer = await ask_select(q["question"], q["options"], allow_custom=True)
            all_answers[q["question"]] = answer

    # Generate metrics
    dynamic_metrics = await generate_metrics(goal_description, purpose, all_answers, eval_mode)

    # Combine base + dynamic metrics
    all_metrics = BASE_METRICS + dynamic_metrics

    # Display metrics
    console.print("\n[bold green]生成的评估指标[/bold green]\n")

    table = Table(show_header=True)
    table.add_column("指标", style="cyan", width=25)
    table.add_column("类型", style="dim", width=10)
    table.add_column("目标", style="yellow", width=15)
    table.add_column("自动化", width=10)
    table.add_column("为什么重要", width=35)

    for m in all_metrics:
        auto_display = {
            "auto": "[green]自动[/green]",
            "manual": "[yellow]人工[/yellow]",
            "hybrid": "[cyan]混合[/cyan]",
        }.get(m.get("automation", "auto"), "自动")

        table.add_row(
            m.get('name', ''),
            m.get('type', ''),
            m.get('target', ''),
            auto_display,
            m.get('why', '')[:35],
        )

    console.print(table)

    # Confirm
    console.print()
    confirm = await ask_select(
        "这些指标可以吗？",
        ["可以，就这样", "需要调整"],
        allow_custom=True,
    )

    if confirm != "可以，就这样":
        console.print(f"[dim]记录调整意见: {confirm}[/dim]")
        all_answers["adjustment"] = confirm

    # Convert to MetricsConfig
    category = detect_category(goal_description)
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


async def clarify_requirements(initial_prompt: str, cwd: str = ".") -> str:
    """
    Clarify requirements through interactive Q&A with the user.

    Args:
        initial_prompt: The user's initial feature request
        cwd: Working directory

    Returns:
        The clarified goal content (also written to goal.md)
    """
    init_ralph_dir(cwd)

    console.print(Panel(f"[bold]Feature Request:[/bold]\n{initial_prompt}", title="Input"))

    # Phase 1: Generate clarifying questions about functionality
    console.print("\n[yellow]Analyzing requirements...[/yellow]\n")

    questions_prompt = f"""User's feature request:
{initial_prompt}

First, explore the codebase to understand:
1. Project structure and tech stack
2. Existing patterns and conventions
3. Related existing functionality

Then generate 3-5 clarifying questions with lettered options to better understand the requirements.
Focus on scope, target users, core functionality.
"""

    questions_text = ""
    async for message in query(
        prompt=questions_prompt,
        options=ClaudeCodeOptions(
            system_prompt=CLARIFIER_SYSTEM_PROMPT,
            allowed_tools=[
                "Read", "Glob", "Grep", "LSP",
                "WebFetch", "WebSearch",
            ],
            max_turns=8,
            cwd=cwd,
        ),
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    questions_text += block.text

    # Display questions and get answers
    console.print(Panel(questions_text, title="Clarifying Questions"))

    answers = await questionary.text(
        "Your answers (e.g., 1A, 2B, 3C or detailed response):",
        style=custom_style,
    ).ask_async() or ""

    # Phase 2: Generate summary
    console.print("\n[yellow]Generating clarified requirements...[/yellow]\n")

    summary_prompt = f"""Original request:
{initial_prompt}

Questions asked:
{questions_text}

User's answers:
{answers}

Based on this, provide a clear goal document with:
1. A clear, detailed description of what needs to be built
2. The scope (what's included)
3. Non-goals (what's explicitly NOT included)
4. Any important context from the codebase exploration

Format as markdown.
"""

    summary_text = ""
    async for message in query(
        prompt=summary_prompt,
        options=ClaudeCodeOptions(
            system_prompt=CLARIFIER_SYSTEM_PROMPT,
            allowed_tools=["Read", "Glob", "Grep", "LSP", "WebFetch", "WebSearch"],
            max_turns=5,
            cwd=cwd,
        ),
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    summary_text += block.text

    console.print(Panel(summary_text, title="Clarified Requirements"))

    # Confirm
    confirm = await ask_select(
        "Proceed with these requirements?",
        ["Yes", "No", "Edit"],
        allow_custom=False,
    )

    if confirm == "No":
        raise KeyboardInterrupt("User cancelled")
    elif confirm == "Edit":
        edited = await questionary.text("Enter your revised requirements:", style=custom_style).ask_async()
        if edited:
            summary_text = edited

    # Phase 3: Clarify success metrics (dynamic)
    metrics_config, eval_config, _ = await clarify_metrics(summary_text)

    # Build goal.md content
    goal_content = generate_goal_md(
        initial_prompt=initial_prompt,
        summary_text=summary_text,
        metrics_config=metrics_config,
        eval_config=eval_config,
        qa_text=questions_text,
        answers_text=answers,
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

    # Use default metrics based on detected category
    category = detect_category(initial_prompt)
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


async def explore_and_propose(initial_prompt: str, cwd: str = ".") -> str:
    """
    Clarifier v2: Explore possible approaches and propose options to user.

    This is the "explore + propose" mode that:
    1. Researches this direction to discover possibilities
    2. Deeply analyzes each approach's pros/cons
    3. Makes recommendations for user to choose from

    Args:
        initial_prompt: The user's initial (possibly vague) request
        cwd: Working directory

    Returns:
        The clarified goal content (also written to goal.md)
    """
    init_ralph_dir(cwd)

    console.print(Panel(
        f"[bold]用户需求:[/bold]\n{initial_prompt}",
        title="[cyan]Clarifier v2: 探索+提议模式[/cyan]",
        border_style="cyan",
    ))

    # Phase 1: Deep exploration with AI
    console.print("\n[yellow]🔍 Phase 1: 深度探索中...[/yellow]")
    console.print("[dim]Agent 正在研究可能的实现方案，这可能需要几分钟...[/dim]\n")

    explore_prompt = CLARIFIER_V2_EXPLORE_PROMPT.format(user_request=initial_prompt)

    exploration_result = ""
    async for message in query(
        prompt=explore_prompt,
        options=ClaudeCodeOptions(
            system_prompt=CLARIFIER_V2_SYSTEM_PROMPT,
            allowed_tools=[
                "Read", "Glob", "Grep", "LSP",
                "WebFetch", "WebSearch", "Task",
            ],
            max_turns=20,  # Allow more turns for deep exploration
            cwd=cwd,
        ),
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    exploration_result += block.text

    # Display the exploration result
    console.print("\n[bold cyan]═══ 探索完成 ═══[/bold cyan]\n")
    console.print(Panel(exploration_result, title="方案分析", border_style="cyan"))

    # Phase 2: User selection
    console.print("\n[bold yellow]请选择一个方案:[/bold yellow]\n")

    # Parse proposals from the exploration result to create options
    # Look for "方案 A:", "方案 B:", etc.
    proposal_pattern = r"###\s*方案\s*([A-Z]):\s*([^\n]+)"
    proposals = re.findall(proposal_pattern, exploration_result)

    if proposals:
        options = [f"方案 {letter}: {name.strip()}" for letter, name in proposals]
        options.append("其他想法 (自己输入)")

        selection = await ask_select(
            "选择你想要的方案:",
            options,
            allow_custom=False,
        )

        if "其他想法" in selection:
            selection = await questionary.text(
                "请描述你的想法:",
                style=custom_style,
            ).ask_async() or ""
    else:
        # Fallback if parsing failed
        selection = await questionary.text(
            "选择一个方案 (A/B/C) 或输入其他想法:",
            style=custom_style,
        ).ask_async() or "A"

    console.print(f"\n[green]✓ 选择了: {selection}[/green]")

    # Phase 3: Generate clarified goal based on selection
    console.print("\n[yellow]📝 Phase 2: 生成明确目标...[/yellow]\n")

    goal_generation_prompt = f"""用户的原始需求：
{initial_prompt}

探索分析结果：
{exploration_result}

用户选择：
{selection}

请根据用户的选择，生成一个明确、可执行的目标描述。格式要求：

1. **Clarified Description** - 基于选择的方案，详细描述要做什么
2. **Scope** - 包含哪些功能
3. **Non-goals** - 明确不包含什么
4. **Technical Approach** - 选定方案的技术细节
5. **Risks and Mitigations** - 主要风险和应对策略

使用 Markdown 格式输出。
"""

    summary_text = ""
    async for message in query(
        prompt=goal_generation_prompt,
        options=ClaudeCodeOptions(
            system_prompt="你是一个帮助生成项目目标文档的助手。请基于用户的选择生成清晰、详细的目标描述。",
            allowed_tools=[],
            max_turns=1,
            cwd=cwd,
        ),
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    summary_text += block.text

    console.print(Panel(summary_text, title="明确后的目标", border_style="green"))

    # Phase 4: Confirm or edit
    confirm = await ask_select(
        "确认这个目标？",
        ["确认", "需要修改"],
        allow_custom=False,
    )

    if confirm == "需要修改":
        edited = await questionary.text(
            "请输入修改后的目标描述:",
            style=custom_style,
        ).ask_async()
        if edited:
            summary_text = edited

    # Phase 5: Configure metrics
    metrics_config, eval_config, _ = await clarify_metrics(summary_text)

    # Build goal.md
    goal_content = generate_goal_md(
        initial_prompt=initial_prompt,
        summary_text=summary_text,
        metrics_config=metrics_config,
        eval_config=eval_config,
        qa_text=f"## 探索分析\n\n{exploration_result}",
        answers_text=f"用户选择: {selection}",
    )

    # Write to goal.md
    write_goal(goal_content, cwd)
    console.print("\n[green]✓ Goal saved to .ralph/goal.md[/green]")

    return goal_content


async def clarify_requirements_v2(
    initial_prompt: str,
    cwd: str = ".",
    mode: str = "auto",
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

    Returns:
        The clarified goal content
    """
    if mode == "ask":
        return await clarify_requirements(initial_prompt, cwd)
    elif mode == "explore":
        return await explore_and_propose(initial_prompt, cwd)
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
            return await explore_and_propose(initial_prompt, cwd)
        else:
            console.print("[dim]需求相对明确，使用传统 Q&A 模式[/dim]")
            return await clarify_requirements(initial_prompt, cwd)
