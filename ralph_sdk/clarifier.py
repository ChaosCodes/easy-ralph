"""Requirements clarification module."""

import asyncio
from claude_code_sdk import query, ClaudeCodeOptions, AssistantMessage
from ralph_sdk.models import ClarifiedRequirements
from ralph_sdk.prompts import CLARIFIER_SYSTEM_PROMPT
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel

console = Console()


async def clarify_requirements(initial_prompt: str) -> ClarifiedRequirements:
    """
    Clarify requirements through interactive Q&A with the user.

    Args:
        initial_prompt: The user's initial feature request

    Returns:
        ClarifiedRequirements with all gathered information
    """
    result = ClarifiedRequirements(original_prompt=initial_prompt)

    console.print(Panel(f"[bold]Feature Request:[/bold]\n{initial_prompt}", title="Input"))

    # Phase 1: Generate clarifying questions
    console.print("\n[yellow]Analyzing requirements and generating questions...[/yellow]\n")

    questions_prompt = f"""User's feature request:
{initial_prompt}

Generate 3-5 clarifying questions with lettered options to better understand the requirements.
Focus on scope, target users, core functionality, and success criteria.
"""

    questions_text = ""
    async for message in query(
        prompt=questions_prompt,
        options=ClaudeCodeOptions(
            system_prompt=CLARIFIER_SYSTEM_PROMPT,
            max_turns=1,
        ),
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    questions_text += block.text

    # Display questions and get answers
    console.print(Panel(questions_text, title="Clarifying Questions"))

    answers = Prompt.ask(
        "\n[bold cyan]Your answers[/bold cyan] (e.g., 1A, 2B, 3C or type detailed response)"
    )

    result.clarifications["questions"] = questions_text
    result.clarifications["answers"] = answers

    # Phase 2: Generate final summary
    console.print("\n[yellow]Generating clarified requirements...[/yellow]\n")

    summary_prompt = f"""Original request:
{initial_prompt}

Questions asked:
{questions_text}

User's answers:
{answers}

Based on this, provide:
1. A clear, detailed description of what needs to be built
2. A suggested project name (short, descriptive)
3. The scope (what's included)
4. Non-goals (what's explicitly NOT included)

Format as:
PROJECT: [name]
DESCRIPTION: [detailed description]
SCOPE: [what's included]
NON-GOALS: [comma-separated list of what's not included]
"""

    summary_text = ""
    async for message in query(
        prompt=summary_prompt,
        options=ClaudeCodeOptions(
            system_prompt=CLARIFIER_SYSTEM_PROMPT,
            max_turns=1,
        ),
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    summary_text += block.text

    # Parse the summary
    for line in summary_text.split("\n"):
        line = line.strip()
        if line.startswith("PROJECT:"):
            result.project_name = line.replace("PROJECT:", "").strip()
        elif line.startswith("DESCRIPTION:"):
            result.final_description = line.replace("DESCRIPTION:", "").strip()
        elif line.startswith("SCOPE:"):
            result.scope = line.replace("SCOPE:", "").strip()
        elif line.startswith("NON-GOALS:"):
            non_goals = line.replace("NON-GOALS:", "").strip()
            result.non_goals = [g.strip() for g in non_goals.split(",") if g.strip()]

    # If parsing failed, use the raw summary
    if not result.final_description:
        result.final_description = summary_text

    console.print(Panel(summary_text, title="Clarified Requirements"))

    # Confirm with user
    confirm = Prompt.ask(
        "\n[bold]Proceed with these requirements?[/bold]",
        choices=["y", "n", "edit"],
        default="y",
    )

    if confirm == "n":
        raise KeyboardInterrupt("User cancelled")
    elif confirm == "edit":
        edited = Prompt.ask("Enter your revised requirements")
        result.final_description = edited

    return result


async def quick_clarify(initial_prompt: str) -> ClarifiedRequirements:
    """
    Quick clarification without interactive Q&A.
    Useful for simple, well-defined requests.
    """
    result = ClarifiedRequirements(
        original_prompt=initial_prompt,
        final_description=initial_prompt,
    )

    # Extract project name from prompt
    words = initial_prompt.split()[:3]
    result.project_name = "".join(w.capitalize() for w in words if w.isalnum())

    return result
