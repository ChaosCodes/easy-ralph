"""Requirements clarification module."""

from claude_code_sdk import AssistantMessage, ClaudeCodeOptions, query
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from ralph_sdk.models import ClarifiedRequirements
from ralph_sdk.prompts import CLARIFIER_SYSTEM_PROMPT

console = Console()

# Mapping of prefixes to result fields
_SUMMARY_FIELDS = {
    "PROJECT:": "project_name",
    "DESCRIPTION:": "final_description",
    "SCOPE:": "scope",
}


async def clarify_requirements(initial_prompt: str, cwd: str = ".") -> ClarifiedRequirements:
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

First, explore the codebase to understand:
1. Project structure and tech stack
2. Existing patterns and conventions
3. Related existing functionality

Then generate 3-5 clarifying questions with lettered options to better understand the requirements.
Focus on scope, target users, core functionality, and success criteria.
"""

    questions_text = ""
    async for message in query(
        prompt=questions_prompt,
        options=ClaudeCodeOptions(
            system_prompt=CLARIFIER_SYSTEM_PROMPT,
            allowed_tools=[
                "Read", "Glob", "Grep", "LSP",  # Explore codebase
                "WebFetch", "WebSearch",         # Research best practices & docs
            ],
            max_turns=8,  # Allow exploration + research before generating questions
            cwd=cwd,
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
            allowed_tools=[
                "Read", "Glob", "Grep",      # Check codebase for scope
                "WebFetch", "WebSearch",     # Verify technical feasibility
            ],
            max_turns=5,
            cwd=cwd,
        ),
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    summary_text += block.text

    # Parse the summary
    _parse_summary_into_result(summary_text, result)

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
    # Extract project name from first 3 alphanumeric words
    words = [w for w in initial_prompt.split()[:3] if w.isalnum()]
    project_name = "".join(w.capitalize() for w in words)

    return ClarifiedRequirements(
        original_prompt=initial_prompt,
        final_description=initial_prompt,
        project_name=project_name,
    )


def _parse_summary_into_result(summary_text: str, result: ClarifiedRequirements) -> None:
    """Parse summary text and populate result fields."""
    for line in summary_text.split("\n"):
        line = line.strip()

        # Handle simple fields
        for prefix, field in _SUMMARY_FIELDS.items():
            if line.startswith(prefix):
                setattr(result, field, line[len(prefix):].strip())
                break

        # Handle non-goals separately (needs list parsing)
        if line.startswith("NON-GOALS:"):
            value = line[len("NON-GOALS:"):].strip()
            result.non_goals = [g.strip() for g in value.split(",") if g.strip()]
