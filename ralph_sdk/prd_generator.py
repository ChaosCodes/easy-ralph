"""PRD generation module."""

import json
import re
from claude_code_sdk import query, ClaudeCodeOptions, AssistantMessage
from ralph_sdk.models import PRD, UserStory, ClarifiedRequirements
from ralph_sdk.prompts import PRD_GENERATOR_SYSTEM_PROMPT
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def to_kebab_case(text: str) -> str:
    """Convert text to kebab-case."""
    # Remove non-alphanumeric characters except spaces
    text = re.sub(r"[^\w\s-]", "", text.lower())
    # Replace spaces with hyphens
    text = re.sub(r"[\s_]+", "-", text)
    # Remove leading/trailing hyphens
    return text.strip("-")


async def generate_prd(requirements: ClarifiedRequirements, cwd: str = ".") -> PRD:
    """
    Generate a PRD from clarified requirements.

    Args:
        requirements: Clarified requirements from the clarifier
        cwd: Working directory for code exploration

    Returns:
        PRD object with user stories
    """
    console.print("\n[yellow]Generating PRD with user stories...[/yellow]\n")

    prompt = f"""Generate a PRD for this feature:

Project: {requirements.project_name or "MyProject"}
Description: {requirements.final_description}
Scope: {requirements.scope or "As described"}
Non-goals: {", ".join(requirements.non_goals) if requirements.non_goals else "None specified"}

Original request: {requirements.original_prompt}
Additional context: {json.dumps(requirements.clarifications)}

Explore the codebase first to understand:
1. Existing patterns and conventions
2. Database schema (if relevant)
3. UI component structure (if relevant)
4. Testing patterns

Then generate a PRD with properly sized, ordered user stories.
Output ONLY valid JSON.
"""

    prd_json = ""
    async for message in query(
        prompt=prompt,
        options=ClaudeCodeOptions(
            system_prompt=PRD_GENERATOR_SYSTEM_PROMPT,
            allowed_tools=["Read", "Glob", "Grep"],
            max_turns=10,
            cwd=cwd,
        ),
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    prd_json += block.text

    # Extract JSON from response
    prd_data = extract_json(prd_json)

    if not prd_data:
        raise ValueError(f"Failed to parse PRD JSON from response:\n{prd_json[:500]}")

    # Convert to PRD model
    prd = PRD(
        project=prd_data.get("project", requirements.project_name or "MyProject"),
        branch_name=prd_data.get(
            "branchName", f"ralph/{to_kebab_case(requirements.project_name or 'feature')}"
        ),
        description=prd_data.get("description", requirements.final_description),
        user_stories=[
            UserStory(
                id=s.get("id", f"US-{i+1:03d}"),
                title=s.get("title", ""),
                description=s.get("description", ""),
                acceptance_criteria=s.get("acceptanceCriteria", []),
                priority=s.get("priority", i + 1),
                passes=s.get("passes", False),
                notes=s.get("notes", ""),
            )
            for i, s in enumerate(prd_data.get("userStories", []))
        ],
    )

    # Display PRD summary
    display_prd(prd)

    return prd


def extract_json(text: str) -> dict | None:
    """Extract JSON object from text that may contain other content."""
    # Try to find JSON in code blocks first
    code_block_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON object
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    # Try the whole text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def display_prd(prd: PRD) -> None:
    """Display PRD summary in a nice table."""
    console.print(
        Panel(
            f"[bold]{prd.project}[/bold]\n{prd.description}\n\nBranch: [cyan]{prd.branch_name}[/cyan]",
            title="PRD Generated",
        )
    )

    table = Table(title="User Stories")
    table.add_column("ID", style="cyan")
    table.add_column("Title", style="white")
    table.add_column("Priority", justify="center")
    table.add_column("Criteria", justify="center")

    for story in prd.user_stories:
        table.add_row(
            story.id,
            story.title[:50] + "..." if len(story.title) > 50 else story.title,
            str(story.priority),
            str(len(story.acceptance_criteria)),
        )

    console.print(table)


def save_prd(prd: PRD, filepath: str) -> None:
    """Save PRD to JSON file."""
    data = {
        "project": prd.project,
        "branchName": prd.branch_name,
        "description": prd.description,
        "userStories": [
            {
                "id": s.id,
                "title": s.title,
                "description": s.description,
                "acceptanceCriteria": s.acceptance_criteria,
                "priority": s.priority,
                "passes": s.passes,
                "notes": s.notes,
            }
            for s in prd.user_stories
        ],
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    console.print(f"\n[green]PRD saved to {filepath}[/green]")


def load_prd(filepath: str) -> PRD:
    """Load PRD from JSON file."""
    with open(filepath) as f:
        data = json.load(f)

    return PRD(
        project=data.get("project", ""),
        branch_name=data.get("branchName", ""),
        description=data.get("description", ""),
        user_stories=[
            UserStory(
                id=s.get("id", ""),
                title=s.get("title", ""),
                description=s.get("description", ""),
                acceptance_criteria=s.get("acceptanceCriteria", []),
                priority=s.get("priority", 0),
                passes=s.get("passes", False),
                notes=s.get("notes", ""),
            )
            for s in data.get("userStories", [])
        ],
    )
