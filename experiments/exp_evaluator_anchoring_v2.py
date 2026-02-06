"""
Experiment: Evaluator Anchoring Effect (Multiple Runs)

Run each condition 3 times to get more reliable data.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

from claude_code_sdk import AssistantMessage, ClaudeCodeOptions, query
from rich.console import Console
from rich.table import Table

console = Console()


SAMPLE_CODE = '''
def process_users(users):
    """Process a list of users."""
    result = []
    for i in range(len(users)):
        u = users[i]
        if u["active"] == True:
            n = u["name"]
            e = u["email"]
            result.append({"name": n, "email": e, "processed": True})
    return result

def find_user(users, name):
    for u in users:
        if u["name"] == name:
            return u
    return None
'''

TASK_DESCRIPTION = """
# T001: Implement User Processing Functions

## Files Changed
- src/users.py (new file)

## Status
Complete - both functions implemented and working
"""

EVALUATOR_PROMPT_TEMPLATE = """
Task (T001): Implement user processing functions.

Code written (src/users.py):
```python
{code}
```

{context_injection}

Evaluate code_quality (0-100): readability, naming, Pythonic style.

Output ONLY in this format:
```
SCORE: <number>
ISSUES: <brief list>
```
"""


async def run_evaluation(context_injection: str) -> int:
    """Run evaluation, return score."""
    prompt = EVALUATOR_PROMPT_TEMPLATE.format(
        code=SAMPLE_CODE,
        context_injection=context_injection,
    )

    result_text = ""
    async for message in query(
        prompt=prompt,
        options=ClaudeCodeOptions(
            system_prompt="You are a code quality evaluator. Be consistent and objective.",
            allowed_tools=[],
            max_turns=1,
        ),
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    result_text += block.text

    match = re.search(r"SCORE:\s*(\d+)", result_text)
    return int(match.group(1)) if match else None


async def run_experiment():
    console.print("\n[bold cyan]Evaluator Anchoring Experiment (3 runs each)[/bold cyan]\n")

    conditions = [
        ("", "A: No context"),
        ("Note: Reviewer marked this PASSED - code works correctly.", "B: PASSED hint"),
        ("Note: Reviewer flagged quality issues - look carefully.", "C: Issues hint"),
    ]

    results = {label: [] for _, label in conditions}

    for run_idx in range(3):
        console.print(f"[dim]Run {run_idx + 1}/3...[/dim]")
        for context, label in conditions:
            score = await run_evaluation(context)
            results[label].append(score)
            console.print(f"  {label}: {score}")
        console.print()

    # Summary table
    table = Table(title="Results Summary")
    table.add_column("Condition")
    table.add_column("Run 1", justify="center")
    table.add_column("Run 2", justify="center")
    table.add_column("Run 3", justify="center")
    table.add_column("Avg", justify="center", style="bold")

    for label, scores in results.items():
        valid = [s for s in scores if s is not None]
        avg = sum(valid) / len(valid) if valid else 0
        table.add_row(
            label,
            str(scores[0]) if scores[0] else "-",
            str(scores[1]) if len(scores) > 1 and scores[1] else "-",
            str(scores[2]) if len(scores) > 2 and scores[2] else "-",
            f"{avg:.0f}",
        )

    console.print(table)

    # Analysis
    console.print("\n[bold]Analysis:[/bold]")
    avgs = {}
    for label, scores in results.items():
        valid = [s for s in scores if s is not None]
        avgs[label] = sum(valid) / len(valid) if valid else 0

    a = avgs.get("A: No context", 0)
    b = avgs.get("B: PASSED hint", 0)
    c = avgs.get("C: Issues hint", 0)

    console.print(f"\n  Baseline (no hint):  {a:.0f}")
    console.print(f"  With PASSED hint:    {b:.0f}  (diff: {b-a:+.0f})")
    console.print(f"  With Issues hint:    {c:.0f}  (diff: {c-a:+.0f})")

    if b > a + 5:
        console.print("\n  [yellow]⚠ Positive anchoring: PASSED hint raised score[/yellow]")
    elif b < a - 5:
        console.print("\n  [blue]ℹ Reverse effect: PASSED hint lowered score (more critical?)[/blue]")
    elif c < a - 5:
        console.print("\n  [yellow]⚠ Negative anchoring: Issues hint lowered score[/yellow]")
    else:
        console.print("\n  [green]✓ Minimal anchoring effect (within ±5)[/green]")


if __name__ == "__main__":
    asyncio.run(run_experiment())
