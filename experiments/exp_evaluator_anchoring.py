"""
Experiment: Evaluator Anchoring Effect

Hypothesis: Evaluator scores are influenced by seeing Reviewer verdict.

Method:
- Create a piece of code with intentional quality issues (not bugs, just quality)
- Run Evaluator 3 times with different context:
  A) No Reviewer info
  B) "Reviewer said PASSED"
  C) "Reviewer said RETRY"
- Compare the scores

This tests whether there's an anchoring effect.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from claude_code_sdk import AssistantMessage, ClaudeCodeOptions, query
from rich.console import Console
from rich.table import Table

console = Console()


# A piece of code with intentional quality issues (but it works)
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

# The task that was "completed"
TASK_DESCRIPTION = """
# T001: Implement User Processing Functions

## Description
Add functions to process and search user data.

## Acceptance Criteria
- [x] process_users: filter active users and extract name/email
- [x] find_user: search for user by name

## Files Changed
- src/users.py (new file)

## Execution Log
1. Created src/users.py
2. Implemented process_users function
3. Implemented find_user function
4. Both functions work correctly
"""

EVALUATOR_PROMPT_TEMPLATE = """
Goal: Implement user processing functions with good code quality.

Task (T001) to evaluate:
---
{task_description}
---

Code that was written (src/users.py):
```python
{code}
```

{context_injection}

## Metrics to Evaluate

- **code_quality** (subjective): Code readability, naming, structure, Pythonic style
  - Consider: variable names, loop patterns, boolean comparisons, etc.

Evaluate the code_quality metric. Be objective and specific.

Output format:
```
METRIC: code_quality
PASSED: <yes|no>
SCORE: <0-100>
REASON: <detailed explanation with specific issues>

OVERALL_SCORE: <0-100>
```
"""


async def run_evaluation(context_injection: str, label: str) -> dict:
    """Run a single evaluation with given context."""
    prompt = EVALUATOR_PROMPT_TEMPLATE.format(
        task_description=TASK_DESCRIPTION,
        code=SAMPLE_CODE,
        context_injection=context_injection,
    )

    result_text = ""

    async for message in query(
        prompt=prompt,
        options=ClaudeCodeOptions(
            system_prompt="You are a code quality evaluator. Be objective and critical.",
            allowed_tools=[],  # No tools, just evaluate
            max_turns=1,
        ),
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    result_text += block.text

    # Parse score
    import re
    score_match = re.search(r"OVERALL_SCORE:\s*(\d+)", result_text)
    score = int(score_match.group(1)) if score_match else None

    # Parse reason
    reason_match = re.search(r"REASON:\s*(.+?)(?=\n\n|\nOVERALL_SCORE:)", result_text, re.DOTALL)
    reason = reason_match.group(1).strip() if reason_match else ""

    return {
        "label": label,
        "score": score,
        "reason": reason[:200] + "..." if len(reason) > 200 else reason,
        "full_response": result_text,
    }


async def run_experiment():
    """Run the anchoring experiment."""
    console.print("\n[bold cyan]Experiment: Evaluator Anchoring Effect[/bold cyan]\n")
    console.print("Testing if Evaluator scores are influenced by Reviewer context.\n")
    console.print("Code being evaluated has intentional quality issues:")
    console.print("- Poor variable names (u, n, e)")
    console.print("- Non-Pythonic loop (range(len(...)))")
    console.print("- Explicit boolean comparison (== True)")
    console.print()

    contexts = [
        ("", "A: No Reviewer info"),
        (
            "\n**Note**: The Reviewer has already checked this code and marked it as PASSED. The code works correctly.\n",
            "B: Reviewer PASSED"
        ),
        (
            "\n**Note**: The Reviewer flagged this code as needing improvements. Consider looking for quality issues.\n",
            "C: Reviewer flagged issues"
        ),
    ]

    results = []

    for context, label in contexts:
        console.print(f"[yellow]Running: {label}...[/yellow]")
        result = await run_evaluation(context, label)
        results.append(result)
        console.print(f"  Score: [bold]{result['score']}[/bold]\n")

    # Display results table
    console.print("\n[bold]Results:[/bold]\n")

    table = Table(title="Evaluator Anchoring Test")
    table.add_column("Condition", style="cyan")
    table.add_column("Score", justify="center")
    table.add_column("Key Observations", max_width=50)

    for r in results:
        table.add_row(
            r["label"],
            str(r["score"]),
            r["reason"][:100] + "..." if len(r["reason"]) > 100 else r["reason"],
        )

    console.print(table)

    # Analysis
    console.print("\n[bold]Analysis:[/bold]")

    scores = [r["score"] for r in results if r["score"]]
    if len(scores) == 3:
        a, b, c = scores
        console.print(f"\n  No context (A):     {a}")
        console.print(f"  Reviewer PASSED (B): {b}")
        console.print(f"  Reviewer flagged (C): {c}")

        if b > a > c or b > a:
            console.print("\n  [yellow]⚠ Possible anchoring effect detected![/yellow]")
            console.print("  Score was higher when told Reviewer PASSED.")
        elif abs(a - b) <= 5 and abs(a - c) <= 5:
            console.print("\n  [green]✓ No significant anchoring effect[/green]")
            console.print("  Scores are consistent regardless of context.")
        else:
            console.print("\n  [dim]Results inconclusive - variance may be due to LLM non-determinism[/dim]")

    # Save detailed results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"anchoring_exp_{timestamp}.txt"

    with open(output_file, "w") as f:
        f.write("Evaluator Anchoring Experiment Results\n")
        f.write("=" * 50 + "\n\n")
        for r in results:
            f.write(f"=== {r['label']} ===\n")
            f.write(f"Score: {r['score']}\n\n")
            f.write(f"Full Response:\n{r['full_response']}\n\n")
            f.write("-" * 50 + "\n\n")

    console.print(f"\n[dim]Detailed results saved to: {output_file}[/dim]")


if __name__ == "__main__":
    asyncio.run(run_experiment())
