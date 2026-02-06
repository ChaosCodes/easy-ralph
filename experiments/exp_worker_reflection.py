"""
Experiment: Worker Self-Reflection

Hypothesis: Adding explicit self-check step improves code quality on first pass.

Task: Implement a function with common edge cases that are easy to miss.
Compare Worker with and without explicit reflection prompt.

This is a more realistic test - we actually run the Worker and check the result.
"""

import asyncio
import shutil
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from claude_code_sdk import AssistantMessage, ClaudeCodeOptions, query
from rich.console import Console
from rich.table import Table

console = Console()


def setup_test_project(base_dir: Path, name: str) -> Path:
    """Create a minimal test project."""
    project_dir = base_dir / name
    if project_dir.exists():
        shutil.rmtree(project_dir)
    project_dir.mkdir(parents=True)

    (project_dir / "src").mkdir()
    (project_dir / "src" / "__init__.py").write_text("")
    (project_dir / "src" / "validators.py").write_text('"""Validation utilities."""\n')

    return project_dir


# The task - validate email with many edge cases
TASK_PROMPT = """
Implement `validate_email(email: str) -> bool` in src/validators.py.

Requirements:
- Return True for valid emails, False for invalid
- Must have exactly one @ symbol
- Local part (before @) must not be empty
- Domain (after @) must have at least one dot
- No spaces allowed anywhere
- Handle None and empty string gracefully

After implementation, write the code to the file.
"""

WORKER_PROMPT_BASELINE = """You are a Python developer.

Task: {task}

Write the implementation directly to src/validators.py using the Write tool.
"""

WORKER_PROMPT_REFLECTION = """You are a Python developer who practices defensive coding.

Task: {task}

Process:
1. Implement the function
2. Write it to src/validators.py
3. IMPORTANT: After writing, read the file back and trace through these test cases mentally:
   - None
   - ""
   - "test"
   - "test@"
   - "@domain.com"
   - "test@domain"
   - "test@@domain.com"
   - "test @domain.com"
   - "valid@domain.com"
4. If you find any bugs, fix them immediately
5. Show me your trace results
"""


async def run_worker(project_dir: Path, prompt_template: str, task: str) -> dict:
    """Run worker and return the generated code."""
    prompt = prompt_template.format(task=task)

    result_text = ""
    async for message in query(
        prompt=prompt,
        options=ClaudeCodeOptions(
            system_prompt="You are a careful Python developer.",
            allowed_tools=["Read", "Write", "Edit"],
            permission_mode="acceptEdits",
            max_turns=10,
            cwd=str(project_dir),
        ),
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    result_text += block.text

    # Read the generated code
    code_file = project_dir / "src" / "validators.py"
    code = code_file.read_text() if code_file.exists() else ""

    return {
        "code": code,
        "response": result_text,
    }


def evaluate_code(code: str) -> dict:
    """Evaluate the generated code against test cases."""
    # Try to import and test the function
    test_cases = [
        (None, False, "None input"),
        ("", False, "Empty string"),
        ("test", False, "No @ symbol"),
        ("test@", False, "Empty domain"),
        ("@domain.com", False, "Empty local part"),
        ("test@domain", False, "No dot in domain"),
        ("test@@domain.com", False, "Multiple @ symbols"),
        ("test @domain.com", False, "Space in email"),
        ("test@domain.com", True, "Valid email"),
        ("user.name@sub.domain.com", True, "Valid complex email"),
    ]

    # Create a temp module to test
    import tempfile
    import importlib.util

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_path = f.name

    try:
        spec = importlib.util.spec_from_file_location("test_module", temp_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, 'validate_email'):
            return {"error": "Function not found", "passed": 0, "total": len(test_cases)}

        func = module.validate_email

        passed = 0
        failed = []
        for input_val, expected, desc in test_cases:
            try:
                result = func(input_val)
                if result == expected:
                    passed += 1
                else:
                    failed.append(f"{desc}: got {result}, expected {expected}")
            except Exception as e:
                failed.append(f"{desc}: raised {type(e).__name__}: {e}")

        return {
            "passed": passed,
            "total": len(test_cases),
            "failed_cases": failed,
            "score": passed / len(test_cases) * 100,
        }

    except Exception as e:
        return {"error": str(e), "passed": 0, "total": len(test_cases)}
    finally:
        Path(temp_path).unlink(missing_ok=True)


async def run_experiment():
    console.print("\n[bold cyan]Experiment: Worker Self-Reflection[/bold cyan]\n")
    console.print("Task: Implement email validation with many edge cases")
    console.print("Compare: Baseline prompt vs Reflection prompt\n")

    base_dir = Path("/tmp/ralph_exp_reflection")
    base_dir.mkdir(exist_ok=True)

    results = {"baseline": [], "reflection": []}
    runs_per_variant = 2

    for run_idx in range(runs_per_variant):
        console.print(f"[bold]Run {run_idx + 1}/{runs_per_variant}[/bold]\n")

        # Baseline
        console.print("  [yellow]Running baseline (no reflection)...[/yellow]")
        project_a = setup_test_project(base_dir, f"baseline_{run_idx}")
        result_a = await run_worker(project_a, WORKER_PROMPT_BASELINE, TASK_PROMPT)
        eval_a = evaluate_code(result_a["code"])
        results["baseline"].append(eval_a)
        console.print(f"    Passed: {eval_a.get('passed', 0)}/{eval_a.get('total', 0)}")
        if eval_a.get("failed_cases"):
            for fc in eval_a["failed_cases"][:3]:
                console.print(f"    [red]✗ {fc}[/red]")

        # Reflection
        console.print("  [yellow]Running with reflection prompt...[/yellow]")
        project_b = setup_test_project(base_dir, f"reflection_{run_idx}")
        result_b = await run_worker(project_b, WORKER_PROMPT_REFLECTION, TASK_PROMPT)
        eval_b = evaluate_code(result_b["code"])
        results["reflection"].append(eval_b)
        console.print(f"    Passed: {eval_b.get('passed', 0)}/{eval_b.get('total', 0)}")
        if eval_b.get("failed_cases"):
            for fc in eval_b["failed_cases"][:3]:
                console.print(f"    [red]✗ {fc}[/red]")

        console.print()

    # Summary
    console.print("\n[bold]Results Summary:[/bold]\n")

    table = Table(title="Worker Reflection Experiment")
    table.add_column("Variant")
    table.add_column("Run 1", justify="center")
    table.add_column("Run 2", justify="center")
    table.add_column("Avg Score", justify="center", style="bold")

    for variant, evals in results.items():
        scores = [e.get("score", 0) for e in evals]
        avg = sum(scores) / len(scores) if scores else 0
        table.add_row(
            variant.capitalize(),
            f"{scores[0]:.0f}%" if len(scores) > 0 else "-",
            f"{scores[1]:.0f}%" if len(scores) > 1 else "-",
            f"{avg:.0f}%",
        )

    console.print(table)

    # Analysis
    baseline_avg = sum(e.get("score", 0) for e in results["baseline"]) / len(results["baseline"])
    reflection_avg = sum(e.get("score", 0) for e in results["reflection"]) / len(results["reflection"])

    console.print(f"\n[bold]Analysis:[/bold]")
    console.print(f"  Baseline avg:    {baseline_avg:.0f}%")
    console.print(f"  Reflection avg:  {reflection_avg:.0f}%")
    console.print(f"  Improvement:     {reflection_avg - baseline_avg:+.0f}%")

    if reflection_avg > baseline_avg + 10:
        console.print("\n  [green]✓ Reflection significantly improves code quality![/green]")
    elif reflection_avg > baseline_avg:
        console.print("\n  [yellow]ℹ Small improvement with reflection[/yellow]")
    else:
        console.print("\n  [dim]No clear improvement from reflection prompt[/dim]")


if __name__ == "__main__":
    asyncio.run(run_experiment())
