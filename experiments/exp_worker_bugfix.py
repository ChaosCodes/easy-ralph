"""
Experiment: Worker Bug-Finding with Reflection

Instead of testing if Worker makes mistakes, test if reflection
helps Worker find and fix bugs in EXISTING code.

Scenario: Given buggy code, can Worker find and fix all bugs?
"""

import asyncio
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from claude_code_sdk import AssistantMessage, ClaudeCodeOptions, query
from rich.console import Console
from rich.table import Table

console = Console()


def setup_test_project(base_dir: Path, name: str) -> Path:
    project_dir = base_dir / name
    if project_dir.exists():
        shutil.rmtree(project_dir)
    project_dir.mkdir(parents=True)
    (project_dir / "src").mkdir()
    (project_dir / "src" / "__init__.py").write_text("")

    # Buggy code - 3 bugs:
    # 1. Off-by-one in range (should be n+1)
    # 2. Wrong operator (should be <=)
    # 3. Missing None check
    buggy_code = '''"""Number utilities with bugs."""

def is_prime(n: int) -> bool:
    """Check if n is prime. Has bugs!"""
    if n < 2:
        return False
    for i in range(2, n):  # BUG 1: should be int(n**0.5) + 1
        if n % i == 0:
            return False
    return True

def fibonacci(n: int) -> list[int]:
    """Return first n fibonacci numbers. Has bugs!"""
    if n < 0:  # BUG 2: should also handle n == 0
        return []
    result = [0, 1]
    for i in range(2, n):  # BUG 3: off by one, produces n-1 numbers
        result.append(result[i-1] + result[i-2])
    return result

def factorial(n):  # BUG 4: no type hint, no None check
    """Calculate factorial."""
    if n == 0:
        return 1
    return n * factorial(n - 1)
'''
    (project_dir / "src" / "numbers.py").write_text(buggy_code)
    return project_dir


TASK_PROMPT = """
Review and fix bugs in src/numbers.py.

The file contains three functions with bugs:
1. is_prime(n) - check if number is prime
2. fibonacci(n) - return first n fibonacci numbers
3. factorial(n) - calculate factorial

Find and fix all bugs. Make sure:
- is_prime works correctly for all inputs
- fibonacci(5) returns [0, 1, 1, 2, 3] (exactly 5 numbers)
- factorial handles None and negative inputs gracefully
"""

WORKER_PROMPT_BASELINE = """You are a Python developer.

Task: {task}

Read the code, find the bugs, and fix them.
"""

WORKER_PROMPT_REFLECTION = """You are a Python developer doing code review.

Task: {task}

Process:
1. Read src/numbers.py carefully
2. For EACH function, trace through these test cases:
   - is_prime: 0, 1, 2, 3, 4, 17, 25
   - fibonacci: 0, 1, 5, 10
   - factorial: None, -1, 0, 5
3. List each bug you find with line number
4. Fix all bugs
5. Re-read and verify your fixes work for the test cases
"""


async def run_worker(project_dir: Path, prompt_template: str, task: str) -> str:
    prompt = prompt_template.format(task=task)
    async for message in query(
        prompt=prompt,
        options=ClaudeCodeOptions(
            system_prompt="You are a careful code reviewer.",
            allowed_tools=["Read", "Write", "Edit"],
            permission_mode="acceptEdits",
            max_turns=20,
            cwd=str(project_dir),
        ),
    ):
        pass

    code_file = project_dir / "src" / "numbers.py"
    return code_file.read_text() if code_file.exists() else ""


def evaluate_code(code: str) -> dict:
    """Test if bugs are fixed."""
    test_cases = [
        # is_prime tests
        ("is_prime(0)", False, "is_prime(0)"),
        ("is_prime(1)", False, "is_prime(1)"),
        ("is_prime(2)", True, "is_prime(2)"),
        ("is_prime(17)", True, "is_prime(17)"),
        ("is_prime(25)", False, "is_prime(25)"),
        # fibonacci tests
        ("fibonacci(0)", [], "fibonacci(0)"),
        ("fibonacci(1)", [0], "fibonacci(1)"),
        ("fibonacci(5)", [0, 1, 1, 2, 3], "fibonacci(5)"),
        ("len(fibonacci(10))", 10, "fibonacci(10) length"),
        # factorial tests
        ("factorial(0)", 1, "factorial(0)"),
        ("factorial(5)", 120, "factorial(5)"),
    ]

    import tempfile
    import importlib.util

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_path = f.name

    try:
        spec = importlib.util.spec_from_file_location("test_module", temp_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        passed = 0
        failed = []

        for expr, expected, desc in test_cases:
            try:
                # Create a namespace with the module's functions
                namespace = {
                    'is_prime': getattr(module, 'is_prime', lambda x: None),
                    'fibonacci': getattr(module, 'fibonacci', lambda x: None),
                    'factorial': getattr(module, 'factorial', lambda x: None),
                }
                result = eval(expr, namespace)
                if result == expected:
                    passed += 1
                else:
                    failed.append(f"{desc}: got {result!r}, expected {expected!r}")
            except Exception as e:
                failed.append(f"{desc}: {type(e).__name__}")

        return {
            "passed": passed,
            "total": len(test_cases),
            "failed_cases": failed,
            "score": passed / len(test_cases) * 100,
        }

    except Exception as e:
        return {"error": str(e), "passed": 0, "total": len(test_cases), "failed_cases": [str(e)]}
    finally:
        Path(temp_path).unlink(missing_ok=True)


async def run_experiment():
    console.print("\n[bold cyan]Experiment: Bug-Finding with Reflection[/bold cyan]\n")
    console.print("Task: Fix bugs in existing code")
    console.print("Testing if reflection helps find more bugs\n")

    base_dir = Path("/tmp/ralph_exp_bugfix")
    base_dir.mkdir(exist_ok=True)

    results = {"baseline": [], "reflection": []}
    runs = 2

    for run_idx in range(runs):
        console.print(f"[bold]Run {run_idx + 1}/{runs}[/bold]\n")

        console.print("  [yellow]Baseline...[/yellow]")
        project_a = setup_test_project(base_dir, f"baseline_{run_idx}")
        code_a = await run_worker(project_a, WORKER_PROMPT_BASELINE, TASK_PROMPT)
        eval_a = evaluate_code(code_a)
        results["baseline"].append(eval_a)
        console.print(f"    {eval_a.get('passed', 0)}/{eval_a.get('total', 0)} tests pass")
        for fc in eval_a.get("failed_cases", [])[:3]:
            console.print(f"    [red]x {fc}[/red]")

        console.print("  [yellow]Reflection...[/yellow]")
        project_b = setup_test_project(base_dir, f"reflection_{run_idx}")
        code_b = await run_worker(project_b, WORKER_PROMPT_REFLECTION, TASK_PROMPT)
        eval_b = evaluate_code(code_b)
        results["reflection"].append(eval_b)
        console.print(f"    {eval_b.get('passed', 0)}/{eval_b.get('total', 0)} tests pass")
        for fc in eval_b.get("failed_cases", [])[:3]:
            console.print(f"    [red]x {fc}[/red]")

        console.print()

    # Summary
    table = Table(title="Bug-Fix Results")
    table.add_column("Variant")
    table.add_column("Run 1")
    table.add_column("Run 2")
    table.add_column("Avg", style="bold")

    for variant, evals in results.items():
        scores = [e.get("score", 0) for e in evals]
        avg = sum(scores) / len(scores) if scores else 0
        table.add_row(
            variant,
            f"{scores[0]:.0f}%",
            f"{scores[1]:.0f}%" if len(scores) > 1 else "-",
            f"{avg:.0f}%",
        )

    console.print(table)

    baseline_avg = sum(e.get("score", 0) for e in results["baseline"]) / len(results["baseline"])
    reflection_avg = sum(e.get("score", 0) for e in results["reflection"]) / len(results["reflection"])

    console.print(f"\n  Baseline:   {baseline_avg:.0f}%")
    console.print(f"  Reflection: {reflection_avg:.0f}%")
    console.print(f"  Diff:       {reflection_avg - baseline_avg:+.0f}%")

    if reflection_avg > baseline_avg + 5:
        console.print("\n  [green]Reflection helps find more bugs![/green]")
    else:
        console.print("\n  [dim]No significant difference[/dim]")


if __name__ == "__main__":
    asyncio.run(run_experiment())
