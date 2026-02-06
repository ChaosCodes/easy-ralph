"""
Experiment: Worker Self-Reflection (Harder Task)

A more challenging task with subtle edge cases.
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
    (project_dir / "src" / "parser.py").write_text('"""CSV parsing utilities."""\n')
    return project_dir


TASK_PROMPT = """
Implement `parse_csv_line(line: str) -> list[str]` in src/parser.py.

Parse a single CSV line into fields.

Rules:
- Fields separated by commas
- Fields can be quoted: "value"
- Quoted fields can contain commas: "hello, world" -> hello, world
- Strip whitespace around unquoted fields
- Empty fields are valid: a,,b -> ["a", "", "b"]
- Handle None and empty string -> []
"""

WORKER_PROMPT_BASELINE = """You are a Python developer.

Task: {task}

Write the implementation to src/parser.py.
"""

WORKER_PROMPT_REFLECTION = """You are a Python developer.

Task: {task}

Process:
1. Implement the function
2. Write to src/parser.py
3. IMPORTANT: After writing, trace through these cases mentally:
   - None -> []
   - "" -> []
   - "a,b,c" -> ["a", "b", "c"]
   - "a, b , c" -> ["a", "b", "c"]  (whitespace stripped)
   - "a,,b" -> ["a", "", "b"]
4. Fix any bugs you find
"""


async def run_worker(project_dir: Path, prompt_template: str, task: str) -> str:
    prompt = prompt_template.format(task=task)
    async for message in query(
        prompt=prompt,
        options=ClaudeCodeOptions(
            system_prompt="You are a careful Python developer.",
            allowed_tools=["Read", "Write", "Edit"],
            permission_mode="acceptEdits",
            max_turns=15,
            cwd=str(project_dir),
        ),
    ):
        pass

    code_file = project_dir / "src" / "parser.py"
    return code_file.read_text() if code_file.exists() else ""


def evaluate_code(code: str) -> dict:
    """Test the CSV parser."""
    test_cases = [
        (None, [], "None input"),
        ("", [], "Empty string"),
        ("a,b,c", ["a", "b", "c"], "Simple"),
        ("a, b , c", ["a", "b", "c"], "Whitespace"),
        ("a,,b", ["a", "", "b"], "Empty field"),
        ("  a  ,  b  ", ["a", "b"], "More whitespace"),
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

        if not hasattr(module, 'parse_csv_line'):
            return {"error": "Function not found", "passed": 0, "total": len(test_cases), "failed_cases": []}

        func = module.parse_csv_line
        passed = 0
        failed = []

        for input_val, expected, desc in test_cases:
            try:
                result = func(input_val)
                if result == expected:
                    passed += 1
                else:
                    failed.append(f"{desc}: got {result!r}")
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
    console.print("\n[bold cyan]Experiment: Worker Reflection (CSV Parser)[/bold cyan]\n")

    base_dir = Path("/tmp/ralph_exp_csv")
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
        console.print(f"    {eval_a.get('passed', 0)}/{eval_a.get('total', 0)} passed")
        for fc in eval_a.get("failed_cases", [])[:2]:
            console.print(f"    [red]x {fc}[/red]")

        console.print("  [yellow]Reflection...[/yellow]")
        project_b = setup_test_project(base_dir, f"reflection_{run_idx}")
        code_b = await run_worker(project_b, WORKER_PROMPT_REFLECTION, TASK_PROMPT)
        eval_b = evaluate_code(code_b)
        results["reflection"].append(eval_b)
        console.print(f"    {eval_b.get('passed', 0)}/{eval_b.get('total', 0)} passed")
        for fc in eval_b.get("failed_cases", [])[:2]:
            console.print(f"    [red]x {fc}[/red]")

        console.print()

    # Summary
    table = Table(title="Results")
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

    console.print(f"\n  Diff: {reflection_avg - baseline_avg:+.0f}%")


if __name__ == "__main__":
    asyncio.run(run_experiment())
