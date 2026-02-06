"""
Main orchestrator: runs the task pool loop.

Flow:
1. Clarify -> goal.md
2. Initialize -> pool.md + tasks/
3. Loop: Planner -> Worker -> (Reviewer) -> update files
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from claude_code_sdk import AssistantMessage, ClaudeCodeOptions, query
from rich.console import Console
from rich.prompt import Prompt

from .clarifier import clarify_requirements, clarify_requirements_v2, quick_clarify
from .evaluator import evaluate
from .logger import SessionLogger
from .metrics import (
    AutomationLevel,
    TaskCategory,
    get_default_metrics,
    parse_metrics_from_goal,
)
from .notification import (
    notify_checkpoint,
    notify_complete,
    notify_decision,
    notify_info,
    notify_pivot,
    notify_progress,
    notify_warning,
)
from .planner import Action, plan
from .pool import (
    Checkpoint,
    add_checkpoint,
    append_failure_assumptions,
    append_to_progress_log,
    create_checkpoint,
    ensure_task_files_exist,
    generate_feedback_template,
    get_pending_checkpoints,
    goal_exists,
    init_ralph_dir,
    is_waiting_for_user,
    mark_pending_test,
    parse_feedback,
    pool_exists,
    read_checkpoint_manifest,
    read_eval_config_from_goal,
    read_goal,
    read_pool,
    read_task,
    save_checkpoint_manifest,
    write_pool,
    write_task,
)
from .prompts import INITIALIZER_SYSTEM_PROMPT
from .reviewer import ReviewResult, Verdict, review
from .worker import work

console = Console()

# Default target score for task completion
# Tasks scoring below this will continue iterating
DEFAULT_TARGET_SCORE = 95


def parse_target_score_from_goal(cwd: str = ".") -> int:
    """
    Parse target_score from goal.md if specified.

    Looks for patterns like:
    - target_score: 90
    - **Target Score**: 95

    Returns DEFAULT_TARGET_SCORE if not found.
    """
    import re
    goal_content = read_goal(cwd)
    if not goal_content:
        return DEFAULT_TARGET_SCORE

    # Look for target_score patterns
    patterns = [
        r'target_score:\s*(\d+)',
        r'\*\*Target Score\*\*:\s*(\d+)',
        r'target score:\s*(\d+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, goal_content, re.IGNORECASE)
        if match:
            return int(match.group(1))

    return DEFAULT_TARGET_SCORE


def enter_waiting_state(
    checkpoints: list[Checkpoint] | list[dict],
    instructions: str,
    cwd: str = ".",
) -> None:
    """
    Enter waiting state for user testing.

    Creates feedback.md template and checkpoint manifest.

    Args:
        checkpoints: List of Checkpoint objects or dicts
        instructions: Testing instructions for user
    """
    # Generate feedback template
    generate_feedback_template(checkpoints, instructions, cwd)

    # Save checkpoint manifest
    save_checkpoint_manifest(
        checkpoints=checkpoints,
        status="waiting_for_user",
        instructions=instructions,
        cwd=cwd,
    )

    append_to_progress_log(
        f"WAITING - Generated {len(checkpoints)} checkpoints for user testing",
        cwd
    )

    console.print("\n[bold yellow]═══ WAITING FOR USER TESTING ═══[/bold yellow]\n")
    console.print(f"生成了 {len(checkpoints)} 个 checkpoint 待测试:\n")

    for cp in checkpoints:
        if isinstance(cp, Checkpoint):
            cp_id = cp.id
            proxy_overall = cp.proxy_overall
            proxy_scores = cp.proxy_scores
        else:
            cp_id = cp.get("id", "unknown")
            proxy_overall = cp.get("proxy_overall", 0)
            proxy_scores = cp.get("proxy_scores", {})

        proxy_info = ""
        if proxy_overall > 0:
            proxy_info = f"  [dim](代理总分: {proxy_overall:.0f})[/dim]"
        elif proxy_scores:
            if isinstance(proxy_scores, list):
                scores = ", ".join(f"{ps.metric_name}={ps.score:.0f}" if hasattr(ps, 'metric_name') else f"{ps.get('metric_name', '?')}={ps.get('score', 0):.0f}" for ps in proxy_scores)
            else:
                scores = ", ".join(f"{k}={v}" for k, v in proxy_scores.items())
            proxy_info = f"  [dim](代理分数: {scores})[/dim]"

        console.print(f"  - {cp_id}{proxy_info}")

    console.print("\n请编辑 [cyan].ralph/feedback.md[/cyan] 填写测试结果")
    console.print("完成后运行 [cyan]ralph-sdk resume[/cyan] 继续\n")


# -----------------------------------------------------------------------------
# Parallel Execution Support
# -----------------------------------------------------------------------------

# Default maximum number of concurrent workers
DEFAULT_MAX_PARALLEL = 3


@dataclass
class ParallelTaskResult:
    """Result of a single task in parallel execution."""
    task_id: str
    task_type: str
    success: bool
    error: Optional[str] = None
    confidence: Optional[str] = None  # For EXPLORE tasks


@dataclass
class ParallelExecutionResult:
    """Aggregated result of parallel task execution."""
    results: list[ParallelTaskResult]
    total_tasks: int
    successful: int
    failed: int

    @property
    def all_succeeded(self) -> bool:
        return self.failed == 0


async def _execute_single_task(
    task_id: str,
    task_type: str,
    cwd: str,
    verbose: bool,
    semaphore: asyncio.Semaphore,
) -> ParallelTaskResult:
    """
    Execute a single task with semaphore for concurrency control.

    Args:
        task_id: The task ID to execute
        task_type: Either "EXPLORE" or "IMPLEMENT"
        cwd: Working directory
        verbose: Show detailed output
        semaphore: Semaphore for limiting concurrent executions

    Returns:
        ParallelTaskResult with execution outcome
    """
    async with semaphore:
        try:
            console.print(f"[cyan]⚡ Starting parallel task {task_id} ({task_type})...[/cyan]")
            result = await work(task_id, task_type, cwd, verbose=verbose)
            return ParallelTaskResult(
                task_id=task_id,
                task_type=task_type,
                success=result.success,
                error=result.error,
                confidence=result.confidence,
            )
        except Exception as e:
            console.print(f"[red]✗ Task {task_id} failed with exception: {e}[/red]")
            return ParallelTaskResult(
                task_id=task_id,
                task_type=task_type,
                success=False,
                error=str(e),
            )


async def execute_parallel_tasks(
    task_ids: list[str],
    cwd: str = ".",
    verbose: bool = False,
    max_parallel: int = DEFAULT_MAX_PARALLEL,
) -> ParallelExecutionResult:
    """
    Execute multiple tasks in parallel.

    Uses asyncio.gather to run workers concurrently, with a semaphore
    to limit the maximum number of concurrent executions.

    Args:
        task_ids: List of task IDs to execute
        cwd: Working directory
        verbose: Show detailed output
        max_parallel: Maximum number of concurrent workers

    Returns:
        ParallelExecutionResult with aggregated results
    """
    if not task_ids:
        return ParallelExecutionResult(
            results=[],
            total_tasks=0,
            successful=0,
            failed=0,
        )

    console.print(f"\n[bold magenta]═══ Parallel Execution: {len(task_ids)} tasks (max {max_parallel} concurrent) ═══[/bold magenta]\n")

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_parallel)

    # Determine task types by reading task files
    tasks_with_types = []
    for task_id in task_ids:
        task_content = read_task(task_id, cwd)
        if "Type\nEXPLORE" in task_content or "## Type\nEXPLORE" in task_content:
            task_type = "EXPLORE"
        else:
            task_type = "IMPLEMENT"
        tasks_with_types.append((task_id, task_type))

    # Create coroutines for all tasks
    coroutines = [
        _execute_single_task(task_id, task_type, cwd, verbose, semaphore)
        for task_id, task_type in tasks_with_types
    ]

    # Run all tasks concurrently (with semaphore limiting actual concurrency)
    results = await asyncio.gather(*coroutines, return_exceptions=True)

    # Process results
    processed_results = []
    successful = 0
    failed = 0

    for result in results:
        if isinstance(result, Exception):
            # Handle unexpected exceptions from gather
            processed_results.append(ParallelTaskResult(
                task_id="unknown",
                task_type="unknown",
                success=False,
                error=str(result),
            ))
            failed += 1
        else:
            processed_results.append(result)
            if result.success:
                successful += 1
            else:
                failed += 1

    console.print(f"\n[bold magenta]═══ Parallel Execution Complete: {successful} succeeded, {failed} failed ═══[/bold magenta]\n")

    return ParallelExecutionResult(
        results=processed_results,
        total_tasks=len(task_ids),
        successful=successful,
        failed=failed,
    )


async def initialize_pool(cwd: str = ".") -> None:
    """
    Initialize the task pool from goal.md.

    Creates pool.md and initial task files.
    """
    goal = read_goal(cwd)
    if not goal:
        raise ValueError("goal.md not found. Run clarifier first.")

    console.print("\n[yellow]Initializing task pool...[/yellow]\n")

    prompt = f"""Goal:
---
{goal}
---

Analyze this goal and create the initial Task Pool.

1. First, explore the codebase to understand the current state.
2. Then create appropriate tasks (start with EXPLORE if uncertain).
3. Write the task table to .ralph/pool.md
4. Create detailed task files in .ralph/tasks/

Remember: keep initial tasks coarse-grained. It's OK to have only 2-3 tasks.
"""

    async for message in query(
        prompt=prompt,
        options=ClaudeCodeOptions(
            system_prompt=INITIALIZER_SYSTEM_PROMPT,
            allowed_tools=[
                "Read", "Write", "Glob", "Grep", "LSP",
                "WebFetch", "WebSearch",  # Research for better task planning
            ],
            permission_mode="acceptEdits",
            max_turns=20,
            cwd=cwd,
        ),
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    console.print(block.text)

    console.print("\n[green]✓ Task pool initialized[/green]")


async def run(
    goal: str,
    cwd: str = ".",
    max_iterations: int = 30,
    skip_clarify: bool = False,
    verbose: bool = False,
    max_parallel: int = DEFAULT_MAX_PARALLEL,
    clarify_mode: str = "auto",
    target_score: int = None,
    explore_mode: bool = False,
) -> bool:
    """
    Run the full task pool loop.

    Args:
        goal: The user's goal/feature request
        cwd: Working directory
        max_iterations: Maximum number of iterations
        skip_clarify: If True, skip interactive clarification
        verbose: Show detailed output
        max_parallel: Maximum number of concurrent workers for PARALLEL_EXECUTE
        clarify_mode: Clarification mode - "auto" | "ask" | "explore"
            - auto: Automatically choose based on request clarity
            - ask: Use traditional Q&A mode (Clarifier v1)
            - explore: Use explore+propose mode (Clarifier v2)
        target_score: Target quality score (0-100). Tasks scoring below this
            will continue iterating. If None, parsed from goal.md or defaults to 95.
        explore_mode: If True, don't DONE until user explicitly says stop.
            Continues exploring alternatives even after tasks complete.

    Returns:
        True if goal was completed, False if max iterations reached
    """
    init_ralph_dir(cwd)

    # Initialize session logger
    logger = SessionLogger(cwd)
    logger.log_session_start(goal)

    # Phase 1: Clarify
    if not goal_exists(cwd):
        console.print("\n[bold]Phase 1: Clarifying requirements[/bold]\n")
        if skip_clarify:
            await quick_clarify(goal, cwd)
        elif clarify_mode == "explore":
            await clarify_requirements_v2(goal, cwd, mode="explore")
        elif clarify_mode == "ask":
            await clarify_requirements(goal, cwd)
        else:  # auto mode
            await clarify_requirements_v2(goal, cwd, mode="auto")
    else:
        console.print("\n[dim]Goal already exists, skipping clarification[/dim]")

    # Phase 2: Initialize pool
    if not pool_exists(cwd):
        console.print("\n[bold]Phase 2: Initializing task pool[/bold]\n")
        await initialize_pool(cwd)
    else:
        console.print("\n[dim]Pool already exists, skipping initialization[/dim]")

    # Determine target score
    if target_score is None:
        target_score = parse_target_score_from_goal(cwd)
    console.print(f"[dim]Target score: {target_score}/100[/dim]")

    # Show explore mode status
    if explore_mode:
        console.print("[bold magenta]🔍 Explore mode: ON[/bold magenta]")
        console.print("[dim]Will continue exploring until user says stop in feedback.md[/dim]")

    # Phase 3: Iteration loop
    console.print("\n[bold]Phase 3: Executing tasks[/bold]\n")

    for i in range(1, max_iterations + 1):
        console.print(f"\n[bold cyan]═══ Iteration {i}/{max_iterations} ═══[/bold cyan]\n")
        logger.log_iteration_start(i, max_iterations)

        # Planner decides next action
        console.print("[yellow]Planner deciding...[/yellow]")
        decision = await plan(cwd, verbose=verbose, explore_mode=explore_mode)

        console.print(f"[bold]Action:[/bold] {decision.action.value}")
        if decision.target:
            console.print(f"[bold]Target:[/bold] {decision.target}")
        if decision.task_ids:
            console.print(f"[bold]Tasks:[/bold] {', '.join(decision.task_ids)}")
        console.print(f"[bold]Reason:[/bold] {decision.reason}")

        # Log planner decision
        logger.log_planner_decision(
            action=decision.action.value,
            target=decision.target,
            reason=decision.reason,
        )

        # Handle DONE
        if decision.action == Action.DONE:
            # Check for unprocessed PIVOT recommendations (P0: block DONE if pivot pending)
            pool_content = read_pool(cwd)
            if "[PIVOT_RECOMMENDED]" in pool_content:
                console.print("[yellow]⚠️ 有 PIVOT 建议未处理，继续迭代[/yellow]")
                console.print("[dim]如需强制结束，请在 feedback.md 中写 next_step: stop[/dim]")
                logger.log_iteration_end(i, "DONE_BLOCKED", reason="Unprocessed pivot recommendation")
                append_to_progress_log(f"DONE_BLOCKED - Unprocessed pivot recommendation", cwd)
                continue

            # Check explore mode - need user confirmation to stop
            if explore_mode:
                feedback = parse_feedback(cwd)
                user_wants_stop = feedback and feedback.get("next_step") == "stop"
                if not user_wants_stop:
                    console.print("[yellow]⚠️ 探索模式：不允许 DONE，创建新探索任务[/yellow]")
                    console.print("[dim]如需停止，请在 feedback.md 中写 next_step: stop[/dim]")
                    logger.log_iteration_end(i, "DONE_BLOCKED", reason="Explore mode - user stop not confirmed")
                    append_to_progress_log(f"DONE_BLOCKED - Explore mode active, waiting for user stop signal", cwd)
                    continue

            console.print("\n[bold green]✓ Goal completed![/bold green]")
            append_to_progress_log(f"DONE - Goal completed after {i} iterations", cwd)
            logger.log_session_end(success=True, reason=f"Completed after {i} iterations")
            return True

        # Handle CREATE / DECOMPOSE / DELETE
        # These are handled by the planner itself (it writes to files)
        if decision.action in (Action.CREATE, Action.DECOMPOSE, Action.DELETE):
            append_to_progress_log(
                f"{decision.action.value.upper()} - {decision.reason}",
                cwd
            )
            # Ensure task files exist for any new tasks added to pool.md
            if decision.action in (Action.CREATE, Action.DECOMPOSE):
                created = ensure_task_files_exist(cwd)
                if created:
                    console.print(f"[dim]Auto-created missing task files: {', '.join(created)}[/dim]")
                    for task_id in created:
                        logger.log_task_created(task_id, "IMPLEMENT", "Auto-created")
            logger.log_iteration_end(i, decision.action.value)
            continue

        # Handle EXPLORE
        if decision.action == Action.EXPLORE:
            if not decision.target:
                console.print("[red]Error: EXPLORE requires a target task[/red]")
                logger.log_error("EXPLORE requires a target task")
                continue

            console.print(f"\n[yellow]Exploring {decision.target}...[/yellow]\n")
            result = await work(decision.target, "EXPLORE", cwd, verbose=verbose)

            # Log worker result
            logger.log_worker_complete(
                task_id=decision.target,
                task_type="EXPLORE",
                success=result.success,
                error=result.error,
            )

            if result.success:
                console.print(f"[green]✓ Exploration complete[/green]")
                if result.confidence:
                    console.print(f"[dim]Confidence: {result.confidence}[/dim]")
            else:
                console.print(f"[red]✗ Exploration failed: {result.error}[/red]")

            append_to_progress_log(
                f"EXPLORE {decision.target} - {'success' if result.success else 'failed'}"
                + (f" (confidence: {result.confidence})" if result.confidence else ""),
                cwd
            )
            logger.log_iteration_end(i, decision.action.value, success=result.success)
            continue

        # Handle PARALLEL_EXECUTE
        if decision.action == Action.PARALLEL_EXECUTE:
            if not decision.task_ids:
                console.print("[red]Error: PARALLEL_EXECUTE requires TASK_IDS[/red]")
                logger.log_error("PARALLEL_EXECUTE requires TASK_IDS")
                continue

            if len(decision.task_ids) < 2:
                console.print("[yellow]Warning: Only one task provided, using regular EXECUTE[/yellow]")
                # Fall through to single task execution
                decision.target = decision.task_ids[0]
                decision.action = Action.EXECUTE
                # Don't continue - let it fall through to EXECUTE handler

            else:
                # Execute tasks in parallel
                parallel_result = await execute_parallel_tasks(
                    task_ids=decision.task_ids,
                    cwd=cwd,
                    verbose=verbose,
                    max_parallel=max_parallel,
                )

                # Log each task result
                for task_result in parallel_result.results:
                    logger.log_worker_complete(
                        task_id=task_result.task_id,
                        task_type=task_result.task_type,
                        success=task_result.success,
                        error=task_result.error,
                    )

                # Review successful IMPLEMENT tasks
                for task_result in parallel_result.results:
                    if task_result.success and task_result.task_type == "IMPLEMENT":
                        console.print(f"\n[dim]Reviewing {task_result.task_id}...[/dim]")
                        try:
                            review_result = await review(task_result.task_id, cwd, verbose=verbose)
                            console.print(f"[dim]{task_result.task_id} review: {review_result.verdict.value}[/dim]")
                            logger.log_reviewer_verdict(
                                task_id=task_result.task_id,
                                verdict=review_result.verdict.value,
                                reason=review_result.reason,
                            )
                            # Note: For simplicity, we log but don't handle RETRY/FAILED here
                            # The Planner will see the review results in task files and decide next steps
                        except Exception as e:
                            console.print(f"[red]Review failed for {task_result.task_id}: {e}[/red]")
                            logger.log_error(f"Review failed for {task_result.task_id}: {e}")

                # Log aggregate result
                status_parts = []
                for task_result in parallel_result.results:
                    status = "✓" if task_result.success else "✗"
                    status_parts.append(f"{task_result.task_id}:{status}")

                append_to_progress_log(
                    f"PARALLEL_EXECUTE [{', '.join(decision.task_ids)}] - "
                    f"{parallel_result.successful}/{parallel_result.total_tasks} succeeded "
                    f"({'; '.join(status_parts)})",
                    cwd
                )

                # Display summary
                if parallel_result.all_succeeded:
                    console.print(f"[green]✓ All {parallel_result.total_tasks} parallel tasks completed successfully[/green]")
                else:
                    console.print(f"[yellow]⚠ Parallel execution: {parallel_result.successful} succeeded, {parallel_result.failed} failed[/yellow]")
                    for task_result in parallel_result.results:
                        if not task_result.success:
                            console.print(f"  [red]✗ {task_result.task_id}: {task_result.error}[/red]")

                logger.log_iteration_end(i, decision.action.value, success=parallel_result.all_succeeded)
                continue

        # Handle EXECUTE
        if decision.action == Action.EXECUTE:
            if not decision.target:
                console.print("[red]Error: EXECUTE requires a target task[/red]")
                logger.log_error("EXECUTE requires a target task")
                continue

            console.print(f"\n[yellow]Executing {decision.target}...[/yellow]\n")
            result = await work(decision.target, "IMPLEMENT", cwd, verbose=verbose)

            # Log worker result
            logger.log_worker_complete(
                task_id=decision.target,
                task_type="IMPLEMENT",
                success=result.success,
                error=result.error,
            )

            if result.success:
                console.print(f"[green]✓ Execution complete, reviewing...[/green]\n")

                # Review the result with exception handling
                try:
                    review_result = await review(decision.target, cwd, verbose=verbose)
                except Exception as e:
                    console.print(f"[red]Review failed: {e}[/red]")
                    logger.log_error(f"Review failed for {decision.target}: {e}")
                    review_result = ReviewResult(
                        verdict=Verdict.RETRY,
                        reason=f"Review failed with error: {e}",
                    )

                console.print(f"[bold]Review verdict:[/bold] {review_result.verdict.value}")
                console.print(f"[dim]{review_result.reason}[/dim]")

                # Log reviewer verdict
                logger.log_reviewer_verdict(
                    task_id=decision.target,
                    verdict=review_result.verdict.value,
                    reason=review_result.reason,
                )

                if review_result.verdict == Verdict.PASSED:
                    from .evaluator import AutomationLevel as EvalAutomationLevel
                    from .evaluator import Metric
                    from .evaluator import MetricType as EvalMetricType

                    # Run quality evaluation
                    console.print(f"\n[yellow]Evaluating quality...[/yellow]")

                    # Get metrics from goal.md or use defaults
                    goal_content = read_goal(cwd)
                    metrics_config = parse_metrics_from_goal(goal_content)
                    if not metrics_config:
                        metrics_config = get_default_metrics(TaskCategory.GENERAL)

                    # Convert MetricDefinition to evaluator Metric format
                    eval_metrics = [
                        Metric(
                            name=m.name,
                            type=EvalMetricType(m.type.value),
                            description=m.description,
                            target=m.target,
                            automation=EvalAutomationLevel(m.automation.value),
                            proxy_metric=m.proxy_metric,
                        )
                        for m in metrics_config.all_metrics()
                    ]

                    eval_result = await evaluate(
                        decision.target,
                        cwd=cwd,
                        metrics=eval_metrics if eval_metrics else None,
                        verbose=verbose,
                    )

                    # Log evaluation
                    logger.log_evaluation(
                        task_id=decision.target,
                        passed=eval_result.overall_passed,
                        score=eval_result.overall_score,
                        issues=eval_result.issues,
                    )

                    # P0: If Evaluator suggests pivot, write to pool.md for Planner to see
                    if eval_result.should_pivot:
                        from .pool import append_to_findings
                        append_to_findings(
                            f"**[PIVOT_RECOMMENDED]** {decision.target}: {eval_result.pivot_reason}",
                            cwd
                        )
                        console.print(f"[yellow]⚠️ Evaluator 建议转向: {eval_result.pivot_reason}[/yellow]")

                    # Check if user testing is needed
                    if eval_result.needs_user_testing:
                        eval_config = read_eval_config_from_goal(cwd)
                        pending_metrics = eval_result.get_pending_manual_metrics()

                        # Create checkpoint with proper structure
                        checkpoint = create_checkpoint(
                            task_id=decision.target,
                            path=f"checkpoints/{decision.target}",
                            artifact_type="code",
                            description=f"Implementation of {decision.target}",
                            cwd=cwd,
                        )

                        # Add proxy scores from evaluation
                        for mr in eval_result.metrics:
                            if mr.proxy_score is not None:
                                checkpoint.add_proxy_score(
                                    metric_name=mr.metric.name,
                                    score=mr.proxy_score,
                                    target=mr.metric.target,
                                    notes=mr.proxy_notes,
                                )
                            elif not mr.pending_manual and mr.score is not None:
                                # Also add auto-evaluated scores as reference
                                checkpoint.add_proxy_score(
                                    metric_name=mr.metric.name,
                                    score=mr.score,
                                    target=mr.metric.target,
                                    notes="Auto-evaluated",
                                )

                        # Generate test instructions
                        instructions = "请测试以下指标:\n"
                        for mr in pending_metrics:
                            instructions += f"- {mr.metric.name}: {mr.metric.description}\n"
                            if mr.metric.target:
                                instructions += f"  目标: {mr.metric.target}\n"
                            if mr.proxy_score is not None:
                                instructions += f"  代理分数: {mr.proxy_score:.0f}\n"

                        # Decide whether to pause based on batch preference
                        batch_pref = eval_config.get("batch_preference", "")

                        # Always add checkpoint first
                        add_checkpoint(checkpoint, cwd)

                        if "一个一个" in batch_pref or not batch_pref:
                            # Mark as pending test but DON'T pause
                            # Let the Planner decide to HEDGE or continue
                            mark_pending_test(decision.target, cwd)

                            # Use notification system (non-blocking)
                            notify_checkpoint(
                                checkpoint_id=checkpoint.id,
                                task_id=decision.target,
                                proxy_score=checkpoint.proxy_overall,
                                description=f"{len(pending_metrics)} 项指标待用户测试",
                            )

                            # Don't return False - let the loop continue
                            # The Planner will decide whether to HEDGE or do other work
                        else:
                            # For batch mode, use notification
                            notify_checkpoint(
                                checkpoint_id=checkpoint.id,
                                task_id=decision.target,
                                proxy_score=checkpoint.proxy_overall,
                                description=f"{len(pending_metrics)} 项指标待批量测试",
                            )

                        # Check if we've accumulated enough for batch testing
                        pending = get_pending_checkpoints(cwd)
                        batch_size = int(eval_config.get("batch_size", 3) or 3)
                        if len(pending) >= batch_size:
                            console.print(f"\n[yellow]Batch size reached ({len(pending)} checkpoints)[/yellow]")
                            enter_waiting_state(
                                checkpoints=pending,
                                instructions=instructions,
                                cwd=cwd,
                            )
                            logger.log_session_end(success=False, reason=f"Waiting for batch testing ({len(pending)} checkpoints)")
                            return False

                    if eval_result.overall_passed and eval_result.overall_score >= target_score:
                        console.print(f"[green]✓ Task {decision.target} completed (score: {eval_result.overall_score:.0f}/{target_score})[/green]")
                        append_to_progress_log(
                            f"EXECUTE {decision.target} - PASSED (score: {eval_result.overall_score:.0f}/{target_score})",
                            cwd
                        )
                    else:
                        # Task passed functionally but quality needs improvement
                        console.print(f"[yellow]⚠ Task {decision.target} needs improvement (score: {eval_result.overall_score:.0f}/{target_score})[/yellow]")

                        # Record issues and suggestions in task file for Planner to see
                        task_content = read_task(decision.target, cwd)
                        if task_content:
                            now = datetime.now().strftime("%Y-%m-%d %H:%M")
                            eval_note = f"\n\n## Quality Evaluation ({now})\n"
                            eval_note += f"**Score**: {eval_result.overall_score:.0f}/100 (target: {target_score})\n"
                            eval_note += f"**Status**: {'PASSED' if eval_result.overall_score >= target_score else 'NEEDS IMPROVEMENT'}\n"
                            if eval_result.issues:
                                eval_note += "\n**Issues**:\n"
                                for issue in eval_result.issues:
                                    eval_note += f"- {issue}\n"
                            if eval_result.suggestions:
                                eval_note += "\n**Suggestions**:\n"
                                for suggestion in eval_result.suggestions:
                                    eval_note += f"- {suggestion}\n"
                            write_task(decision.target, task_content + eval_note, cwd)

                        append_to_progress_log(
                            f"EXECUTE {decision.target} - NEEDS IMPROVEMENT (score: {eval_result.overall_score:.0f}/{target_score})",
                            cwd
                        )
                elif review_result.verdict == Verdict.RETRY:
                    console.print(f"[yellow]↻ Task {decision.target} needs retry[/yellow]")

                    # Write review feedback to task file for next attempt
                    task_content = read_task(decision.target, cwd)
                    if task_content:
                        now = datetime.now().strftime("%Y-%m-%d %H:%M")
                        review_note = f"\n\n## Review Feedback ({now})\n"
                        review_note += f"**Verdict**: RETRY\n"
                        review_note += f"**Reason**: {review_result.reason}\n"
                        if review_result.suggestions:
                            review_note += f"**Suggestions**: {review_result.suggestions}\n"
                        write_task(decision.target, task_content + review_note, cwd)

                    append_to_progress_log(
                        f"EXECUTE {decision.target} - RETRY: {review_result.reason}",
                        cwd
                    )
                else:  # FAILED
                    console.print(f"[red]✗ Task {decision.target} failed[/red]")

                    # Write review feedback to task file for Planner to see
                    task_content = read_task(decision.target, cwd)
                    if task_content:
                        now = datetime.now().strftime("%Y-%m-%d %H:%M")
                        review_note = f"\n\n## Review Feedback ({now})\n"
                        review_note += f"**Verdict**: FAILED\n"
                        review_note += f"**Reason**: {review_result.reason}\n"
                        if review_result.suggestions:
                            review_note += f"**Suggestions**: {review_result.suggestions}\n"
                        write_task(decision.target, task_content + review_note, cwd)

                    append_to_progress_log(
                        f"EXECUTE {decision.target} - FAILED: {review_result.reason}",
                        cwd
                    )
            else:
                console.print(f"[red]✗ Execution failed: {result.error}[/red]")
                append_to_progress_log(
                    f"EXECUTE {decision.target} - ERROR: {result.error}",
                    cwd
                )
            logger.log_iteration_end(i, decision.action.value, success=result.success)
            continue

        # Handle MODIFY
        if decision.action == Action.MODIFY:
            if not decision.target:
                console.print("[red]Error: MODIFY requires a target task[/red]")
                logger.log_error("MODIFY requires a target task")
                continue

            console.print(f"\n[yellow]Modifying {decision.target}...[/yellow]")
            console.print(f"[dim]Modification: {decision.modification}[/dim]")

            # Read existing task content and append modification note
            task_content = read_task(decision.target, cwd)
            if task_content:
                now = datetime.now().strftime("%Y-%m-%d %H:%M")
                modification_note = f"\n\n## Modification ({now})\n{decision.modification}\n"
                write_task(decision.target, task_content + modification_note, cwd)
                console.print(f"[green]✓ Task {decision.target} modified[/green]")
            else:
                console.print(f"[red]Error: Task {decision.target} not found[/red]")
                logger.log_error(f"Task {decision.target} not found for MODIFY")

            logger.log_task_modified(decision.target, decision.modification)
            append_to_progress_log(
                f"MODIFY {decision.target} - {decision.modification[:100]}",
                cwd
            )
            logger.log_iteration_end(i, decision.action.value)
            continue

        # Handle SKIP
        if decision.action == Action.SKIP:
            console.print(f"\n[yellow]⏭ Skipping {decision.target or 'current task'}[/yellow]")
            console.print(f"[dim]Reason: {decision.reason}[/dim]")

            logger.log_task_skipped(decision.target, decision.reason)
            append_to_progress_log(
                f"SKIP {decision.target or 'task'} - {decision.reason}",
                cwd
            )
            logger.log_iteration_end(i, decision.action.value)
            continue

        # Handle ASK
        if decision.action == Action.ASK:
            console.print(f"\n[bold cyan]❓ Planner needs your input:[/bold cyan]")
            console.print(f"\n{decision.question}\n")

            # Get user input
            user_answer = Prompt.ask("[bold]Your answer[/bold]")

            # Log the Q&A
            logger.log_user_question(decision.question, user_answer)
            append_to_progress_log(
                f"ASK - Q: {decision.question[:100]}... A: {user_answer[:100]}",
                cwd
            )

            # Append Q&A to pool.md Findings section for future reference
            pool_content = read_pool(cwd)
            if "## Findings" in pool_content:
                qa_note = f"\n\n**User Decision**: {decision.question}\n**Answer**: {user_answer}\n"
                pool_content = pool_content.replace(
                    "## Findings",
                    f"## Findings{qa_note}"
                )
                write_pool(pool_content, cwd)

            console.print(f"[green]✓ Answer recorded[/green]")
            logger.log_iteration_end(i, decision.action.value)
            continue

        # Handle HEDGE / PIVOT_WAIT - Pessimistic Preparation
        if decision.action in (Action.HEDGE, Action.PIVOT_WAIT):
            console.print(f"\n[yellow]🛡️ HEDGE: 为 {decision.target} 探索替代方案[/yellow]")
            console.print(f"[dim]Reason: {decision.reason}[/dim]")

            # Send notification (non-blocking)
            notify_pivot(
                task_id=decision.target,
                reason=decision.reason,
                from_approach="当前实现",
                to_approach="替代方案探索中",
                trigger="wait",
            )

            # Record failure assumptions in pool.md
            if decision.failure_assumptions:
                append_failure_assumptions(
                    decision.target,
                    decision.failure_assumptions,
                    cwd
                )
                console.print(f"[dim]记录了失败假设到 pool.md[/dim]")

            # Ensure task files exist for new exploration tasks
            if decision.new_tasks:
                created = ensure_task_files_exist(cwd)
                if created:
                    console.print(f"[dim]创建了探索任务: {', '.join(created)}[/dim]")
                    for task_id in created:
                        logger.log_task_created(task_id, "EXPLORE", "Hedge alternative")

            append_to_progress_log(
                f"HEDGE {decision.target} - 探索替代方案: {decision.reason[:100]}",
                cwd
            )

            # Clear the PIVOT_RECOMMENDED marker for this task
            from .pool import clear_pivot_recommendation
            clear_pivot_recommendation(decision.target, cwd)

            logger.log_iteration_end(i, decision.action.value)
            continue

        # Handle PIVOT_RESEARCH - Research confirmed direction not viable
        if decision.action == Action.PIVOT_RESEARCH:
            console.print(f"\n[magenta]🔄 PIVOT_RESEARCH: {decision.target} 方向不可行[/magenta]")
            console.print(f"[dim]当前方案: {decision.current_approach}[/dim]")
            console.print(f"[dim]阻断原因: {decision.blocker}[/dim]")
            console.print(f"[dim]新方向: {decision.new_direction}[/dim]")

            # Send notification (non-blocking)
            notify_pivot(
                task_id=decision.target,
                reason=decision.blocker,
                from_approach=decision.current_approach,
                to_approach=decision.new_direction,
                trigger="research",
            )

            # Log the pivot decision
            notify_decision(
                decision=f"放弃 {decision.current_approach}，转向 {decision.new_direction}",
                reason=decision.blocker,
                task_id=decision.target,
            )

            # Create new tasks for the new direction
            if decision.new_tasks:
                created = ensure_task_files_exist(cwd)
                if created:
                    console.print(f"[dim]创建了新方向任务: {', '.join(created)}[/dim]")
                    for task_id in created:
                        logger.log_task_created(task_id, "EXPLORE", "Pivot to new direction")

            append_to_progress_log(
                f"PIVOT_RESEARCH {decision.target} - 放弃 [{decision.current_approach}] 因为 [{decision.blocker}]，转向 [{decision.new_direction}]",
                cwd
            )

            # Clear the PIVOT_RECOMMENDED marker for this task
            from .pool import clear_pivot_recommendation
            clear_pivot_recommendation(decision.target, cwd)

            logger.log_iteration_end(i, decision.action.value)
            continue

        # Handle PIVOT_ITERATION - Multiple attempts failed
        if decision.action == Action.PIVOT_ITERATION:
            console.print(f"\n[magenta]🔄 PIVOT_ITERATION: {decision.target} 多次尝试未达标[/magenta]")
            console.print(f"[dim]尝试次数: {decision.attempt_count}[/dim]")
            console.print(f"[dim]最高分数: {decision.best_score}[/dim]")
            console.print(f"[dim]失败模式: {decision.failure_pattern}[/dim]")
            console.print(f"[dim]新方案: {decision.new_approach}[/dim]")

            # Send notification (non-blocking)
            notify_pivot(
                task_id=decision.target,
                reason=f"尝试 {decision.attempt_count} 次，最高分 {decision.best_score}，{decision.failure_pattern}",
                from_approach=f"现有实现 (尝试 {decision.attempt_count} 次)",
                to_approach=decision.new_approach,
                trigger="iteration",
            )

            # Log the pivot decision
            notify_decision(
                decision=f"停止优化现有实现，采用新方案: {decision.new_approach}",
                reason=decision.failure_pattern,
                task_id=decision.target,
            )

            # Create new tasks for the new approach
            if decision.new_tasks:
                created = ensure_task_files_exist(cwd)
                if created:
                    console.print(f"[dim]创建了新方案任务: {', '.join(created)}[/dim]")
                    for task_id in created:
                        logger.log_task_created(task_id, "IMPLEMENT", "Pivot to new approach")

            append_to_progress_log(
                f"PIVOT_ITERATION {decision.target} - 尝试 {decision.attempt_count} 次，最高分 {decision.best_score}，转向 [{decision.new_approach}]",
                cwd
            )

            # Clear the PIVOT_RECOMMENDED marker for this task
            from .pool import clear_pivot_recommendation
            clear_pivot_recommendation(decision.target, cwd)

            logger.log_iteration_end(i, decision.action.value)
            continue

    console.print(f"\n[yellow]Reached max iterations ({max_iterations})[/yellow]")
    logger.log_session_end(success=False, reason=f"Reached max iterations ({max_iterations})")
    return False


async def resume(
    cwd: str = ".",
    max_iterations: int = 30,
    verbose: bool = False,
    max_parallel: int = DEFAULT_MAX_PARALLEL,
    target_score: int = None,
    explore_mode: bool = False,
) -> bool:
    """
    Resume an existing task pool session.

    If the system is waiting for user feedback, it will process
    the feedback and incorporate it into the next iteration.

    Args:
        cwd: Working directory containing .ralph/
        max_iterations: Maximum number of iterations
        verbose: If True, show detailed tool calls and thinking
        max_parallel: Maximum number of concurrent workers for PARALLEL_EXECUTE
        target_score: Target quality score. If None, parsed from goal.md or defaults to 95.
        explore_mode: If True, don't DONE until user explicitly says stop.

    Returns:
        True if goal was completed, False if max iterations reached
    """
    if not goal_exists(cwd):
        raise ValueError("No existing session found (goal.md missing)")
    if not pool_exists(cwd):
        raise ValueError("No existing session found (pool.md missing)")

    console.print("\n[bold]Resuming existing session[/bold]\n")

    # Check if we were waiting for user feedback
    if is_waiting_for_user(cwd):
        feedback = parse_feedback(cwd)
        manifest = read_checkpoint_manifest(cwd)

        if feedback and manifest:
            # Process feedback - append to pool.md findings
            console.print("[yellow]Processing user feedback...[/yellow]\n")

            feedback_summary = "**User Testing Feedback**:\n"

            # Find best checkpoint
            best_cp = None
            best_score = 0
            for cp_id, results in feedback.get("checkpoint_results", {}).items():
                score = int(results.get("评分 (1-5)", "0") or "0")
                feedback_summary += f"- {cp_id}: 评分 {score}/5"
                if results.get("成功率"):
                    feedback_summary += f", 成功率 {results['成功率']}"
                feedback_summary += "\n"
                if score > best_score:
                    best_score = score
                    best_cp = cp_id

            if feedback.get("overall_feedback"):
                feedback_summary += f"\n**Overall**: {feedback['overall_feedback']}\n"

            if best_cp:
                feedback_summary += f"\n**Best checkpoint**: {best_cp} (score: {best_score}/5)\n"

            feedback_summary += f"\n**Next step**: {feedback.get('next_step', 'continue')}\n"

            # Append to pool findings
            pool_content = read_pool(cwd)
            if "## Findings" in pool_content:
                pool_content = pool_content.replace(
                    "## Findings",
                    f"## Findings\n\n{feedback_summary}"
                )
                write_pool(pool_content, cwd)

            # Update manifest status
            save_checkpoint_manifest(
                checkpoints=manifest.get("checkpoints", []),
                status="feedback_processed",
                instructions=manifest.get("instructions", ""),
                cwd=cwd,
            )

            append_to_progress_log(
                f"Processed user feedback. Best checkpoint: {best_cp}. Next: {feedback.get('next_step', 'continue')}",
                cwd
            )

            # Handle next step
            if feedback.get("next_step") == "finish":
                console.print("[green]✓ User indicated task is complete[/green]")
                return True

    # Just run the iteration loop (goal already exists)
    return await run(
        goal="",  # Not used since goal.md exists
        cwd=cwd,
        max_iterations=max_iterations,
        skip_clarify=True,
        verbose=verbose,
        max_parallel=max_parallel,
        target_score=target_score,
        explore_mode=explore_mode,
    )
