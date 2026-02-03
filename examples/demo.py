#!/usr/bin/env python3
"""Demo script showing how to use Ralph SDK programmatically."""

import asyncio
from ralph_sdk import run_ralph
from ralph_sdk.orchestrator import run_from_prd_file
from ralph_sdk.prd_generator import load_prd


async def demo_full_workflow():
    """Run the complete workflow from a feature request."""
    success = await run_ralph(
        initial_prompt="Add a simple todo list with add, delete, and mark complete",
        cwd="./demo-project",
        max_iterations=10,
        skip_clarify=False,  # Set True to skip Q&A
    )
    print(f"Completed: {success}")


async def demo_from_prd():
    """Run from an existing PRD file."""
    success = await run_from_prd_file(
        prd_path="./prd.json",
        max_iterations=10,
    )
    print(f"Completed: {success}")


async def demo_custom_flow():
    """Custom flow with more control."""
    from ralph_sdk.clarifier import clarify_requirements
    from ralph_sdk.prd_generator import generate_prd, save_prd
    from ralph_sdk.executor import execute_story
    from ralph_sdk.progress import ProgressTracker
    from ralph_sdk.models import ExecutionContext

    # Step 1: Clarify requirements
    requirements = await clarify_requirements(
        "Build a simple REST API for managing books"
    )
    print(f"Project: {requirements.project_name}")
    print(f"Description: {requirements.final_description}")

    # Step 2: Generate PRD
    prd = await generate_prd(requirements, cwd=".")
    save_prd(prd, "prd.json")
    print(f"Generated {len(prd.user_stories)} stories")

    # Step 3: Execute stories with custom logic
    context = ExecutionContext(
        cwd=".",
        branch_name=prd.branch_name,
    )
    progress = ProgressTracker("progress.txt")

    for story in prd.get_pending_stories():
        print(f"\nExecuting: {story.id} - {story.title}")

        result = await execute_story(
            story=story,
            context=context,
            progress_context=progress.get_context(),
        )

        if result.success:
            prd.mark_complete(story.id)
            progress.append_log(story.id, result)
            progress.save()
            save_prd(prd, "prd.json")
            print(f"✓ {story.id} complete")
        else:
            print(f"✗ {story.id} failed: {result.error}")
            # Custom retry logic could go here
            break

    print(f"\nFinal progress: {prd.progress_summary()}")


if __name__ == "__main__":
    # Run whichever demo you want
    asyncio.run(demo_full_workflow())
    # asyncio.run(demo_from_prd())
    # asyncio.run(demo_custom_flow())
