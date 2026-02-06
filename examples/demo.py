#!/usr/bin/env python3
"""Demo script showing how to use Ralph SDK programmatically."""

import asyncio

from ralph_sdk import run, resume
from ralph_sdk.clarifier import clarify_requirements


async def demo_basic():
    """Run the simplest workflow - just provide a prompt."""
    success = await run(
        initial_prompt="Add a simple todo list with add, delete, and mark complete",
        cwd="./demo-project",
        max_iterations=10,
    )
    print(f"Completed: {success}")


async def demo_with_clarification():
    """Run with explicit requirement clarification first."""
    # Step 1: Clarify requirements interactively
    requirements = await clarify_requirements(
        "Build a simple REST API for managing books"
    )
    print(f"Project: {requirements.project_name}")
    print(f"Description: {requirements.final_description}")

    # Step 2: Run with the clarified requirements
    success = await run(
        initial_prompt=requirements.final_description,
        cwd="./demo-project",
        max_iterations=10,
    )
    print(f"Completed: {success}")


async def demo_resume():
    """Resume a previous session."""
    # Resume continues from the last checkpoint
    success = await resume(
        cwd="./demo-project",
        max_iterations=10,
    )
    print(f"Resumed and completed: {success}")


if __name__ == "__main__":
    # Run whichever demo you want
    asyncio.run(demo_basic())
    # asyncio.run(demo_with_clarification())
    # asyncio.run(demo_resume())
