"""Data models for Ralph SDK."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class UserStory(BaseModel):
    """A single user story in the PRD."""

    id: str = Field(description="Story ID, e.g. US-001")
    title: str = Field(description="Short descriptive title")
    description: str = Field(description="As a [user], I want [feature] so that [benefit]")
    acceptance_criteria: list[str] = Field(description="Verifiable checklist")
    priority: int = Field(description="Execution order, lower = higher priority")
    passes: bool = Field(default=False, description="Whether this story is complete")
    notes: str = Field(default="", description="Implementation notes")

    def to_prompt(self) -> str:
        """Convert story to a prompt for the executor."""
        criteria = "\n".join(f"- {c}" for c in self.acceptance_criteria)
        return f"""## {self.id}: {self.title}

{self.description}

### Acceptance Criteria:
{criteria}

### Notes:
{self.notes if self.notes else "None"}
"""


class PRD(BaseModel):
    """Product Requirements Document."""

    project: str = Field(description="Project name")
    branch_name: str = Field(description="Git branch name, e.g. ralph/feature-name")
    description: str = Field(description="Feature description")
    user_stories: list[UserStory] = Field(default_factory=list)

    def get_pending_stories(self) -> list[UserStory]:
        """Get stories that haven't passed yet, sorted by priority."""
        pending = [s for s in self.user_stories if not s.passes]
        return sorted(pending, key=lambda s: s.priority)

    def get_next_story(self) -> Optional[UserStory]:
        """Get the next story to work on."""
        pending = self.get_pending_stories()
        if not pending:
            return None
        return pending[0]

    def mark_complete(self, story_id: str, notes: str = "") -> None:
        """Mark a story as complete."""
        for story in self.user_stories:
            if story.id == story_id:
                story.passes = True
                if notes:
                    story.notes = notes
                break

    def is_complete(self) -> bool:
        """Check if all stories are complete."""
        return all(s.passes for s in self.user_stories)

    def progress_summary(self) -> str:
        """Get a summary of progress."""
        total = len(self.user_stories)
        done = sum(1 for s in self.user_stories if s.passes)
        return f"{done}/{total} stories complete"


class StoryResult(BaseModel):
    """Result of executing a single story."""

    story_id: str
    success: bool
    error: Optional[str] = None
    learnings: list[str] = Field(default_factory=list)
    files_changed: list[str] = Field(default_factory=list)
    commit_hash: Optional[str] = None


class ExecutionContext(BaseModel):
    """Context for story execution."""

    cwd: str = Field(description="Working directory")
    branch_name: str = Field(description="Git branch to work on")
    progress_file: str = Field(default="progress.txt")
    prd_file: str = Field(default="prd.json")
    max_retries: int = Field(default=1)


class ProgressEntry(BaseModel):
    """A single entry in the progress log."""

    timestamp: datetime = Field(default_factory=datetime.now)
    story_id: str
    summary: str
    files_changed: list[str] = Field(default_factory=list)
    learnings: list[str] = Field(default_factory=list)


class ClarifiedRequirements(BaseModel):
    """Requirements after clarification with user."""

    original_prompt: str
    clarifications: dict[str, str] = Field(default_factory=dict)
    final_description: str = ""
    project_name: str = ""
    scope: str = ""
    non_goals: list[str] = Field(default_factory=list)
