"""Progress tracking module."""

import os
from datetime import datetime
from ralph_sdk.models import ProgressEntry, StoryResult
from rich.console import Console

console = Console()


class ProgressTracker:
    """Tracks and persists progress across iterations."""

    def __init__(self, filepath: str = "progress.txt"):
        self.filepath = filepath
        self.patterns: list[str] = []
        self.entries: list[ProgressEntry] = []
        self._load()

    def _load(self) -> None:
        """Load existing progress from file."""
        if not os.path.exists(self.filepath):
            return

        with open(self.filepath) as f:
            content = f.read()

        # Parse patterns section
        if "## Codebase Patterns" in content:
            patterns_section = content.split("## Codebase Patterns")[1]
            if "---" in patterns_section:
                patterns_section = patterns_section.split("---")[0]
            for line in patterns_section.strip().split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    self.patterns.append(line[2:])

    def append_log(self, story_id: str, result: StoryResult) -> None:
        """Append a new progress entry."""
        entry = ProgressEntry(
            story_id=story_id,
            summary=f"{'Success' if result.success else 'Failed'}: {story_id}",
            files_changed=result.files_changed,
            learnings=result.learnings,
        )
        self.entries.append(entry)

    def add_pattern(self, pattern: str) -> None:
        """Add a reusable pattern."""
        if pattern not in self.patterns:
            self.patterns.append(pattern)

    def get_context(self) -> str:
        """Get context string for the executor."""
        lines = []

        if self.patterns:
            lines.append("## Codebase Patterns")
            for p in self.patterns:
                lines.append(f"- {p}")
            lines.append("")

        if self.entries:
            lines.append("## Recent Progress")
            for entry in self.entries[-5:]:  # Last 5 entries
                lines.append(f"- [{entry.story_id}] {entry.summary}")
                for learning in entry.learnings[:3]:  # Top 3 learnings
                    lines.append(f"  - {learning}")
            lines.append("")

        return "\n".join(lines)

    def save(self) -> None:
        """Save progress to file."""
        lines = ["# Ralph Progress Log", f"Updated: {datetime.now().isoformat()}", ""]

        if self.patterns:
            lines.append("## Codebase Patterns")
            lines.append("")
            for p in self.patterns:
                lines.append(f"- {p}")
            lines.append("")
            lines.append("---")
            lines.append("")

        for entry in self.entries:
            lines.append(f"## {entry.timestamp.strftime('%Y-%m-%d %H:%M')} - {entry.story_id}")
            lines.append("")
            lines.append(f"**Status:** {entry.summary}")
            lines.append("")

            if entry.files_changed:
                lines.append("**Files changed:**")
                for f in entry.files_changed:
                    lines.append(f"- {f}")
                lines.append("")

            if entry.learnings:
                lines.append("**Learnings:**")
                for l in entry.learnings:
                    lines.append(f"- {l}")
                lines.append("")

            lines.append("---")
            lines.append("")

        with open(self.filepath, "w") as f:
            f.write("\n".join(lines))

        console.print(f"[dim]Progress saved to {self.filepath}[/dim]")

    def consolidate_patterns(self, new_learnings: list[str]) -> None:
        """
        Extract reusable patterns from learnings.
        This is a simple heuristic - could be enhanced with LLM.
        """
        pattern_keywords = [
            "always",
            "never",
            "must",
            "pattern",
            "convention",
            "use",
            "don't",
            "remember",
        ]

        for learning in new_learnings:
            learning_lower = learning.lower()
            if any(kw in learning_lower for kw in pattern_keywords):
                # This looks like a pattern
                if learning not in self.patterns:
                    self.patterns.append(learning)
                    console.print(f"[cyan]New pattern:[/cyan] {learning}")


def init_progress_file(filepath: str) -> None:
    """Initialize a new progress file."""
    if os.path.exists(filepath):
        return

    content = f"""# Ralph Progress Log
Started: {datetime.now().isoformat()}

## Codebase Patterns

(Patterns discovered during implementation will appear here)

---

"""
    with open(filepath, "w") as f:
        f.write(content)
