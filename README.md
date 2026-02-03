# Easy Ralph

An autonomous AI agent that turns natural language requirements into working code through a three-phase pipeline.

Built on top of [Claude Code SDK](https://github.com/anthropics/claude-code), Easy Ralph automates the entire software development workflow: from requirements clarification to PRD generation to iterative code implementation.

## Features

- **Autonomous Execution** - Minimal human intervention required
- **Three-Phase Pipeline** - Clarify → Plan → Execute
- **Progress Tracking** - Resume from where you left off
- **Learning Accumulation** - Each iteration builds on previous learnings
- **Real-time Visualization** - See exactly what the agent is doing

## Installation

```bash
# 1. Install Claude Code CLI (required runtime)
npm install -g @anthropic-ai/claude-code

# 2. Authenticate with your Anthropic API key
claude  # Follow the prompts to set up

# 3. Install Easy Ralph
pip install git+https://github.com/ChaosCodes/easy-ralph.git

# Or install from source
git clone https://github.com/ChaosCodes/easy-ralph.git
cd easy-ralph
pip install -e .
```

> **Note**: Easy Ralph uses [claude-code-sdk](https://pypi.org/project/claude-code-sdk/) which requires Claude Code CLI as its underlying runtime. The SDK calls CLI internally to interact with Claude and execute tools (Read, Write, Edit, Bash, etc.).

## Quick Start

```bash
# Full pipeline: clarify → plan → execute
ralph-sdk run "Add user authentication with JWT"

# Quick mode (skip clarification)
ralph-sdk run "Fix the bug in auth.py" --quick
```

## Usage

### CLI Commands

```bash
# Full pipeline
ralph-sdk run "Add a todo list feature" --max 20

# Plan only (generate PRD without executing)
ralph-sdk plan "Add task priority feature" -o prd.json

# Execute from existing PRD
ralph-sdk execute prd.json --max 20

# Check progress
ralph-sdk status

# Reset a story to re-execute
ralph-sdk reset --story US-003

# Reset all stories
ralph-sdk reset --all
```

### Programmatic API

```python
import asyncio
from ralph_sdk import run_ralph

async def main():
    success = await run_ralph(
        initial_prompt="Add a search feature",
        cwd="./my-project",
        max_iterations=10,
        skip_clarify=False,  # Set True to skip clarification
    )
    print(f"Completed: {success}")

asyncio.run(main())
```

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                     ralph-sdk run "prompt"                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: Requirements Clarification                            │
│  ─────────────────────────────────────                          │
│  • Generate 3-5 clarifying questions                            │
│  • Collect user answers interactively                           │
│  • Produce structured requirements (scope, goals, non-goals)    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2: PRD Generation                                        │
│  ───────────────────────                                        │
│  • Explore codebase for patterns and conventions                │
│  • Generate structured PRD with user stories                    │
│  • Decompose into right-sized tasks (one context window each)   │
│  • Order by dependencies (schema → backend → UI)                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 3: Story Execution Loop                                  │
│  ────────────────────────────                                   │
│  For each user story (by priority):                             │
│    1. Load progress context from previous iterations            │
│    2. Execute with fresh Claude Code session                    │
│    3. Run quality checks (typecheck, lint, tests)               │
│    4. Commit changes if all checks pass                         │
│    5. Extract learnings for next iteration                      │
│    6. Update prd.json and progress.txt                          │
│                                                                 │
│  Continue until: all stories done OR max iterations reached     │
└─────────────────────────────────────────────────────────────────┘
```

## Output Example

```
╭────────────────────── 🚀 Executing Story ──────────────────────╮
│ US-001 Set up project structure with dependencies              │
│                                                                │
│ As a developer, I want a proper project structure...           │
╰────────────────────────────────────────────────────────────────╯

[ 1] 🔍 Glob **/*.py
[ 2] 📖 Read pyproject.toml
[ 3] 📖 Read src/main.py
     💭 I'll set up the project structure...
[ 4] ✏️  Edit pyproject.toml -1 +5
[ 5] 💻 Bash  pip install -e .
[ 6] 💻 Bash  python -c "import myproject"

╭─────────────────────── US-001 ─────────────────────────────────╮
│ ✓ Success  6 turns, 6 tool calls                               │
│ Files: pyproject.toml, src/__init__.py                         │
│ Commit: abc1234                                                │
│ Learnings:                                                     │
│   • This project uses pytest for testing                       │
╰────────────────────────────────────────────────────────────────╯
```

## Project Structure

```
ralph_sdk/
├── __init__.py         # Public API
├── models.py           # Data models (PRD, UserStory, StoryResult, etc.)
├── prompts.py          # System prompts for each phase
├── clarifier.py        # Phase 1: Requirements clarification
├── prd_generator.py    # Phase 2: PRD generation
├── executor.py         # Phase 3: Story execution
├── progress.py         # Progress tracking and learning accumulation
├── orchestrator.py     # Main pipeline orchestration
└── cli.py              # CLI entry point
```

## Configuration

Easy Ralph creates two files in your project directory:

- **`prd.json`** - The generated PRD with user stories and their completion status
- **`progress.txt`** - Execution log with accumulated learnings

These files enable:
- Resuming interrupted executions
- Tracking what has been done
- Passing learnings between iterations

## Requirements

- Python 3.10+
- [Claude Code CLI](https://github.com/anthropics/claude-code) installed and authenticated
- Anthropic API key (set via `ANTHROPIC_API_KEY` environment variable)

## License

MIT

---

# 中文说明

Easy Ralph 是一个自主 AI 代理，通过三阶段流水线将自然语言需求转化为可运行的代码。

## 三阶段流程

1. **需求澄清** - 生成澄清问题，收集用户回答，生成结构化需求
2. **PRD 生成** - 探索代码库，生成用户故事，按依赖排序
3. **故事执行** - 迭代执行每个故事，运行质量检查，提交代码

## 快速使用

```bash
# 1. 安装 Claude Code CLI（底层运行时）
npm install -g @anthropic-ai/claude-code
claude  # 首次运行完成认证

# 2. 安装 Easy Ralph
pip install git+https://github.com/ChaosCodes/easy-ralph.git

# 3. 运行
ralph-sdk run "添加用户认证功能"

# 快速模式（跳过澄清）
ralph-sdk run "修复 auth.py 中的 bug" --quick

# 从现有 PRD 继续执行
ralph-sdk execute prd.json
```

> **说明**: Easy Ralph 使用 claude-code-sdk，它需要 Claude Code CLI 作为底层运行时。SDK 内部调用 CLI 来与 Claude 交互并执行工具操作。

## 特点

- 自主执行，最小化人工干预
- 支持断点续跑
- 迭代间学习积累
- 实时可视化执行过程
