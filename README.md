# Ralph SDK

基于 Claude Agent SDK 的自主 AI agent 循环，将 PRD 生成和任务执行整合到统一流程中。

## 安装

```bash
cd ralph-sdk
pip install -e .
```

需要先安装 Claude Code CLI：
```bash
npm install -g @anthropic-ai/claude-code
```

## 使用

### 完整流程（推荐）

一条命令完成：需求澄清 → PRD 生成 → 自动执行

```bash
ralph-sdk run "添加用户认证功能"
```

### 分步骤

```bash
# 只生成 PRD（不执行）
ralph-sdk plan "添加任务优先级功能" -o prd.json

# 从现有 PRD 执行
ralph-sdk execute prd.json --max 20

# 查看进度
ralph-sdk status

# 重置某个 story 重新执行
ralph-sdk reset --story US-003
```

### 快速模式

跳过需求澄清阶段（适用于需求已经很明确的情况）：

```bash
ralph-sdk run "Fix the bug in auth.py" --quick
```

## 编程使用

```python
import asyncio
from ralph_sdk import run_ralph

async def main():
    success = await run_ralph(
        initial_prompt="Add a todo list feature",
        cwd="./my-project",
        max_iterations=10,
    )
    print(f"Completed: {success}")

asyncio.run(main())
```

## 工作流程

```
┌─────────────────────────────────────────────────────────────┐
│                        ralph-sdk run                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: Requirements Clarification                        │
│  - Analyze initial request                                  │
│  - Generate clarifying questions                            │
│  - Collect user answers                                     │
│  - Produce clarified requirements                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 2: PRD Generation                                    │
│  - Explore codebase for patterns                            │
│  - Generate structured PRD                                  │
│  - Decompose into right-sized user stories                  │
│  - Save to prd.json                                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 3: Story Execution Loop                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  For each story (by priority):                         │ │
│  │  1. Load progress context                              │ │
│  │  2. Execute story (fresh context each time)            │ │
│  │  3. Run quality checks                                 │ │
│  │  4. Commit if passed                                   │ │
│  │  5. Update prd.json and progress.txt                   │ │
│  │  6. Extract learnings for next iteration               │ │
│  └────────────────────────────────────────────────────────┘ │
│                              │                              │
│                              ▼                              │
│                    All stories done?                        │
│                     /            \                          │
│                   Yes             No                        │
│                    │               │                        │
│                    ▼               └──────────┐             │
│               [COMPLETE]                      │             │
│                                               ▼             │
│                                    Max iterations?          │
│                                     /            \          │
│                                   Yes             No        │
│                                    │               │        │
│                                    ▼               │        │
│                               [TIMEOUT]      [NEXT STORY]   │
└─────────────────────────────────────────────────────────────┘
```

## 文件结构

```
ralph_sdk/
├── __init__.py         # 公开接口
├── models.py           # 数据模型 (PRD, UserStory, etc.)
├── prompts.py          # System prompts
├── clarifier.py        # 需求澄清模块
├── prd_generator.py    # PRD 生成模块
├── executor.py         # Story 执行模块
├── progress.py         # 进度跟踪
├── orchestrator.py     # 主流程编排
└── cli.py              # 命令行入口
```

## 与原版 Ralph 对比

| 特性 | 原版 Ralph | Ralph SDK |
|------|-----------|-----------|
| PRD 生成 | 手动触发 skill | 自动集成 |
| JSON 转换 | 手动触发 skill | 自动完成 |
| 执行循环 | Bash 脚本 | Python 代码 |
| 上下文隔离 | 每次启动新进程 | 每次新 query() |
| 进度持久化 | 文件 | 文件 + 内存 |
| 自定义逻辑 | 修改 bash/prompt | 修改 Python |
| 失败处理 | 继续下一轮 | 可配置重试 |

## License

MIT
