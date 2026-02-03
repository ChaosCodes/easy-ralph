# Ralph SDK 版本计划

## 目标

用 Claude Agent SDK 重新实现 Ralph，将 PRD 生成和任务执行整合到一个统一流程中。

## 原版 Ralph 工作流（分散）

```
用户 → (手动) 运行 prd skill → 生成 PRD.md
     → (手动) 运行 ralph skill → 生成 prd.json
     → (手动) 运行 ralph.sh → 循环执行任务
```

## SDK 版本工作流（统一）

```
用户 → 输入需求描述
     → Phase 1: 需求澄清 + PRD 生成（对话式）
     → Phase 2: 自动分解为 user stories
     → Phase 3: 自主循环执行，直到全部完成
```

---

## 技术选型

- **语言**: Python 3.10+
- **SDK**: `claude-agent-sdk`
- **依赖**:
  - `claude-agent-sdk`
  - `pydantic` (数据模型)
  - `rich` (终端 UI)

---

## 核心数据结构

```python
from pydantic import BaseModel
from typing import Literal

class UserStory(BaseModel):
    id: str                          # "US-001"
    title: str
    description: str
    acceptance_criteria: list[str]
    priority: int
    passes: bool = False
    notes: str = ""

class PRD(BaseModel):
    project: str
    branch_name: str
    description: str
    user_stories: list[UserStory]
```

---

## 模块设计

### 1. `ralph_sdk/clarifier.py` - 需求澄清模块

职责：与用户交互，收集需求细节

```python
async def clarify_requirements(initial_prompt: str) -> ClarifiedRequirements:
    """
    1. 分析初始需求
    2. 生成 3-5 个澄清问题（带选项）
    3. 收集用户回答
    4. 返回完整需求描述
    """
```

### 2. `ralph_sdk/prd_generator.py` - PRD 生成模块

职责：基于澄清后的需求生成结构化 PRD

```python
async def generate_prd(requirements: ClarifiedRequirements) -> PRD:
    """
    1. 调用 Claude 生成 PRD
    2. 自动分解为合适大小的 user stories
    3. 确保依赖顺序正确
    4. 返回 PRD 对象
    """
```

### 3. `ralph_sdk/executor.py` - 任务执行模块

职责：自主执行单个 user story

```python
async def execute_story(story: UserStory, context: ExecutionContext) -> StoryResult:
    """
    1. 读取当前代码状态
    2. 实现 story 的功能
    3. 运行质量检查
    4. 提交代码（如果通过）
    5. 返回执行结果
    """
```

### 4. `ralph_sdk/orchestrator.py` - 主循环编排

职责：协调整个流程

```python
async def run_ralph(initial_prompt: str, max_iterations: int = 10):
    """
    1. Phase 1: 需求澄清
    2. Phase 2: PRD 生成
    3. Phase 3: 循环执行
       - 选择下一个 passes=false 的 story
       - 执行 story
       - 更新状态
       - 记录 progress
       - 检查是否全部完成
    """
```

### 5. `ralph_sdk/progress.py` - 进度管理

职责：跟踪和持久化进度

```python
class ProgressTracker:
    def append_log(self, story_id: str, learnings: list[str])
    def update_patterns(self, patterns: list[str])
    def mark_story_complete(self, story_id: str)
    def save(self)
    def load(self)
```

---

## 关键实现细节

### 使用 ClaudeSDKClient 保持对话上下文

```python
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

async def clarify_requirements(initial_prompt: str):
    options = ClaudeAgentOptions(
        system_prompt=CLARIFIER_SYSTEM_PROMPT,
        allowed_tools=["AskUserQuestion"],
        max_turns=10
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.query(f"用户需求: {initial_prompt}\n请生成澄清问题")
        # 收集回答...
```

### 执行器使用完整工具集

```python
async def execute_story(story: UserStory, cwd: str):
    options = ClaudeAgentOptions(
        system_prompt=EXECUTOR_SYSTEM_PROMPT.format(story=story),
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        permission_mode="acceptEdits",
        cwd=cwd,
        max_turns=50
    )

    async for message in query(prompt=story.to_prompt(), options=options):
        # 处理执行结果...
```

### 每次迭代清空上下文

```python
# 原版 Ralph 的关键设计：每次迭代是全新实例
# SDK 版本通过每次创建新的 query() 调用实现相同效果

for story in prd.get_pending_stories():
    # 每次 query() 都是全新上下文
    result = await execute_story(story, context)

    # 只通过文件系统传递状态
    progress.append_log(story.id, result.learnings)
    progress.save()
```

---

## 文件结构

```
ralph-sdk/
├── PLAN.md              # 本计划文件
├── pyproject.toml       # 项目配置
├── ralph_sdk/
│   ├── __init__.py
│   ├── models.py        # Pydantic 数据模型
│   ├── clarifier.py     # 需求澄清
│   ├── prd_generator.py # PRD 生成
│   ├── executor.py      # 任务执行
│   ├── orchestrator.py  # 主流程编排
│   ├── progress.py      # 进度跟踪
│   ├── prompts.py       # 系统 prompt 模板
│   └── cli.py           # 命令行入口
└── examples/
    └── demo.py          # 使用示例
```

---

## 命令行接口

```bash
# 完整流程（从需求到执行）
ralph-sdk run "添加用户认证功能"

# 只生成 PRD（不执行）
ralph-sdk plan "添加用户认证功能" --output prd.json

# 从现有 PRD 继续执行
ralph-sdk execute prd.json --max-iterations 10

# 查看进度
ralph-sdk status
```

---

## 与原版 Ralph 的对比

| 特性 | 原版 Ralph | SDK 版本 |
|------|-----------|---------|
| PRD 生成 | 手动触发 skill | 自动集成 |
| JSON 转换 | 手动触发 skill | 自动完成 |
| 执行循环 | Bash 脚本 | Python 代码 |
| 上下文隔离 | 每次启动新进程 | 每次新 query() |
| 进度持久化 | 文件 | 文件 + 内存 |
| 自定义逻辑 | 修改 bash/prompt | 修改 Python 代码 |
| 失败重试 | 无 | 可配置 |
| 并行执行 | 无 | 可扩展 |

---

## 实现步骤

1. [ ] 初始化项目结构
2. [ ] 实现数据模型 (models.py)
3. [ ] 实现 prompts 模板 (prompts.py)
4. [ ] 实现需求澄清模块 (clarifier.py)
5. [ ] 实现 PRD 生成模块 (prd_generator.py)
6. [ ] 实现任务执行模块 (executor.py)
7. [ ] 实现进度管理 (progress.py)
8. [ ] 实现主编排器 (orchestrator.py)
9. [ ] 实现 CLI 入口 (cli.py)
10. [ ] 测试完整流程

---

## 下一步

确认计划后，我将开始实现代码。
