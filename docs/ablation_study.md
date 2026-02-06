# Ralph SDK Ablation Study 实验记录

> 记录日期: 2026-02-05
> 目的: 验证 Ralph SDK 各组件的实际价值，找到 pipeline 的真正优势场景

## 实验概览

| 实验 | 假设 | 结果 | 结论 |
|------|------|------|------|
| Evaluator 锚定效应 | Reviewer 结论会影响 Evaluator 评分 | **否定** | 9次评估全部 45 分，无锚定 |
| Worker 反思能力 | 加反思 prompt 能提升质量 | **无显著提升** | 简单任务本身就 100% |
| Claude 找隐藏 bug | Claude 能否发现非显式 bug | **100%** | 并发 bug、LRU cache bug 都能找到 |
| 复杂任务对比 | Pipeline 比 one-shot 更强 | **意外：one-shot 更快** | 定义清晰的任务不需要 pipeline |
| 时效性验证 | 主动搜索能减少 hallucination | **成功** | 机制完全按设计工作 |

---

## 实验 1: Evaluator 锚定效应

### 假设
Evaluator 基于 Reviewer 的结论继续评估，可能会受到 Reviewer 主观评价的"锚定"影响。

### 实验设计
- 相同代码，不同的 Reviewer 评价（正面 vs 负面 vs 无评价）
- 观察 Evaluator 给出的分数是否有显著差异

### 结果
```
9 次评估，全部返回 45 分
无论 Reviewer 说"代码很好"还是"代码有问题"，Evaluator 都给出相同分数
```

### 结论
**Evaluator 没有锚定效应**。它基于客观标准（代码实际情况）评分，不受 Reviewer 主观描述影响。

### 代码位置
`experiments/exp_evaluator_anchoring.py`

---

## 实验 2: Worker 反思能力

### 假设
在 Worker prompt 中加入反思要求（完成后自查、列出可能问题），能提升代码质量。

### 实验设计
两组对比：
- 控制组：标准 Worker prompt
- 实验组：加入反思 prompt（"完成后检查边界情况、思考哪里可能出错"）

测试任务：
- 简单任务：实现 `add(a, b)` 函数
- 中等任务：实现带边界检查的函数

### 结果
```
简单任务：两组都 100% 正确
中等任务：两组都 100% 正确
```

### 结论
**反思 prompt 对简单任务无显著提升**。Claude 本身在纯代码任务上已经很强，反思是多余的。

### 洞察
反思能力的价值可能在于：
- 需求模糊时帮助发现遗漏
- 多步骤任务中检查一致性
- 不在于简单函数实现

### 代码位置
`experiments/exp_worker_reflection.py`

---

## 实验 3: Claude 找隐藏 Bug

### 假设
Claude 能否发现代码中非显式标注的 bug（不是 `// BUG HERE` 这种）？

### 实验设计
两个隐藏 bug 场景：
1. **并发 bug**: 计数器没有加锁
2. **LRU Cache bug**: 访问时没有更新顺序

不给任何提示，只说"检查这段代码"。

### 结果
```
并发 bug: 100% 发现率
LRU Cache bug: 100% 发现率
```

Claude 能准确指出：
- "counter increment is not thread-safe"
- "get() doesn't update the access order"

### 结论
**Claude 在纯代码分析任务上非常强**。不需要提示就能发现隐藏的逻辑问题。

### 洞察
这解释了为什么简单任务不需要复杂 pipeline：Claude 单次就能做好。
Pipeline 的价值在于 Claude 单次做不好的任务。

### 代码位置
`experiments/exp_hidden_bug.py`

---

## 实验 4: 复杂任务对比 (Pipeline vs One-shot)

### 假设
对于"单次 Claude Code 做不出来"的复杂任务，Ralph SDK pipeline 应该更强。

### 实验设计
任务：给 ralph-sdk 添加并行任务执行支持
- 5 个验收标准：PARALLEL_EXECUTE action, task_ids field, asyncio.gather, file locking, max_parallel config

两组对比：
- Ralph SDK pipeline（完整流程）
- Claude Code one-shot（单次会话）

### 结果

| 指标 | Ralph SDK | Claude Code One-shot |
|------|-----------|---------------------|
| 时间 | 761+ 秒（未完成） | 323 秒 |
| Tool calls | 77+ | 50 |
| 评估分数 | T002 需要 RETRY | **5/5** |
| 完成度 | 部分完成 | 完全完成 |

### 结论
**意外：One-shot 在这个任务上更高效**

### 分析
这个任务虽然"复杂"，但有以下特点：
1. **定义清晰** - 5 个明确的验收标准
2. **纯代码任务** - 没有模糊的研究/探索需求
3. **单次会话能力足够** - Claude 可以在 50 个 tool calls 内完成

### 洞察
**Ralph SDK pipeline 的真正价值不在于"更能完成任务"，而在于：**
- 任务分解和追踪（适合多人/多天项目）
- Review/Retry 机制捕获问题
- 需求模糊需要反复澄清的任务
- 需要用户反馈迭代的任务

**对于定义清晰的编程任务，单次 Claude Code 反而更高效。**

### 代码位置
- `experiments/test_real_task.py`
- `experiments/test_claude_code_oneshot.py`

---

## 实验 5: 时效性验证机制

### 假设
Agent 基于过时知识（模型截止 2025-05）工作会产生 hallucination。
通过主动搜索验证可以减少这个问题。

### 实验设计
任务：使用 HuggingFace transformers 写文本生成脚本，要求"使用最新 API"

观察：
1. Clarifier 是否识别时效性话题
2. Worker 是否主动 WebSearch
3. Reviewer 是否检查未验证信息
4. 是否正确缓存验证结果

### 结果

| 检查项 | 结果 |
|--------|------|
| Initializer 识别时效性话题 | ✅ 创建 EXPLORE 先验证 |
| Worker 主动 WebSearch | ✅ 7+ 次搜索 |
| Reviewer TEMPORAL_CHECK | ✅ 输出 `needs_verification` |
| 验证结果缓存 | ✅ `[Verified 2026-02-05]` |
| 代码包含验证标注 | ✅ docstring 中标注来源 |

### 流程验证
```
Initializer → 识别"最新 API" → 创建 EXPLORE T001 先验证
    ↓
Worker T001 → 7次 WebSearch → 记录到 [Verified] Findings
    ↓
Worker T002 → 实现代码
    ↓
Reviewer → 检测未验证项 → TEMPORAL_CHECK: needs_verification
    ↓
Retry → 补充验证 6 个 UNVERIFIED_ITEMS → 更新代码标注
    ↓
DONE
```

### 结论
**时效性验证机制成功**。完全按设计工作，有效减少基于过时知识的 hallucination。

### 代码位置
- `experiments/test_temporal_verification.py`
- 机制实现：`ralph_sdk/prompts.py` (TEMPORAL_VERIFICATION_PRINCIPLE)
- 辅助函数：`ralph_sdk/pool.py` (add_verified_info, get_verified_info, etc.)

---

## 总结与洞察

### Claude 的能力边界

| 任务类型 | Claude 表现 | 需要 Pipeline? |
|----------|-------------|----------------|
| 纯代码实现（定义清晰） | 很强 | ❌ 不需要 |
| 代码分析/找 bug | 很强 | ❌ 不需要 |
| 需要最新信息的任务 | 会 hallucinate | ✅ 需要验证机制 |
| 需求模糊需要迭代 | 单次难以完成 | ✅ 需要 pipeline |
| 多步骤、多天的项目 | 需要状态管理 | ✅ 需要 pipeline |

### Ralph SDK 的真正价值

1. **不是"比 Claude 更聪明"** - 同一个模型，不会更聪明
2. **而是"更好的工作流"**：
   - 任务分解和追踪
   - Review/Retry 机制
   - 时效性验证
   - 悲观准备（探索替代方案）
   - 状态持久化（可中断恢复）

### 什么时候用 Ralph SDK

✅ **适合使用**：
- 需求不明确，需要迭代澄清
- 需要用户测试反馈的迭代任务
- 涉及时效性信息（API、最佳实践）
- 多天/多人协作的项目
- 需要追踪进度和状态

❌ **不需要使用**：
- 定义清晰的编程任务
- 单次能完成的代码实现
- 简单的代码分析/修复

### 后续改进方向

1. **更智能的任务判断** - 自动识别是否需要 pipeline
2. **更轻量的模式** - 简单任务跳过 Review/Evaluate
3. **更好的并行支持** - 真正的任务并行执行
4. **用户反馈循环** - 更好的 checkpoint/测试机制

---

## 实验代码索引

```
experiments/
├── exp_evaluator_anchoring.py    # Evaluator 锚定效应实验
├── exp_worker_reflection.py      # Worker 反思能力实验
├── exp_hidden_bug.py             # 隐藏 bug 发现实验
├── test_real_task.py             # 复杂任务 pipeline 测试
├── test_claude_code_oneshot.py   # 复杂任务 one-shot 测试
└── test_temporal_verification.py # 时效性验证机制测试
```
