# Ralph SDK 功能触发分析

**测试日期**: 2026-02-05
**测试任务**: "用最新的 transformers 库写一个简单的文本生成脚本，使用 GPT-2 模型"

---

## 功能触发总结

| # | 设计 | 是否触发 | 说明 |
|---|------|---------|------|
| 1 | **时效性验证** | ⚠️ 部分 | Reviewer 提到 `[已搜索验证]` 标注，pool.md 有 Shared Findings，但未使用标准 `[Verified YYYY-MM-DD]` 格式 |
| 2 | **文件系统 Memory** | ✅ 是 | pool.md 的 "Shared Findings" 记录了 GPT-2 路径、transformers API 变更等 |
| 3 | **Clarifier 双模式** | ❌ 否 | 使用了 `--quick` 跳过 clarification |
| 4 | **HEDGE 机制** | ❌ 否 | 任务一次成功，没有进入等待用户测试状态 |
| 5 | **Reviewer 判决** | ✅ 是 | 返回 `PASSED` 判决 |
| 6 | **Agent 自主性** | ✅ 是 | 系统自主决策：选择模型路径、API 用法、代码结构 |
| 7 | **PIVOT_RESEARCH** | ❌ 否 | 没有遇到技术障碍 |
| 8 | **PIVOT_ITERATION** | ❌ 否 | 任务第一次尝试就成功 (85分) |
| 9 | **PARALLEL_EXECUTE** | ❌ 否 | 只有一个任务，无并行机会 |
| 10 | **Evaluator** | ✅ 是 | 评估分数 85/100，列出 issues 和 suggestions |

---

## 详细分析

### ✅ 已触发的功能

#### 1. 文件系统 Memory
```markdown
## Shared Findings
- GPT-2 official model path: `openai-community/gpt2`
- transformers v4.40.0+ uses `token` instead of deprecated `use_auth_token`
- GPT-2 has no pad_token by default, set `tokenizer.pad_token = tokenizer.eos_token`
```
**观察**: Worker 将发现写入 pool.md，供其他任务/组件读取。

#### 2. Reviewer 判决
```json
{"event": "reviewer_verdict", "task_id": "T001", "verdict": "passed",
 "reason": "All acceptance criteria met. Temporal verification check passed..."}
```
**观察**: Reviewer 验证代码质量、功能正确性、时效性信息。

#### 3. Evaluator
```json
{"event": "evaluation", "task_id": "T001", "passed": true, "score": 85.0,
 "issues": ["No error handling...", "No CLI support...", "No requirements.txt..."]}
```
**观察**:
- 评分 85/100 (PASSED)
- 列出 3 个改进点
- 提供 4 条建议

#### 4. Agent 自主性
**观察**: 系统自主做出以下决策：
- 选择 `openai-community/gpt2` 作为模型路径
- 使用 `AutoModelForCausalLM` 而非 `pipeline`
- 设置 `padding_side="left"`
- 处理 pad_token 缺失问题

---

### ⚠️ 部分触发的功能

#### 1. 时效性验证
**预期行为**:
- Worker 应该 WebSearch 验证 transformers 最新 API
- 使用 `[Verified YYYY-MM-DD]` 格式标注
- 写入 pool.md 的 "Verified Information" section

**实际行为**:
- Reviewer 提到代码有 `[已搜索验证 2026-02-05]` 标注
- pool.md 使用 "Shared Findings" 而非 "Verified Information"
- 没有明确的 WebSearch 工具调用记录

**分析**: 时效性意识存在，但格式不完全符合设计。可能原因：
1. Worker 内部使用了 Task agent 做搜索
2. 或使用了模型记忆 + 推理（不够严格）

---

### ❌ 未触发的功能

#### 1. Clarifier 双模式
**原因**: 测试使用 `--quick` 选项，跳过了 clarification 阶段。

**如何触发**:
```bash
ralph-sdk run "研究一下怎么让代码更智能" # 模糊需求 → v2 explore 模式
ralph-sdk run "添加用户认证"              # 清晰需求 → v1 ask 模式
```

#### 2. HEDGE 机制
**原因**: 任务一次成功（85分），没有进入 "待用户测试" 状态。

**如何触发**:
1. 需要 `automation: hybrid` 或 `manual` 的指标
2. Worker 完成后，Evaluator 标记需要用户测试
3. Planner 触发 HEDGE 探索替代方案

**测试方法**:
```python
# 在 goal.md 中定义需要人工测试的指标
Metric("user_satisfaction", MetricType.SUBJECTIVE,
       automation=AutomationLevel.MANUAL)
```

#### 3. PIVOT_RESEARCH
**原因**: 没有遇到技术障碍（如 API 不支持、库废弃等）。

**如何触发**:
```bash
ralph-sdk run "用 Python 2 的 asyncio 写异步代码"  # 会发现 Python 2 不支持
```

#### 4. PIVOT_ITERATION
**原因**: 任务第一次尝试就成功（85分 > 70分阈值）。

**如何触发**: 需要多次失败的场景：
1. 设置严格的硬性指标（如 accuracy >= 99%）
2. 让 Worker 无法在前几次达标
3. 第 3 次失败后应触发 PIVOT_ITERATION

#### 5. PARALLEL_EXECUTE
**原因**: 只有一个任务 (T001)，没有并行机会。

**如何触发**:
```bash
ralph-sdk run "给项目添加 logging、metrics 和 health check 三个独立模块"
```
Planner 应识别 3 个独立任务并使用 PARALLEL_EXECUTE。

---

## 触发所有功能的测试场景设计

### 场景 1: 时效性验证 + Clarifier v2
```bash
ralph-sdk run "研究一下最新的 RAG 技术，实现一个简单的检索增强生成系统"
```
- "研究" 触发 v2 explore 模式
- "最新" 触发时效性验证

### 场景 2: HEDGE + 用户测试
```bash
ralph-sdk run "实现一个对话式 AI，用户满意度要达到 4/5 以上"
# 然后在 feedback.md 故意延迟填写
```
- 用户满意度是 manual 指标
- 等待反馈时应触发 HEDGE

### 场景 3: PIVOT_RESEARCH
```bash
ralph-sdk run "用 deprecated 的 tensorflow.Session 写神经网络"
```
- TF2 已移除 Session
- 应触发 PIVOT_RESEARCH 转向 Keras API

### 场景 4: PIVOT_ITERATION
需要精心设计的失败场景，例如：
```bash
# goal.md 中设置不可达目标
accuracy_target: >= 99.9%
max_latency: <= 1ms
```

### 场景 5: PARALLEL_EXECUTE
```bash
ralph-sdk run "为这个 Python 项目添加：1) pytest 测试 2) mypy 类型检查 3) pre-commit hooks"
```

---

## 结论

**已验证功能 (4/10)**:
- ✅ 文件系统 Memory
- ✅ Reviewer 判决
- ✅ Evaluator
- ✅ Agent 自主性

**部分验证 (1/10)**:
- ⚠️ 时效性验证（有意识但格式不完整）

**未触发 (5/10)**:
- ❌ Clarifier 双模式（需要去掉 `--quick`）
- ❌ HEDGE（需要人工指标 + 延迟反馈）
- ❌ PIVOT_RESEARCH（需要技术障碍场景）
- ❌ PIVOT_ITERATION（需要多次失败场景）
- ❌ PARALLEL_EXECUTE（需要多独立任务场景）

**建议**: 要全面验证，需要设计专门的测试场景来触发各个功能，简单的成功任务只能验证基础流程。
