# Code → Prompt 实验结果

## 概要

| 实验 | 目标 | 成功标准 | 结果 | 判定 |
|------|------|---------|------|------|
| 1: Reviewer JSON | 替换 3 regex (24行) | verdict >= 95% | 100% | PASSED |
| 2: Evaluator Pivot | 替换 5 条件 (48行) | 判断 >= 90% | 87.5% | FAILED (borderline) |
| 3: Planner JSON | 替换 15+ regex (124行) | action >= 90%, target >= 85% | 91.7% / 100% | PASSED |

## 实验 1: Reviewer JSON 输出

### Phase 1: 解析器准确率 (合成输入)
- Regex 解析器: 12/20 (60%) — 无法处理 JSON 格式输入
- JSON 解析器: 20/20 (100%) — 处理所有格式

### Phase 2: Claude API 真实输出 (20 trials)
- JSON 解析器: 20/20 (100%)
- Regex 解析器: 5/20 (25%) — Claude 输出 JSON 时 regex 失效

### 结论
JSON 解析器是 regex 解析器的**严格超集**。可以安全替换。

### 建议修改
```python
# reviewer.py:37-60 可以替换为:
def parse_reviewer_output(text: str) -> ReviewResult:
    json_obj = _extract_json(text)
    if json_obj:
        return ReviewResult(verdict=Verdict(json_obj["verdict"].lower()), ...)
    # fallback to regex for backwards compatibility
    ...
```

---

## 实验 2: Evaluator Pivot 检测

### Phase 1: 代码逻辑验证
- 8/8 场景全部正确

### Phase 2: Claude prompt 判断 (40 trials)
- 总准确率: 35/40 (87.5%) — 低于 90% 阈值
- 失败场景: `hard_metric_keeps_failing` (0/5)

### 失败分析
Claude 的推理:
> "Score improved from 50 to 55 (+5 points). While tests_pass is failing,
> it has only failed for 2 attempts — right at the threshold..."

**这不是 Claude "犯错"，而是合理的不同判断。** 代码的硬编码规则说 "hard metric fails 2+ times = pivot"，但 Claude 看到分数在改善，认为继续更合理。

### 结论
- 7/8 场景 100% 正确
- 唯一失败是 borderline case，Claude 的判断可能更合理
- **建议**: 保留代码逻辑作为硬限制，但可以用 prompt 处理更模糊的判断

---

## 实验 3: Planner 结构化输出

### 关键发现: tool-use 不可用
`claude-code-sdk` 的 `ClaudeCodeOptions` 不支持自定义 tool 定义。只能用内置工具 (Read, Write, Bash 等)。

**替代方案**: 改用 JSON 输出 (与实验 1 相同策略)。

### Phase 1: Regex 解析器验证
- 12/12 (100%) — 在标准格式输入上完美

### Phase 2: JSON 输出 (36 trials)
- JSON 解析率: 36/36 (100%) — Claude 总是输出有效 JSON
- Action 准确率: 33/36 (91.7%)
- Target 准确率: 24/24 (100%)

### 失败分析
唯一失败: `hedge` 场景 (0/3) — Claude 选择 `ask` 而非 `hedge`

原因: `hedge` 是 Ralph 特有的概念 (悲观准备)，Claude 自然倾向于"问用户"而非"探索替代方案"。这是 **prompt engineering** 问题，不是结构问题。

### 结论
JSON 输出可以替换 15+ regex，但需要在 prompt 中明确 `hedge` 的含义和使用场景。

---

## 横向发现

### 1. JSON 是可靠的结构化输出通道
三个实验中 JSON 解析率:
- 实验 1: 100%
- 实验 2: N/A (用的是 DECISION: 格式)
- 实验 3: 100%

**结论**: 告诉 Claude 输出 JSON，它就会输出 JSON，且格式正确。

### 2. claude-code-sdk 不支持自定义 tool 定义
这排除了 Experiment 3 的 Option B (tool-use)。JSON 输出是更好的替代方案。

### 3. Claude 在 borderline case 有自己的判断
实验 2 显示 Claude 不会机械执行数值规则 — 它会综合考虑上下文。这对于简单阈值判断是劣势 (87.5% < 90%)，但对于需要 nuance 的判断可能是优势。

### 4. Ralph 特有概念需要强化 prompt
`hedge` (悲观准备) 是 Ralph 独有的，Claude 不理解其含义。需要在 prompt 中用例子明确说明。

---

## 实验 4: Orchestrator Agent Loop

### 结果 (15 trials)
- JSON 解析率: 15/15 (100%)
- **最终状态正确率: 15/15 (100%)**
- 迭代次数限制内: 0/15 (0%) — Claude 生成的步骤更详细

### 分析

| 场景 | 期望最终状态 | 结果 | Python 迭代 | Claude 步骤 |
|------|------------|------|------------|------------|
| A: 简单通过 | done | 100% correct | ~3 | 7-9 |
| B: 重试后通过 | done | 100% correct | ~5 | 10-14 |
| C: 分数下降触发 pivot | pivot | 100% correct | ~6 | 21-29 |
| D: 需要用户测试 | checkpoint | 100% correct | ~4 | 8 |
| E: 并行执行 | done | 100% correct | ~6 | 12-19 |

### 关键发现

**Claude 理解工作流逻辑，但输出更多步骤。** 这是因为：
1. Claude 把 plan/work/review/evaluate 拆成独立步骤
2. Python 代码在一个 iteration 内完成 work→review→evaluate
3. Claude 的步骤数 ≈ Python 迭代数 × 3（因为每个子操作是一步）

**最终状态 100% 正确 = Claude 完全理解了调度逻辑。**

### 结论
- 路由逻辑可以交给 agent
- **必须保留在代码中的**: max_iterations 硬限制、error recovery、文件锁
- 迭代效率需要优化（agent 步骤过多 = API 调用过多 = 成本高）

---

## 总结决策

### 前 3 个实验中:
- 实验 1: PASSED — JSON 输出可靠
- 实验 2: 87.5% (borderline) — prompt 可改进
- 实验 3: PASSED — JSON 输出可靠
- 实验 4: Final state 100%, 但步骤效率低

### 已实施的改动
1. ✅ Reviewer: JSON 解析 + regex fallback (`reviewer.py`)
2. ✅ Planner: JSON 解析 + regex fallback (`planner.py`)
3. ✅ Prompts: 请求 JSON 输出格式 (`prompts.py`)

### 建议后续
1. **Evaluator pivot**: 保留代码硬限制，prompt 可作为辅助信号
2. **Planner hedge**: 强化 prompt 中 hedge 概念
3. **Orchestrator**: 保留 Python 循环结构，但路由逻辑可简化（agent 理解所有场景）
4. **不建议**: 完全用 agent 替代 orchestrator（步骤效率太低）

---

## 渐进合并路线图

### 立即可做 (低风险)
1. **Reviewer**: 添加 JSON 解析路径，保留 regex fallback
2. **Planner**: 添加 JSON 解析路径，保留 regex fallback

### 需要 prompt 改进后做 (中风险)
3. **Planner hedge**: 强化 prompt 中 hedge 概念的说明
4. **Evaluator pivot**: 改进 hard_metric_keeps_failing 条件的 prompt 描述

### 需要实验 4 验证后做 (高风险)
5. **Orchestrator**: 将路由逻辑转为 agent 决策
6. 保留: max_iterations, error recovery, 文件锁
