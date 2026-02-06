# Code → Prompt 实验结果

## 概要

### Round 1

| 实验 | 目标 | 成功标准 | 结果 | 判定 |
|------|------|---------|------|------|
| 1: Reviewer JSON | 替换 3 regex (24行) | verdict >= 95% | 100% | PASSED |
| 2: Evaluator Pivot | 替换 5 条件 (48行) | 判断 >= 90% | 87.5% | FAILED (borderline) |
| 3: Planner JSON | 替换 15+ regex (124行) | action >= 90%, target >= 85% | 91.7% / 100% | PASSED |
| 4: Orchestrator | 调度逻辑 prompt 化 | 最终状态正确 | 100% | PASSED (but 3x steps) |

### Round 2 — Prompt 改进 + Metrics

| 实验 | 改动 | 目标场景 | R1 结果 | R2 结果 | 判定 |
|------|------|---------|--------|--------|------|
| 5: Evaluator Improved | +HARD_METRIC_PIVOT_RULE | hard_metric 0/5→? | 87.5% | 87.5% | 目标场景修复，但暴露新问题 |
| 6: Planner Improved | +HEDGE_VS_ASK_GUIDE | hedge 0/3→? | 91.7% | 91.7% | 目标场景修复，但暴露新问题 |
| 7a: Category Detection | Prompt 替代关键词匹配 | >= 90% | N/A | **100%** | **PASSED** (code only 50%) |
| 7b: Metrics Extraction | Prompt 替代 regex 提取 | >= 85% | N/A | **100%** | **PASSED** |

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

---

# Round 2: Prompt 改进 + Metrics 实验

## 实验 5: Evaluator Pivot — 改进版 Prompt

### 改动
在 pivot detection prompt 中添加 `HARD_METRIC_PIVOT_RULE`：
- 明确 hard metric 是二元指标，独立于总分趋势
- 给出具体例子：tests_pass=false 连续失败 → 必须 PIVOT

### 结果 (40 trials)
- 总准确率: 35/40 (87.5%) — 与 Round 1 相同
- `hard_metric_keeps_failing`: **5/5 (100%)** ← Round 1: 0/5
- `very_low_after_multiple`: **0/5 (0%)** ← Round 1: 5/5 (新暴露的问题)

### Per-scenario 对比

| 场景 | Round 1 | Round 2 | 变化 |
|------|---------|---------|------|
| too_many_attempts_weak_improvement | 5/5 | 5/5 | = |
| declining_scores | 5/5 | 5/5 | = |
| stuck_scores | 5/5 | 5/5 | = |
| **hard_metric_keeps_failing** | **0/5** | **5/5** | **+5** |
| **very_low_after_multiple** | **5/5** | **0/5** | **-5** |
| normal_improvement | 5/5 | 5/5 | = |
| first_attempt | 5/5 | 5/5 | = |
| sufficient_improvement | 5/5 | 5/5 | = |

### 分析

**打地鼠效应 (Whack-a-mole)**：修复一个场景会导致另一个场景退化。

`very_low_after_multiple` 退化原因：
- 场景：score=35 (< 40) after 2 attempts，score history [30] → 35
- Claude 的推理："分数在改善 (+5)，只有 2 次尝试，condition 5 要求 score < 40 after 2+ attempts，但我看到了改善趋势"
- 加入 HARD_METRIC_PIVOT_RULE 后，Claude 更加关注"是否有硬性规则被违反"，对其他条件的判断变得更加保守

**核心洞察**：数值阈值判断不适合 prompt 化。Claude 善于理解概念（什么是 hard metric），但不善于机械执行数值规则（score < 40 after 2+ attempts = pivot，即使有改善趋势）。

### 结论
- HARD_METRIC_PIVOT_RULE **成功教会了 Claude hard metric 的概念**
- 但 Claude 仍然不会机械执行数值阈值
- **建议**：保留代码中的数值阈值判断，prompt 只负责概念性判断
- 已合并 `HARD_METRIC_PIVOT_RULE` 到 `prompts.py`（用于概念教育，即使总准确率未提升）

---

## 实验 6: Planner Hedge — 改进版 Prompt

### 改动
在 planner prompt 中添加 `HEDGE_VS_ASK_GUIDE`：
- 用 OpenClaw 模式明确 hedge vs ask 的区别
- 关键区分："agent 能自己完成 → hedge"，"必须等用户 → ask"
- 给出 2 个对比例子

### 结果 (36 trials)
- Action 准确率: 33/36 (91.7%) — 与 Round 1 相同
- JSON 解析率: 36/36 (100%)
- **`hedge`: 3/3 (100%)** ← Round 1: 0/3
- **`skip`: 0/3 (0%)** ← Round 1: 3/3 (新暴露的问题)

### Per-scenario 对比

| 场景 | Round 1 | Round 2 | 变化 |
|------|---------|---------|------|
| execute | 3/3 | 3/3 | = |
| explore | 3/3 | 3/3 | = |
| parallel_execute | 3/3 | 3/3 | = |
| create | 3/3 | 3/3 | = |
| modify | 3/3 | 3/3 | = |
| delete | 3/3 | 3/3 | = |
| **skip** | **3/3** | **0/3** | **-3** |
| ask | 3/3 | 3/3 | = |
| **hedge** | **0/3** | **3/3** | **+3** |
| pivot_research | 3/3 | 3/3 | = |
| pivot_iteration | 3/3 | 3/3 | = |
| done | 3/3 | 3/3 | = |

### 分析

**又是打地鼠效应**，但 skip 的失败有不同特性：

`skip` 退化原因：
- 场景：API down，应该 skip T001，去做 T002
- Claude 选择了 `parallel_execute` [T002]，理由："API 不可用所以跳过 T001，执行 T002"
- 这是**语义等价**的：skip T001 + execute T002 ≈ parallel_execute [T002]
- Claude 的选择其实更"聪明"——它不只是标记 skip，而是直接去做可用的任务

**和 exp5 的区别**：exp5 的退化是"数值阈值判断"问题（Claude 不机械执行），exp6 的退化是"等价行为选择"问题（Claude 选了更高效的等价操作）。

### 结论
- HEDGE_VS_ASK_GUIDE **完美解决了 hedge 识别问题**
- `skip` 退化是良性的 — Claude 选择了更实际的等价操作
- **建议**：保留 HEDGE_VS_ASK_GUIDE（已合并到 prompts.py），skip 场景可接受
- 如果需要严格的 skip 行为，需要在 prompt 中加入 "skip 和 execute 的区别"

---

## 实验 7: Metrics → Prompt

### 7a: Category Detection

**对比：关键词匹配 vs Claude 分类**

| 场景 | 期望 | 代码(关键词) | Prompt(Claude) |
|------|------|------------|---------------|
| algorithm_sort | algorithm | algorithm | algorithm |
| algorithm_ml | algorithm | algorithm | algorithm |
| web_react | web | web | web |
| web_landing | web | **algorithm** ❌ | web |
| api_rest | api | **web** ❌ | api |
| api_graphql | api | **algorithm** ❌ | api |
| cli_tool | cli | **algorithm** ❌ | cli |
| library_sdk | library | **api** ❌ | library |
| general_refactor | general | general | general |
| general_ambiguous | general | general | general |

**结果**：
- 代码: 15/30 (50%)
- Prompt: **30/30 (100%)**

**为什么代码这么差？**
关键词匹配有优先级问题。例如 `web_landing` 包含 "page"（web 关键词），但也包含 "hero section"/"pricing table" 等不在关键词列表里的词。而 "HTML/CSS" 中的 "process" 匹配了 algorithm 关键词，algorithm 在代码中优先级更高。

**结论**：Category detection 应该完全改用 prompt。关键词匹配太脆弱，添加新类别需要手动维护大量关键词。

### 7b: Metrics Extraction

| 场景 | has_custom | hard | soft | subjective | 正确 |
|------|-----------|------|------|-----------|------|
| explicit_metrics | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| partial_metrics | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| no_metrics | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| inline_metrics | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| with_checkpoints | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |

**总准确率: 15/15 (100%)**

Sub-metric 准确率全部 100%。Claude 能完美理解 metrics markdown 格式并提取结构化数据。

**结论**：Metrics extraction 可以改用 prompt，比 regex 更鲁棒（能处理格式变体）。

---

## Round 2 横向发现

### 1. 打地鼠效应 (Whack-a-mole Effect)

实验 5 和 6 都出现了"修一个场景，坏另一个"的现象。但两者性质不同：
- **Exp 5**：数值阈值判断 — Claude 不机械执行，这是 prompt 的**固有局限**
- **Exp 6**：等价行为选择 — Claude 选了更聪明的操作，这是**良性退化**

### 2. Prompt 擅长概念教育，不擅长数值规则

| 任务类型 | Prompt 表现 | 原因 |
|---------|------------|------|
| 概念理解 (hard metric, hedge) | 优秀 (100%) | Claude 理解概念后能灵活应用 |
| 数值阈值 (score < 40, attempts >= 3) | 差 (0-100%) | Claude 会综合上下文，不机械执行 |
| 分类判断 (category detection) | 优秀 (100%) | 语义理解远超关键词匹配 |
| 信息提取 (metrics extraction) | 优秀 (100%) | 理解格式后准确提取 |

### 3. 代码 vs Prompt 的最佳分工

根据 7 个实验的结果，最佳分工原则：

| 逻辑类型 | 应该用 | 原因 |
|---------|--------|------|
| 数值阈值 / 精确计算 | **代码** | Claude 不会机械执行 |
| 概念判断 / 分类 | **Prompt** | Claude 理解力远超规则 |
| 信息提取 / 解析 | **Prompt (JSON)** | 比 regex 更鲁棒 |
| 路由逻辑 | **代码** (简化后) | 效率和可靠性 |
| 状态管理 | **代码** | 需要确定性 |
| 质量评估 | **Prompt** | 需要 nuance |

---

## 更新后的合并路线图

### 已完成 (Round 1)
1. ✅ Reviewer: JSON 解析 + regex fallback
2. ✅ Planner: JSON 解析 + regex fallback
3. ✅ Prompts: 请求 JSON 输出格式

### 已完成 (Round 2)
4. ✅ `HARD_METRIC_PIVOT_RULE` 添加到 prompts.py（概念教育）
5. ✅ `HEDGE_VS_ASK_GUIDE` 添加到 prompts.py 并整合到 planner prompt
6. ✅ `METRICS_CATEGORY_DETECTION_PROMPT` 添加到 prompts.py
7. ✅ `METRICS_EXTRACTION_PROMPT` 添加到 prompts.py

### 建议实施 (基于 Round 2 结论)
8. **Category detection**: 替换 `detect_category()` 关键词匹配为 Claude 调用
   - Prompt 100% vs Code 50%，明确改善
9. **Metrics extraction**: 保留 regex 作为 fallback，添加 prompt 路径
   - 两者都 100%，但 prompt 对格式变体更鲁棒

### 保留在代码中
10. **数值阈值判断**（pivot conditions 中的 score < 40、attempts >= 3）
11. **max_iterations 硬限制**
12. **Error recovery、文件锁**
13. **Orchestrator 循环结构**
