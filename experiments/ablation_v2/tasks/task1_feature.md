# Task 1: Feature Direction - Smart Task Prioritization

## User Request
"给 ralph-sdk 加个智能任务优先级排序"

## Ambiguity Points
- 什么算法？
- 基于什么排序？
- 优先级的定义是什么？

## Expected Clarifier Behavior

### Ask Mode (v1)
Should ask questions like:
- "你希望基于什么因素排序？A. 任务依赖关系 B. 预估时间 C. 重要性标签"
- "排序是否需要实时更新？"

### Explore+Propose Mode (v2)
Should research and propose:
- 方案 A: 基于依赖关系的拓扑排序
- 方案 B: 基于任务类型的优先级规则
- 方案 C: 机器学习预测任务重要性

## Success Criteria
- Goal.md clearly specifies chosen approach
- Technical details are concrete enough to implement
- Non-goals are explicitly listed
