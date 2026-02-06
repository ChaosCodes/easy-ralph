# Ablation Study v2: Architecture Improvements

This directory contains experiments to validate the architecture improvements:
1. Clarifier v2 (explore+propose vs ask)
2. Pivot mechanism (three triggers)
3. Reviewer/Evaluator merge
4. Agent autonomous judgment

## Directory Structure

```
ablation_v2/
├── README.md           # This file
├── tasks/              # Test task definitions
│   ├── task1_feature.md
│   ├── task2_optimization.md
│   ├── task3_research.md
│   └── task4_idea.md
├── results/            # Experiment results
├── run_experiment1.py  # Clarifier mode comparison
├── run_experiment2.py  # Pivot mechanism value
├── run_experiment3.py  # Reviewer vs Evaluator
├── run_experiment4.py  # Autonomous judgment
└── analyze_results.py  # Results analysis
```

## Experiments

### Experiment 1: Clarifier Mode Comparison
**Hypothesis**: Explore+propose mode helps users clarify vague requirements better than ask mode.

**Test Task**: Research-type (task3 or task4)
- A: Clarifier v1 (ask mode)
- B: Clarifier v2 (explore+propose mode)

**Metrics**:
- User satisfaction with chosen approach (1-5)
- Clarity of goal.md (AI evaluation)
- Success rate of subsequent execution

### Experiment 2: Pivot Mechanism Value
**Hypothesis**: Three pivot triggers reduce wasted effort on dead-end approaches.

**Test Task**: Designed to have "first path doesn't work"
- A: No pivot (keep trying same approach)
- B: With pivot mechanism

**Metrics**:
- Time to find viable solution
- Total token consumption
- Final solution quality

### Experiment 3: Reviewer vs Combined Evaluator
**Hypothesis**: Merging Reviewer and Evaluator doesn't reduce quality.

**Test Task**: Standard IMPLEMENT tasks
- A: Separate (Worker → Reviewer → Evaluator)
- B: Combined (Worker → CombinedEvaluator)

**Metrics**:
- Issue discovery rate (real issues found)
- False positive rate (non-issues reported)
- Total latency (time + tokens)

### Experiment 4: Autonomous Judgment vs Wait
**Hypothesis**: Agent self-judgment + notification is more efficient than waiting for user.

**Test Task**: Multi-iteration optimization
- A: Wait for user confirmation each round
- B: Agent judges autonomously, notifies but continues

**Metrics**:
- Wall clock time to completion
- User intervention count
- Final quality

## Running Experiments

```bash
# Run individual experiment
python experiments/ablation_v2/run_experiment1.py

# Run all experiments
python experiments/ablation_v2/run_all.py

# Analyze results
python experiments/ablation_v2/analyze_results.py
```
