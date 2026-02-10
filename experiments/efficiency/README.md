# Efficiency Experiments

Experiments to reduce wasted iterations in Ralph SDK task execution.

## Baseline
- Source: `/tmp/timeparser_v5`
- 7 iterations (1 EXPLORE + 5 EXECUTE + 1 DONE)
- 64 min, $23, score 95/100
- 361 tests, 93.45% coverage

## Branches

| Branch | Experiment | Files Modified |
|--------|-----------|----------------|
| `exp/eval-exhaustive` | Exp 1: Evaluator lists ALL issues in one pass | `evaluator.py` |
| `exp/skip-reviewer` | Exp 2: Skip reviewer in cosmetic fix rounds | `orchestrator.py` |
| `exp/auto-planner` | Exp 3: Auto-continue planner when predictable | `orchestrator.py` |
| `exp/skip-adversarial` | Exp 4: Skip adversarial after 2 clean rounds | `evaluator.py`, `orchestrator.py` |
| `exp/no-test-task` | Exp 5: No separate test task from initializer | `prompts.py` |
| `exp/combined` | Exp 6: All improvements merged | all above |

## Running Experiments

```bash
# Single experiment
./experiments/efficiency/run_experiment.sh eval_exhaustive exp/eval-exhaustive

# All experiments
for exp in eval_exhaustive skip_reviewer auto_planner skip_adversarial no_test_task combined; do
    branch="exp/$(echo $exp | tr '_' '-')"
    ./experiments/efficiency/run_experiment.sh "$exp" "$branch"
done
```

## Verification Criteria
- Final score >= 93
- All 6 success metrics PASS
- Test count >= 300
- Coverage >= 90%
- No functional bugs missed
