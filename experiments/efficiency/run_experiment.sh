#!/bin/bash
# Run a single efficiency experiment
#
# Usage: ./run_experiment.sh <exp_name> <branch_name>
# Example: ./run_experiment.sh eval_exhaustive exp/eval-exhaustive
#
# Prerequisites:
# - Branch must exist with experiment code changes
# - ralph-sdk must be installed in editable mode

set -euo pipefail

EXP_NAME="${1:?Usage: $0 <exp_name> <branch_name>}"
BRANCH="${2:?Usage: $0 <exp_name> <branch_name>}"
TARGET_SCORE="${3:-95}"

EXP_DIR="/tmp/timeparser_exp_${EXP_NAME}"
BASELINE_GOAL="/tmp/timeparser_v5/.ralph/goal.md"
RESULTS_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Efficiency Experiment: ${EXP_NAME} ==="
echo "Branch: ${BRANCH}"
echo "Dir: ${EXP_DIR}"
echo "Target: ${TARGET_SCORE}"
echo ""

# 1. Switch to experiment branch
echo "[1/5] Switching to branch ${BRANCH}..."
cd /Users/chaos/Desktop/ralph-exp/ralph-sdk
git checkout "${BRANCH}"

# 2. Reinstall ralph-sdk in editable mode
echo "[2/5] Reinstalling ralph-sdk..."
pip install -e . -q 2>/dev/null

# 3. Prepare clean experiment directory
echo "[3/5] Preparing experiment directory..."
rm -rf "${EXP_DIR}"
mkdir -p "${EXP_DIR}"

# Copy baseline goal.md into .ralph/
mkdir -p "${EXP_DIR}/.ralph"
cp "${BASELINE_GOAL}" "${EXP_DIR}/.ralph/goal.md"

# 4. Run ralph-sdk
echo "[4/5] Running ralph-sdk..."
START_TIME=$(date +%s)

ralph-sdk run "placeholder - will use existing goal.md" \
    --cwd "${EXP_DIR}" \
    --quick \
    --target "${TARGET_SCORE}" \
    2>&1 | tee "${RESULTS_DIR}/exp_${EXP_NAME}_output.log"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))

echo ""
echo "[5/5] Experiment complete!"
echo "Duration: ${ELAPSED_MIN} minutes (${ELAPSED} seconds)"

# 5. Collect results
echo ""
echo "=== Results Summary ==="
echo "Experiment: ${EXP_NAME}"
echo "Duration: ${ELAPSED_MIN} min"

# Check if session.jsonl exists for cost extraction
if [ -f "${EXP_DIR}/.ralph/logs/session.jsonl" ]; then
    echo "Session log: ${EXP_DIR}/.ralph/logs/session.jsonl"
fi

# Check test results if they exist
if [ -d "${EXP_DIR}/tests" ]; then
    echo ""
    echo "--- Test Results ---"
    cd "${EXP_DIR}"
    PYTHONPATH=src python3 -m pytest tests/ -q --tb=no 2>/dev/null || true
    echo ""
    echo "--- Coverage ---"
    PYTHONPATH=src python3 -m pytest tests/ --cov=timeparser --cov-report=term-missing -q --tb=no 2>/dev/null | tail -5 || true
fi

echo ""
echo "Full output: ${RESULTS_DIR}/exp_${EXP_NAME}_output.log"
echo "Workspace: ${EXP_DIR}"

# Switch back to pro branch
cd /Users/chaos/Desktop/ralph-exp/ralph-sdk
git checkout pro
