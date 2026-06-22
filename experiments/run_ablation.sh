#!/bin/bash
# Exit on error
set -e

# Make sure we run from the project root directory
cd "$(dirname "$0")/.."

INPUT_DIR="./experiments/matlab_code"
REPEATS=3
PYTHON_BIN="/home/zhenyu/miniconda3/envs/tensor/bin/python"

echo "========================================="
echo "=== Starting EvoCoCo Ablation Suite ==="
echo "========================================="

# Helper to clean environment before each run
clear_env() {
    export EVOCOCO_NUM_BRANCHES=6
    export EVOCOCO_ABLATE_RAG=0
    export EVOCOCO_ABLATE_ARCHITECT=0
    export EVOCOCO_ABLATE_RUNTIME_FIXER=0
    export EVOCOCO_ABLATE_MULTI_BRANCH=0
}

# 1. Full EvoCoCo (Full System)
echo ""
echo ">>> Variant 1/5: Full EvoCoCo (Full System) <<<"
clear_env
$PYTHON_BIN experiments/batch_translate.py --input_dir $INPUT_DIR --output_dir ./experiments/results_full --repeats $REPEATS
$PYTHON_BIN evaluation/benchmark.py --dir ./experiments/results_full

# 2. w/o Architect
echo ""
echo ">>> Variant 2/5: w/o Architect (Ablating Stage 3) <<<"
clear_env
export EVOCOCO_ABLATE_ARCHITECT=1
$PYTHON_BIN experiments/batch_translate.py --input_dir $INPUT_DIR --output_dir ./experiments/results_no_architect --repeats $REPEATS
$PYTHON_BIN evaluation/benchmark.py --dir ./experiments/results_no_architect

# 3. w/o Runtime Fixer
echo ""
echo ">>> Variant 3/5: w/o Runtime Fixer (Ablating Stage 6) <<<"
clear_env
export EVOCOCO_ABLATE_RUNTIME_FIXER=1
$PYTHON_BIN experiments/batch_translate.py --input_dir $INPUT_DIR --output_dir ./experiments/results_no_runtime_fixer --repeats $REPEATS
$PYTHON_BIN evaluation/benchmark.py --dir ./experiments/results_no_runtime_fixer

# 4. w/o RAG Rules
echo ""
echo ">>> Variant 4/5: w/o RAG Rules (Ablating Stage 2) <<<"
clear_env
export EVOCOCO_ABLATE_RAG=1
$PYTHON_BIN experiments/batch_translate.py --input_dir $INPUT_DIR --output_dir ./experiments/results_no_rag --repeats $REPEATS
$PYTHON_BIN evaluation/benchmark.py --dir ./experiments/results_no_rag

# 5. w/o Multi-Branch Selection
echo ""
echo ">>> Variant 5/5: w/o Multi-Branch Selection (1 Branch) <<<"
clear_env
export EVOCOCO_ABLATE_MULTI_BRANCH=1
$PYTHON_BIN experiments/batch_translate.py --input_dir $INPUT_DIR --output_dir ./experiments/results_single_branch --repeats $REPEATS
$PYTHON_BIN evaluation/benchmark.py --dir ./experiments/results_single_branch

echo ""
echo "====================================================="
echo "=== Ablation Translation & Benchmarking Complete! ==="
echo "====================================================="
echo "You can now run: $PYTHON_BIN experiments/plot_ablation.py to generate plots and tables."
