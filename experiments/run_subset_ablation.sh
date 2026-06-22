#!/bin/bash
# Exit on error
set -e

# Make sure we run from the project root directory
cd "$(dirname "$0")/.."

INPUT_DIR="./experiments/ablation_subset_input"
REPEATS=5
PYTHON_BIN="/home/zhenyu/miniconda3/envs/tensor/bin/python"
OUT_BASE="./experiments/ablation_subset_results"

echo "=========================================================="
echo "=== Starting EvoCoCo Subset Ablation (5 Repeats) ==="
echo "=========================================================="

# Helper to clean environment before each run
clear_env() {
    export EVOCOCO_NUM_BRANCHES=6
    export EVOCOCO_ABLATE_RAG=0
    export EVOCOCO_ABLATE_ARCHITECT=0
    export EVOCOCO_ABLATE_RUNTIME_FIXER=0
    export EVOCOCO_ABLATE_MULTI_BRANCH=0
}

# 1. w/o Architect
echo ""
echo ">>> Variant 1/4: w/o Architect (Ablating Stage 3) <<<"
clear_env
export EVOCOCO_ABLATE_ARCHITECT=1
$PYTHON_BIN experiments/batch_translate.py --input_dir $INPUT_DIR --output_dir $OUT_BASE/results_no_architect --repeats $REPEATS
$PYTHON_BIN evaluation/benchmark.py --dir $OUT_BASE/results_no_architect

# 2. w/o Runtime Fixer
echo ""
echo ">>> Variant 2/4: w/o Runtime Fixer (Ablating Stage 6) <<<"
clear_env
export EVOCOCO_ABLATE_RUNTIME_FIXER=1
$PYTHON_BIN experiments/batch_translate.py --input_dir $INPUT_DIR --output_dir $OUT_BASE/results_no_runtime_fixer --repeats $REPEATS
$PYTHON_BIN evaluation/benchmark.py --dir $OUT_BASE/results_no_runtime_fixer

# 3. w/o RAG Rules
echo ""
echo ">>> Variant 3/4: w/o RAG Rules (Ablating Stage 2) <<<"
clear_env
export EVOCOCO_ABLATE_RAG=1
$PYTHON_BIN experiments/batch_translate.py --input_dir $INPUT_DIR --output_dir $OUT_BASE/results_no_rag --repeats $REPEATS
$PYTHON_BIN evaluation/benchmark.py --dir $OUT_BASE/results_no_rag

# 4. w/o Multi-Branch Selection
echo ""
echo ">>> Variant 4/4: w/o Multi-Branch Selection (1 Branch) <<<"
clear_env
export EVOCOCO_ABLATE_MULTI_BRANCH=1
$PYTHON_BIN experiments/batch_translate.py --input_dir $INPUT_DIR --output_dir $OUT_BASE/results_single_branch --repeats $REPEATS
$PYTHON_BIN evaluation/benchmark.py --dir $OUT_BASE/results_single_branch

echo ""
echo "====================================================="
echo "=== Ablation Translation & Benchmarking Complete! ==="
echo "====================================================="
echo "You can now run: $PYTHON_BIN experiments/plot_subset_ablation.py to generate plots and tables."
