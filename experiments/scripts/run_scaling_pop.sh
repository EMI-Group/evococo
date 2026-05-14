#!/bin/bash
# ==============================================================================
# Population Scaling Benchmark Script (Time vs PopSize)
#
# QUICK START - To run a specific chunk on a specific GPU, execute:
# ./experiments/scripts/run_scaling_pop.sh <GPU_ID> <CHUNK_IDX>
# 
# COPY & PASTE COMMANDS (Launch 5 chunks across 5 GPUs):
# nohup ./experiments/scripts/run_scaling_pop.sh 0 0 > nohup_scaling_pop_gpu0_chunk0.log 2>&1 &
# nohup ./experiments/scripts/run_scaling_pop.sh 1 1 > nohup_scaling_pop_gpu1_chunk1.log 2>&1 &
# nohup ./experiments/scripts/run_scaling_pop.sh 2 2 > nohup_scaling_pop_gpu2_chunk2.log 2>&1 &
# nohup ./experiments/scripts/run_scaling_pop.sh 3 3 > nohup_scaling_pop_gpu3_chunk3.log 2>&1 &
# nohup ./experiments/scripts/run_scaling_pop.sh 4 4 > nohup_scaling_pop_gpu4_chunk4.log 2>&1 &
# ==============================================================================

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "❌ Error: Missing arguments."
    echo "Usage: $0 <GPU_ID> <CHUNK_IDX>"
    echo "Example: $0 3 0"
    exit 1
fi

GPU_ID=$1
CHUNK_IDX=$2
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Using GPU: $GPU_ID for Chunk: $CHUNK_IDX"

# Use relative paths so it works anywhere
SCRIPT_DIR=$(dirname "$0")

# Global Python environment on server
PYTHON_BIN="python"

# Benchmark Parameters
ALGO_DIR="${SCRIPT_DIR}/../formal_experiment_algorithms/best_of_5"
POP_SIZES="256 512 1024 2048 4096 8192 16384"
REPEATS=11
NUM_CHUNKS=5

echo "Starting Population Scaling Benchmark..."
echo "Population Sizes: $POP_SIZES"
echo "Repeats: $REPEATS"
echo "------------------------------------------------------------"
echo "🚀 Launching Chunk $CHUNK_IDX on GPU $GPU_ID..."

$PYTHON_BIN "${SCRIPT_DIR}/../benchmark_results_final/benchmark_scaling.py" \
    --algo_dir "$ALGO_DIR" \
    --chunk_idx $CHUNK_IDX \
    --num_chunks $NUM_CHUNKS \
    --pop_sizes $POP_SIZES \
    --repeats $REPEATS

echo "------------------------------------------------------------"
echo "✅ Chunk $CHUNK_IDX completed!"
