#!/bin/bash
# ==============================================================================
# Formal Benchmark Script: 50 Algorithms, 7 Problems, 21 Seeds
#
# QUICK START - To run a specific chunk on a specific GPU, execute:
# ./experiments/scripts/run_formal_benchmark.sh <GPU_ID> <CHUNK_IDX>
# 
# COPY & PASTE COMMANDS (Launch 5 chunks across 5 GPUs):
# nohup ./experiments/scripts/run_formal_benchmark.sh 0 0 > nohup_formal_gpu0_chunk0.log 2>&1 &
# nohup ./experiments/scripts/run_formal_benchmark.sh 1 1 > nohup_formal_gpu1_chunk1.log 2>&1 &
# nohup ./experiments/scripts/run_formal_benchmark.sh 2 2 > nohup_formal_gpu2_chunk2.log 2>&1 &
# nohup ./experiments/scripts/run_formal_benchmark.sh 3 3 > nohup_formal_gpu3_chunk3.log 2>&1 &
# nohup ./experiments/scripts/run_formal_benchmark.sh 4 4 > nohup_formal_gpu4_chunk4.log 2>&1 &
# ==============================================================================

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "❌ Error: Missing arguments."
    echo "Usage: $0 <GPU_ID> <CHUNK_IDX>"
    echo "Example: $0 0 0"
    exit 1
fi

GPU_ID=$1
CHUNK_IDX=$2
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Using GPU: $GPU_ID for Chunk: $CHUNK_IDX"

# Paths configured for server
SCRIPT_DIR=$(dirname "$0")
ALGO_DIR="${SCRIPT_DIR}/../formal_experiment_algorithms/best_of_5"
OUTPUT_FILE="dtlz_formal_results_chunk${CHUNK_IDX}.csv"

# Global Python environment on server
PYTHON_BIN="python"

# 7 Problems
PROBLEMS=("DTLZ1" "DTLZ2" "DTLZ3" "DTLZ4" "DTLZ5" "DTLZ6" "DTLZ7")

# 21 Independent Runs
SEEDS=($(seq 1 21))
NUM_CHUNKS=5

# Get all algo files and chunk them
shopt -s nullglob
ALGO_FILES=("$ALGO_DIR"/*.py)
NUM_ALGOS=${#ALGO_FILES[@]}

if [ $NUM_ALGOS -eq 0 ]; then
    echo "No algorithms found in $ALGO_DIR!"
    exit 1
fi

CHUNK_SIZE=$(( (NUM_ALGOS + NUM_CHUNKS - 1) / NUM_CHUNKS ))
START_IDX=$(( CHUNK_IDX * CHUNK_SIZE ))
CHUNK_FILES=("${ALGO_FILES[@]:$START_IDX:$CHUNK_SIZE}")

echo "Starting Formal Benchmark on DTLZ problems (Chunk $CHUNK_IDX)..."
echo "Processing ${#CHUNK_FILES[@]} algorithms."
echo "Results will be appended to $OUTPUT_FILE in the current directory."

for ALGO_PATH in "${CHUNK_FILES[@]}"; do
    algo_file=$(basename "$ALGO_PATH")
    algo_name="${algo_file%.*}"
    
    for prob in "${PROBLEMS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            
            # Resume feature
            if [ -f "$OUTPUT_FILE" ] && grep -q "^${algo_name},${prob},${seed}," "$OUTPUT_FILE"; then
                echo "⏭️ Skipping completed: $algo_name | $prob | Seed $seed"
                continue
            fi
            
            $PYTHON_BIN "${SCRIPT_DIR}/../../evaluation/run_dtlz_benchmark.py" \
                --algo_file "$ALGO_PATH" \
                --problem "$prob" \
                --seed "$seed" \
                --output "$OUTPUT_FILE"
        done
    done
done

echo "Formal benchmark Chunk $CHUNK_IDX completed!"
