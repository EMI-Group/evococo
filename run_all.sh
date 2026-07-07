#!/bin/bash

# ==============================================================================
# EvoCoCo Full Batch Translation Script
# ==============================================================================

# Activate the conda environment (optional, depending on how you run this script)
# source /home/zhenyu/miniconda3/bin/activate tensor

# Change to project root
cd /home/zhenyu/evocoder || exit 1


INPUT_DIR="experiments/matlab_code"
OUTPUT_DIR="experiments/benchmark_results_final"
REPEATS=1
LOG_FILE="batch_translate_all.log"

echo "==============================================================="
echo "Starting EvoCoCo Full Batch Translation..."
echo "Input Directory:  $INPUT_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Repeats:          $REPEATS"
echo "Log File:         $LOG_FILE"
echo "==============================================================="

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Run the batch translation script in the background using nohup
PYTHONPATH=/home/zhenyu/evocoder nohup /home/zhenyu/miniconda3/envs/tensor/bin/python experiments/batch_translate.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --repeats $REPEATS > "$LOG_FILE" 2>&1 &

PID=$!
echo "Task launched in background!"
echo "Background PID: $PID"
echo "To monitor progress, run: tail -f $LOG_FILE"
echo "==============================================================="
