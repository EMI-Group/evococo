#!/bin/bash
cd /home/zhenyu/evocoder

# Create output directory
mkdir -p experiments/formal_experiment_results

echo "Starting formal generation experiment with 5 repeats inside 'tensor' conda environment..."

# Use conda run to ensure it executes within the correct environment
nohup conda run --no-capture-output -n tensor python experiments/batch_translate.py \
    --input_dir experiments/formal_matlab_algorithms \
    --output_dir experiments/formal_experiment_results \
    --repeats 5 > experiments/formal_experiment.log 2>&1 &

PID=$!
echo "Experiment is running in the background (PID: $PID)."
echo "You can check the progress by running: tail -f experiments/formal_experiment.log"
