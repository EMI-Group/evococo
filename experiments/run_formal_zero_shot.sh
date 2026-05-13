#!/bin/bash
cd /home/zhenyu/evocoder

# Create output directory
mkdir -p experiments/baselines

echo "Starting formal zero-shot (strong) baseline experiment with 5 repeats inside 'tensor' conda environment..."

# Extract LITELLM_API_KEY from .env
export OPENAI_API_KEY=$(grep LITELLM_API_KEY .env | cut -d '=' -f2)
export OPENAI_BASE_URL="https://litellm.975738.xyz/v1"
export OPENAI_MODEL="gemini/gemini-3-flash-preview"

# Use conda run to ensure it executes within the correct environment
nohup conda run --no-capture-output -n tensor python -u experiments/batch_zero_shot.py \
    --input_dir experiments/formal_matlab_algorithms \
    --output_dir experiments/baselines \
    --repeats 5 \
    --mode strong > experiments/formal_zero_shot.log 2>&1 &

PID=$!
echo "Zero-shot experiment is running in the background (PID: $PID)."
echo "You can check the progress by running: tail -f experiments/formal_zero_shot.log"
