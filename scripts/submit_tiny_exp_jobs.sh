#!/bin/bash

# Script to submit SLURM jobs for all configs in configs/tiny_exp/
# with random sleep intervals between submissions

CONFIG_DIR="configs/tiny_exp"
SCRIPT_PATH="scripts/short_ctx/gemma_qa_short_ctx_exp_default.sh"

# Check if config directory exists
if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: Config directory $CONFIG_DIR does not exist"
    exit 1
fi

# Check if script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Script $SCRIPT_PATH does not exist"
    exit 1
fi

# Get all yaml config files
configs=($(find "$CONFIG_DIR" -name "*.yaml" -type f))

if [ ${#configs[@]} -eq 0 ]; then
    echo "No .yaml config files found in $CONFIG_DIR"
    exit 1
fi

echo "Found ${#configs[@]} config files:"
for config in "${configs[@]}"; do
    echo "  - $config"
done

echo ""
echo "Starting job submissions..."

# Submit jobs with random sleep intervals
for config in "${configs[@]}"; do
    echo "Submitting job for config: $config"
    sbatch "$SCRIPT_PATH" "$config"

    # Random sleep between 3-30 seconds
    sleep_time=$((3 + RANDOM % 28))
    echo "Waiting $sleep_time seconds before next submission..."
    sleep $sleep_time
done

echo "All jobs submitted!"