#!/bin/bash

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <num_response_samples>"
    exit 1
fi

NUM_SAMPLES="$1"

if ! [[ "$NUM_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: num_response_samples must be a positive integer."
    exit 1
fi

CONFIG_DIR="configs"

# Check if we are in the correct directory (configs and scripts should exist)
if [ ! -d "$CONFIG_DIR" ] || [ ! -d "scripts" ]; then
    echo "Error: Must be run from the root directory of the project (where 'configs' and 'scripts' exist)."
    exit 1
fi

for config_file in "$CONFIG_DIR"/*.json; do
    basename=$(basename "$config_file" .json)
    
    if [ "$basename" = "generation_config_reference" ]; then
        continue
    fi
    
    # Parse parameters from the filename
    # Format expected: {MODEL}_{DATASET}_l{LENGTH}_s{STEPS}_{REMASKING}
    if [[ "$basename" =~ ^([^_]+)_(.*)_l([0-9]+)_s([0-9]+)_(lc|rd)$ ]]; then
        MODEL="${BASH_REMATCH[1]}"
        DATASET="${BASH_REMATCH[2]}"
        LENGTH="${BASH_REMATCH[3]}"
        STEPS="${BASH_REMATCH[4]}"
        REMASKING="${BASH_REMATCH[5]}"
        
        echo "Submitting job for config: $config_file"
        sbatch scripts/run_pipeline.slurm "$MODEL" "$DATASET" "$NUM_SAMPLES" "$LENGTH" "$STEPS" "$REMASKING"
    else
        echo "Warning: Could not parse parameters from filename $basename. Skipping."
    fi
done

echo "Finished submitting jobs."
