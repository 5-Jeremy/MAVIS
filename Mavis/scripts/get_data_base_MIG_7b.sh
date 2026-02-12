#!/bin/bash

# This script should be run from within the base directory of the MAVIS code. The script assumes that you
# have 4 H100 GPUs on a single node, and uses NVIDIA Multi-Instance GPU to partition the GPUs for maximum
# utilization when running get_data_base.py with a 7B generative model. It will not work with a 13B model
# since the amount of memory per MIG instance is insufficient, and likewise it will probably not work with
# the safeRLHF dataset due to its larger sequence lengths. The base model type should be set in 
# utils/gen_utils.py within default_defaults_dict before running this script.

# IMPORTANT: Make sure you source your conda environment before running this script. The script requires
# root priviledges at certain points, but it should not be run with sudo. Instead, you will be prompted
# for your password when needed. You will also need to enable MIG on all GPUs before running this script:
# sudo nvidia-smi -i 0,1,2,3 -mig 1

# The number of prompt to generate data for should be a multiple of 16 to ensure they can be evenly split

# Script arguments:
#     $1: The dataset (anthropic or summary)
#     $2: The number of prompts to generate data for (will be split between subprocesses).
#     $3: The output directory.

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <dataset> <num_prompts> <output_dir>"
    echo "Example: $0 anthropic 6000 Training_Output/anthropic/unlabeled_data/"
    exit 1
fi

# Confirm that there are 4 GPUs and each has at least 80 GB of memory
num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$num_gpus" -ne 4 ]; then
    echo "Error: This script assumes exactly 4 GPUs."
    exit 1
fi
min_memory=80000
for gpu_id in $(seq 0 3); do
    mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $gpu_id)
    if [ "$mem" -lt "$min_memory" ]; then
        echo "Error: This script requires each GPU to have at least 80 GB of memory."
        exit 1
    fi
done

# Determine the number of prompts per process (in this case, it is 1/16 of the total number)
PROMPTS_PER_PROCESS=$(($2 / 16))

# Check that MIG is enabled for all GPUs
# To enable, run "sudo nvidia-smi -i <GPU IDs> -mig 1"
if nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader | grep -qv Enabled; then
    echo "Error: MIG is not enabled on all GPUs."
    exit 1
fi

# Split each GPU into 4 instances; note that elevated priviledges are required
# We want profile ID 15, corresponding to 1g.24gb
MIG_CONFIG="15,15,15,15"
for gpu_id in $(seq 0 3); do
    sudo nvidia-smi mig -i $gpu_id -cgi $MIG_CONFIG -C
done

echo "Beginning data collection for $2 prompts with dataset $1 into directory $3"
date

# Get the list of MIG instances
MIG_INSTANCES=$(sudo nvidia-smi mig -lgi)
echo "$MIG_INSTANCES"

# Data collection
count=0
for uuid in $(nvidia-smi -L | grep "UUID: MIG" | awk -F'UUID: ' '{print $2}' | tr -d ')'); do
    mkdir -p "$3/process_$count"
    CUDA_VISIBLE_DEVICES=$uuid python get_data_base.py $(($count*$PROMPTS_PER_PROCESS)) $PROMPTS_PER_PROCESS --dataset "$1" --output_dir "$3/process_$count/" &>> "$3/process_$count/log.txt" &
    count=$(($count + 1))
done
wait

date

# Remove all created MIG instances and disable MIG mode
echo "Cleaning up MIG instances and disabling MIG mode..."
sudo nvidia-smi mig -dci && sudo nvidia-smi mig -dgi && sudo nvidia-smi -i 0,1,2,3 -mig 0