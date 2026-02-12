#!/bin/bash

# This script should be run from within the base directory of the MAVIS code. The script assumes that you
# have 4 H100 GPUs on a single node, and uses NVIDIA Multi-Instance GPU to partition the GPUs for maximum
# utilization when running get_data_iter.py with a 7B or 13B generative model. The base model type should
# be set in utils/gen_utils.py within default_defaults_dict

# IMPORTANT: Make sure you enable MIG on all GPUs before running this. You can do this by running:
#     sudo nvidia-smi -i 0,1,2,3 -mig 1
# The script requires root priviledges at certain points to do other MIG operations, but it should not be 
# run with sudo. Instead, you will be prompted for your password when needed.

# The number of prompt to generate data for should be a multiple of 8 to ensure they can be evenly split

# Once this script completes, use the merge_hdf5_files function from utils/hdf5_utils.py to combine the
# <obj>_labeled.hdf5 files from the different processes into a single file for training.

# Script arguments:
#     $1: The objective (e.g. help, summarization, safeRLHF_help) - the dataset will be inferred from this
#     $2: The prompt index to start at
#     $3: The number of prompts to generate data for (will be split between subprocesses).
#     $4: The output directory.
#     $5: The value of beta
#     $6: The value model iteration to use (default: 0)

# For additional customization:
EXTRA_ARGS=""

if [ "$#" -lt 5 ]; then
    echo "Usage: $0 <dataset> <num_prompts> <output_dir> <beta> [value_model_iter]"
    echo "Example: $0 anthropic 6000 Training_Output/anthropic/unlabeled_data/ 1 0"
    exit 1
fi

if [ "$1" = "help" ]; then
	DATASET="anthropic"
elif [ "$1" = "harm" ]; then
	DATASET="anthropic"
elif [ "$1" = "humor" ]; then
	DATASET="anthropic"
elif [ "$1" = "summarization" ]; then
	DATASET="summary"
elif [ "$1" = "faithful" ]; then
	DATASET="summary"
elif [ "$1" = "safeRLHF_help" ]; then
    DATASET="safeRLHF"
elif [ "$1" = "safeRLHF_harm" ]; then
    DATASET="safeRLHF"
else
    echo "Error: OBJECTIVE must be one of 'help', 'harm', 'humor', 'summarization', 'faithful', 'safeRLHF_help', or 'safeRLHF_harm'"
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

# Determine the number of prompts per process (in this case, it is 1/8 of the total number)
PROMPTS_PER_PROCESS=$(($3 / 8))

# Check that MIG is enabled for all GPUs
# To enable, run "sudo nvidia-smi -i <GPU IDs> -mig 1"
if nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader | grep -qv Enabled; then
    echo "Error: MIG is not enabled on all GPUs."
    exit 1
fi

# Split each GPU into 2 instances (we cannot do 3 or 4 because more memory is required to load the value model)
# We want profile ID 9, corresponding to 3g.47gb
MIG_CONFIG="9,9"
for gpu_id in $(seq 0 3); do
    sudo nvidia-smi mig -i $gpu_id -cgi $MIG_CONFIG -C
done

echo "Beginning data collection for $3 prompts starting from prompt index $2 for objective $1 with beta $5 using value model iteration ${6:-0}..."
date

# Get the list of MIG instances
MIG_INSTANCES=$(sudo nvidia-smi mig -lgi)
echo "$MIG_INSTANCES"

# Data collection
count=0
for uuid in $(nvidia-smi -L | grep "UUID: MIG" | awk -F'UUID: ' '{print $2}' | tr -d ')'); do
    mkdir -p "$4/process_$count"
    CUDA_VISIBLE_DEVICES=$uuid python get_data_iter.py $(($2 + ($count*$PROMPTS_PER_PROCESS))) $PROMPTS_PER_PROCESS "$1" $EXTRA_ARGS --output_dir "$4/process_$count/" --beta $5 --value_model_iter ${6:-0} &>> "$4/process_$count/collect_log.txt" &
    count=$(($count + 1))
done
wait

# Data labeling
count=0
for uuid in $(nvidia-smi -L | grep "UUID: MIG" | awk -F'UUID: ' '{print $2}' | tr -d ')'); do
    CUDA_VISIBLE_DEVICES=$uuid python label_tree.py "$4/process_$count/all_data.hdf5" all --dataset=$DATASET --compute_KL --check_trees &>> "$4/process_$count/label_log.txt" &
    count=$(($count + 1))
done
wait

date

# Remove all created MIG instances and disable MIG mode
echo "Cleaning up MIG instances and disabling MIG mode..."
sudo nvidia-smi mig -dci && sudo nvidia-smi mig -dgi && sudo nvidia-smi -i 0,1,2,3 -mig 0
