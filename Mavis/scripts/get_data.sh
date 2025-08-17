#!/bin/bash

# This script should be run from within the base directory of the MAVIS code containing the Anthropic and/or Summary directories.

# Script arguments:
#     $1: The iteration number (0 if you are training a brand new value model for the reference model).
#     $2: The number of prompts to generate data for.
#     $3: The output directory.
#     $4: The number of topk tokens.
#     $5: The name of the objective.
#     $6: The value of beta (not required if iteration number is 0).

if [ "$#" -lt 5 ]; then
    echo "Usage: $0 <iteration_number> <num_prompts> <output_dir> <topk> <objective> [beta] [value_model_dir]"
    echo "Example: $0 1 6000 k=60/ 60 help 4 k=60/value_models/"
fi

if [ "$5" == "help" ] || [ "$5" == "harm" ] || [ "$5" == "humor" ]; then
    DATASET="anthropic"
elif [ "$5" == "summarization" ] || [ "$5" == "faithful" ]; then
    DATASET="summary"
else
    echo "Unknown objective: $5"
    echo "Please specify a valid objective: help, harm, humor, summarization, or faithful."
fi

# If iteration number is 0, use get_data_random_branching.py
if [ "$1" -eq 0 ]; then
    python get_data_base.py 0 "$2" --output_dir "$3" --topk "$4" --dataset "$DATASET"
    if [ "$#" -lt 5 ]; then
        echo "No objective specified, defaulting to 'all'."
        LABELING_OBJECTIVE="all"
    else
        LABELING_OBJECTIVE="$5"
    fi
else
    VALUE_MODEL_DIR="value_models"
    # The iteration of the value model to use for generation is the previous iteration
    ITER_TO_LOAD="$(( $1 - 1 ))"
    python get_data_iter.py 0 "$2" "$5" --value_model_dir $VALUE_MODEL_DIR --value_model_iter $ITER_TO_LOAD --output_dir "$3" --topk "$4" --beta "$6" --dynamic_splitting
    LABELING_OBJECTIVE="$5"
fi

if [ "$1" -eq 0 ]; then
    python label_tree.py "$3/" $LABELING_OBJECTIVE --check_trees --dataset "$DATASET"
else
    python label_tree_soft.py "$3/" $LABELING_OBJECTIVE --check_trees --dataset "$DATASET"
fi
