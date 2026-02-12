


#!/bin/bash

# Usage: ./make_RSoup.sh <dataset> <model_size> <model1_obj> <model2_obj> <pref1> <pref2>
# Example: ./make_RSoup.sh anthropic 7b help humor 0.2 0.8

if [ "$#" -ne 6 ]; then
	echo "Usage: $0 <dataset> <model_size> <model1_obj> <model2_obj> <pref1> <pref2>"
	exit 1
fi

DATASET=$1
MODEL_SIZE=$2
MODEL1_OBJ=$3
MODEL2_OBJ=$4
PREF1=$5
PREF2=$6

SFT_MODEL="Models/sft_model/llama_${MODEL_SIZE}/${DATASET}/"
MODEL1="Models/morlhf/llama_${MODEL_SIZE}/${DATASET}/single/${MODEL1_OBJ}"
MODEL2="Models/morlhf/llama_${MODEL_SIZE}/${DATASET}/single/${MODEL2_OBJ}"
OUTPUT="Models/reward_soup/llama_${MODEL_SIZE}/${DATASET}/${MODEL1_OBJ}_${MODEL2_OBJ}/reward_soup_${PREF1}"
python ../Finetuning/reward_soup_merge.py \
	--sft_model="$SFT_MODEL" \
	--model1="$MODEL1" \
	--model2="$MODEL2" \
	--pref1 $PREF1 \
	--pref2 $PREF2 \
	--output="$OUTPUT"