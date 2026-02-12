#!/bin/bash

# This script should be run from within the base directory of the MAVIS code

# This script trains the iteration 0 value models for faithfulness and summarization with varying amounts 
# of training data

# If the DRY_RUN environment variable is set to 1, the data amounts and validation size will be scaled down
# for quick testing purposes

DATA_AMOUNTS_TO_TEST=(4000) # This is the total number of trees used; note that some are used for validation rather than training
VAL_SIZE=100  # Number of trees to use for validation
TOPK=40
COLLECT_LOGFILE=Training_Output/summary/DQC_collect_log.log
LABEL_LOGFILE=Training_Output/summary/DQC_label_log.log
TRAIN_LOGFILE_SUMM=Training_Output/summary/DQC_train_log_summ.log
TRAIN_LOGFILE_FAITH=Training_Output/summary/DQC_train_log_faith.log

# If DRY_RUN is set to 1, scale down the data amounts and val size
if [[ "$DRY_RUN" == "1" ]]; then
    for i in "${!DATA_AMOUNTS_TO_TEST[@]}"; do
        DATA_AMOUNTS_TO_TEST[$i]=$(( DATA_AMOUNTS_TO_TEST[$i] / 50 ))
    done
    VAL_SIZE=$(( VAL_SIZE / 50 ))
fi

MAX_AMOUNT_TO_TEST=${DATA_AMOUNTS_TO_TEST[-1]}
UNLABELED_DIR=Training_Output/summary/unlabeled_data/quantity_comparison/iter_0
FOR_TRAINING_DIR=Training_Output/summary/data_for_training/quantity_comparison/iter_0
TRAINING_OUTPUT_DIR=Training_Output/summary/training_output/quantity_comparison/iter_0
mkdir -p $UNLABELED_DIR
mkdir -p $FOR_TRAINING_DIR

# Step 1: Data collection under the reference policy (only need to do once)
rm -f $COLLECT_LOGFILE
PROMPTS_PER_PROCESS=$(( MAX_AMOUNT_TO_TEST / 4 ))
CUDA_VISIBLE_DEVICES=0 python get_data_base.py 0 $PROMPTS_PER_PROCESS --root_children=4 --non_root_children=2 --num_layers=5 --topk=$TOPK --dataset=summary --output_dir="${UNLABELED_DIR}/proc1"  --silent_loop &>> $COLLECT_LOGFILE &
CUDA_VISIBLE_DEVICES=1 python get_data_base.py $PROMPTS_PER_PROCESS $PROMPTS_PER_PROCESS --root_children=4 --non_root_children=2 --num_layers=5 --topk=$TOPK --dataset=summary --output_dir="${UNLABELED_DIR}/proc2"  --silent_loop &>> $COLLECT_LOGFILE &
CUDA_VISIBLE_DEVICES=2 python get_data_base.py $(( 2 * PROMPTS_PER_PROCESS )) $PROMPTS_PER_PROCESS --root_children=4 --non_root_children=2 --num_layers=5 --topk=$TOPK --dataset=summary --output_dir="${UNLABELED_DIR}/proc3"  --silent_loop &>> $COLLECT_LOGFILE &
CUDA_VISIBLE_DEVICES=3 python get_data_base.py $(( 3 * PROMPTS_PER_PROCESS )) $PROMPTS_PER_PROCESS --root_children=4 --non_root_children=2 --num_layers=5 --topk=$TOPK --dataset=summary --output_dir="${UNLABELED_DIR}/proc4"  --silent_loop &>> $COLLECT_LOGFILE &
wait

# Step 2: Data labeling (only need to do once)
rm -f $LABEL_LOGFILE
CUDA_VISIBLE_DEVICES=0 python label_tree.py "${UNLABELED_DIR}/proc1/all_data.hdf5" all --dataset=summary --check_trees &>> $LABEL_LOGFILE &
CUDA_VISIBLE_DEVICES=1 python label_tree.py "${UNLABELED_DIR}/proc2/all_data.hdf5" all --dataset=summary --check_trees &>> $LABEL_LOGFILE &
CUDA_VISIBLE_DEVICES=2 python label_tree.py "${UNLABELED_DIR}/proc3/all_data.hdf5" all --dataset=summary --check_trees &>> $LABEL_LOGFILE &
CUDA_VISIBLE_DEVICES=3 python label_tree.py "${UNLABELED_DIR}/proc4/all_data.hdf5" all --dataset=summary --check_trees &>> $LABEL_LOGFILE &
wait

# Step 3: Combine the all_tokens.hdf5 files into FOR_TRAINING_DIR
python -c "from utils.hdf5_utils import merge_hdf5_files; merge_hdf5_files(['$UNLABELED_DIR/proc1/all_labeled.hdf5', '$UNLABELED_DIR/proc2/all_labeled.hdf5', '$UNLABELED_DIR/proc3/all_labeled.hdf5', '$UNLABELED_DIR/proc4/all_labeled.hdf5'], '$FOR_TRAINING_DIR/all_labeled.hdf5')"

# Step 6: Do training for each data amount
rm -f $TRAIN_LOGFILE_SUMM
rm -f $TRAIN_LOGFILE_FAITH
CUDA_VISIBLE_DEVICES=0 python train_value_model.py --dataset=summary --objective=summarization --data_file=${FOR_TRAINING_DIR}/all_labeled.hdf5 \
                            --output_dir=${TRAINING_OUTPUT_DIR}/${DATA_AMOUNTS_TO_TEST[0]}/summarization --disable_tqdm \
                            --fraction_bottom_nodes_to_keep=0.5 --batch_size=32 --lr=2e-5 --num_epochs=2 \
                            --num_trees=${DATA_AMOUNTS_TO_TEST[0]} --num_val_trees=$VAL_SIZE &>> $TRAIN_LOGFILE_SUMM &
CUDA_VISIBLE_DEVICES=1 python train_value_model.py --dataset=summary --objective=faithful --data_file=${FOR_TRAINING_DIR}/all_labeled.hdf5 \
                            --output_dir=${TRAINING_OUTPUT_DIR}/${DATA_AMOUNTS_TO_TEST[0]}/faithful --disable_tqdm \
                            --fraction_bottom_nodes_to_keep=0.5 --batch_size=32 --lr=2e-5 --num_epochs=2 \
                            --num_trees=${DATA_AMOUNTS_TO_TEST[0]} --num_val_trees=$VAL_SIZE &>> $TRAIN_LOGFILE_FAITH &
wait