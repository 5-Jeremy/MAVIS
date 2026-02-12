#!/bin/bash

# This script should be run from within the base directory of the MAVIS code

# This script trains the iteration 0 value models for safeRLHF_help and safeRLHF_harm with varying amounts 
# of training data. It assumes that you have 4 GPUs available with at least 80 GB of memory each, and
# uses MIG to partition the GPUs for maximum utilization.

# IMPORTANT: Make sure you source your conda environment before running this script. The script requires
# root priviledges at certain points, but it should not be run with sudo. Instead, you will be prompted
# for your password when needed.

# If the DRY_RUN environment variable is set to 1, the data amount and validation size will be scaled down
# for quick testing purposes

DATA_AMOUNT=3008 # This is the total number of trees used; note that some are used for validation rather than training
VAL_SIZE=100  # Number of trees to use for validation
TOPK=40
COLLECT_LOGFILE=Training_Output/safeRLHF/iter0_collect.log
LABEL_LOGFILE=Training_Output/safeRLHF/iter0_labeling.log
TRAIN_LOGFILE_HELP=Training_Output/safeRLHF/iter0_train_log_help.log
TRAIN_LOGFILE_HARM=Training_Output/safeRLHF/iter0_train_log_harm.log

# If DRY_RUN is set to 1, scale down the data amounts and val size
if [[ "$DRY_RUN" == "1" ]]; then
    DATA_AMOUNT=$(( DATA_AMOUNT / 94 ))
    VAL_SIZE=$(( VAL_SIZE / 50 ))
fi

UNLABELED_DIR=Training_Output/safeRLHF/unlabeled_data/iter0
FOR_TRAINING_DIR=Training_Output/safeRLHF/data_for_training/iter0
TRAINING_OUTPUT_DIR=Training_Output/safeRLHF/training_output/iter0
mkdir -p $UNLABELED_DIR
mkdir -p $FOR_TRAINING_DIR

# Collect data
rm -f $COLLECT_LOGFILE
sudo nvidia-smi -i 0,1,2,3 -mig 1 &>> $COLLECT_LOGFILE
bash scripts/get_data_iter0_full_utilization.sh safeRLHF $DATA_AMOUNT $UNLABELED_DIR &>> $COLLECT_LOGFILE &
GET_DATA_PID=$!
wait $GET_DATA_PID
GET_DATA_EXIT_CODE=$?
if [ $GET_DATA_EXIT_CODE -ne 0 ]; then
    echo "Data collection script failed with exit code $GET_DATA_EXIT_CODE. Exiting."
    exit $GET_DATA_EXIT_CODE
fi

# Combine data into single file before labeling
python -c "import glob; from utils.hdf5_utils import merge_hdf5_files; files = sorted(glob.glob('${UNLABELED_DIR}/proc*/all_data.hdf5')); merge_hdf5_files(files, '${UNLABELED_DIR}/merged_data.hdf5')"

# Label data
rm -f $LABEL_LOGFILE
python label_tree.py "${UNLABELED_DIR}/merged_data.hdf5" all --dataset=safeRLHF --check_trees &>> $LABEL_LOGFILE

# Move the labeled data to the training directory
mv "${UNLABELED_DIR}/all_labeled.hdf5" "${FOR_TRAINING_DIR}/all_labeled.hdf5"

# Do training
rm -f $TRAIN_LOGFILE_HELP
rm -f $TRAIN_LOGFILE_HARM
CUDA_VISIBLE_DEVICES=0 python train_value_model.py --dataset=safeRLHF --objective=safeRLHF_help \
                            --data_file=${FOR_TRAINING_DIR}/all_labeled.hdf5 \
                            --output_dir=${TRAINING_OUTPUT_DIR}/safeRLHF_help --disable_tqdm \
                            --fraction_bottom_nodes_to_keep=0.5 --batch_size=32 --lr=2e-5 --num_epochs=2 \
                            --num_val_trees=$VAL_SIZE &>> $TRAIN_LOGFILE_HELP &
CUDA_VISIBLE_DEVICES=1 python train_value_model.py --dataset=safeRLHF --objective=safeRLHF_harm \
                            --data_file=${FOR_TRAINING_DIR}/all_labeled.hdf5 \
                            --output_dir=${TRAINING_OUTPUT_DIR}/safeRLHF_harm --disable_tqdm \
                            --fraction_bottom_nodes_to_keep=0.5 --batch_size=32 --lr=2e-5 --num_epochs=2 \
                            --num_val_trees=$VAL_SIZE &>> $TRAIN_LOGFILE_HARM &
wait