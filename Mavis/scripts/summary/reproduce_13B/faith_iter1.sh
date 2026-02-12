START_PROMPT=4000
TOPK=40
NUM_PROMPTS=2504  # Total number of prompts to generate trees for
VAL_SIZE=100  # Number of trees to use for validation

PREV_VALUE_MODEL_DIR=Models/value_models

mkdir -p Training_Output/summary/faith_iter1_logs
COLLECT_LOGFILE=Training_Output/summary/faith_iter1_logs/faith_data_collection.log
TRAIN_LOGFILE=Training_Output/summary/faith_iter1_logs/faith_training.log

UNLABELED_DIR=Training_Output/summary/unlabeled_data/faith_rescale_iter1_more
FOR_TRAINING_DIR=Training_Output/summary/data_for_training/faith_rescaled/iter1_more
mkdir -p $FOR_TRAINING_DIR
TRAINING_OUTPUT_DIR=Training_Output/summary/training_output/faith_rescaled/iter1_more

# Data collection
rm -f $COLLECT_LOGFILE
bash scripts/get_data_iter1_full_utilization.sh faithful $START_PROMPT $NUM_PROMPTS $UNLABELED_DIR 8 0 &>> $COLLECT_LOGFILE &
wait

# Combine labeled data into single file for training
python -c "import glob; from utils.hdf5_utils import merge_hdf5_files; files = sorted(glob.glob('${UNLABELED_DIR}/proc*/all_labeled.hdf5')); merge_hdf5_files(files, '${FOR_TRAINING_DIR}/all_labeled.hdf5')"

# Train
rm -f $TRAIN_LOGFILE
python train_value_model.py --dataset=summary --objective=faithful --data_file=${FOR_TRAINING_DIR}/all_labeled.hdf5 \
                                --output_dir=${TRAINING_OUTPUT_DIR} --disable_tqdm --fraction_bottom_nodes_to_keep=0.5 \
                                --batch_size=32 --lr=2e-5 --num_epochs=2 --num_val_trees=$VAL_SIZE \
                                --init_checkpoint=$PREV_VALUE_MODEL_DIR/iter_0/faithful --no_warmup &>> $TRAIN_LOGFILE &
wait