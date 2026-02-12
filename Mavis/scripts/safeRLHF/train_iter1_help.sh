VAL_SIZE=100
TRAIN_LOGFILE_HELP_02=Training_Output/safeRLHF/iter1_logs/train_help_0.02.log
TRAIN_LOGFILE_HELP_03=Training_Output/safeRLHF/iter1_logs/train_help_0.03.log
TRAIN_LOGFILE_HELP_NO_PENALTY=Training_Output/safeRLHF/iter1_logs/train_help_no_penalty.log
TRAIN_LOGFILE_HELP_less_bottom_nodes=Training_Output/safeRLHF/iter1_logs/train_help_less_bottom_nodes.log
FOR_TRAINING_DIR=Training_Output/safeRLHF/data_for_training/iter1/help
INIT_DIR=Models/value_models/iter_0/safeRLHF_help
TRAINING_OUTPUT_DIR=Training_Output/safeRLHF/training_output/iter1/help
mkdir -p Training_Output/safeRLHF/iter1_logs

CUDA_VISIBLE_DEVICES=0 python train_value_model.py --dataset=safeRLHF --objective=safeRLHF_help \
                            --data_file=${FOR_TRAINING_DIR}/all_labeled.hdf5 \
                            --output_dir=${TRAINING_OUTPUT_DIR}/safeRLHF_help_0.02 --disable_tqdm \
                            --fraction_bottom_nodes_to_keep=0.5 --batch_size=32 --lr=2e-5 --num_epochs=2 \
                            --num_val_trees=$VAL_SIZE --KL_penalty=0.02 --init_checkpoint=$INIT_DIR \
                            --no_warmup &>> $TRAIN_LOGFILE_HELP_02 &
CUDA_VISIBLE_DEVICES=1 python train_value_model.py --dataset=safeRLHF --objective=safeRLHF_help \
                            --data_file=${FOR_TRAINING_DIR}/all_labeled.hdf5 \
                            --output_dir=${TRAINING_OUTPUT_DIR}/safeRLHF_help_0.03 --disable_tqdm \
                            --fraction_bottom_nodes_to_keep=0.5 --batch_size=32 --lr=2e-5 --num_epochs=2 \
                            --num_val_trees=$VAL_SIZE --KL_penalty=0.03 --init_checkpoint=$INIT_DIR \
                            --no_warmup &>> $TRAIN_LOGFILE_HELP_03 &
CUDA_VISIBLE_DEVICES=2 python train_value_model.py --dataset=safeRLHF --objective=safeRLHF_help \
                            --data_file=${FOR_TRAINING_DIR}/all_labeled.hdf5 \
                            --output_dir=${TRAINING_OUTPUT_DIR}/safeRLHF_help_no_penalty --disable_tqdm \
                            --fraction_bottom_nodes_to_keep=0.5 --batch_size=32 --lr=2e-5 --num_epochs=2 \
                            --num_val_trees=$VAL_SIZE --init_checkpoint=$INIT_DIR \
                            --no_warmup &>> $TRAIN_LOGFILE_HELP_NO_PENALTY &
CUDA_VISIBLE_DEVICES=3 python train_value_model.py --dataset=safeRLHF --objective=safeRLHF_help \
                            --data_file=${FOR_TRAINING_DIR}/all_labeled.hdf5 \
                            --output_dir=${TRAINING_OUTPUT_DIR}/safeRLHF_help_less_bottom_nodes --disable_tqdm \
                            --fraction_bottom_nodes_to_keep=0.5 --batch_size=32 --lr=2e-5 --num_epochs=2 \
                            --num_val_trees=$VAL_SIZE --init_checkpoint=$INIT_DIR \
                            --no_warmup &>> $TRAIN_LOGFILE_HELP_less_bottom_nodes &
wait