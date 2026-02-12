START_PROMPT=2156
DATA_AMOUNT=3008
BETA=3
LOGFILE=Training_Output/safeRLHF/iter1_logs/get_data_help.log
mkdir -p Training_Output/safeRLHF/iter1_logs/

UNLABELED_DIR=Training_Output/safeRLHF/unlabeled_data/iter1/help
FOR_TRAINING_DIR=Training_Output/safeRLHF/data_for_training/iter1/help
mkdir -p $UNLABELED_DIR
mkdir -p $FOR_TRAINING_DIR

rm -f $LOGFILE
sudo nvidia-smi -i 0,1,2,3 -mig 1 &>> $LOGFILE
bash scripts/get_data_iter_full_utilization.sh safeRLHF_help $START_PROMPT $DATA_AMOUNT $UNLABELED_DIR $BETA 0 &>> $LOGFILE

# Combine labeled data into single file for training
python -c "import glob; from utils.hdf5_utils import merge_hdf5_files; files = sorted(glob.glob('${UNLABELED_DIR}/proc*/all_labeled.hdf5')); merge_hdf5_files(files, '${FOR_TRAINING_DIR}/all_labeled.hdf5')"