#!/bin/bash

# This script should be run from within the base directory of the MAVIS code

# This script is used to train the iteration 0 value models for summarization and faithfulness
# It is assumed that you have already done dataset preprocessing and run SFT to get the reference policy, as detailed in the README

# Create directory to store data and training results
mkdir -p Summary/unlabeled_data/iter_0
mkdir -p Summary/data_for_training/iter_0/train
mkdir -p Summary/data_for_training/iter_0/val
mkdir -p Summary/training_output/iter_0/

# Step 1: Data collection under the reference policy
python get_data_base.py 0 3000 --dataset=summary --output_dir=Summary/unlabeled_data/iter_0/ 
# Step 2: Data labeling
python label_tree.py Summary/unlabeled_data/iter_0/ all --dataset=summary --check_trees
# Step 3: Split data into training and validation sets and move to separate directory
mv Summary/unlabeled_data/iter_0/all_tokens.hdf5 Summary/data_for_training/iter_0/all_tokens.hdf5
# Take the first 200 files for validation and the rest for training
cd Summary/unlabeled_data/iter_0/all_labeled/
ls -d "$PWD/"* -1b | head -n 200 | tr '\n' ' ' | xargs mv -t ../../../data_for_training/iter_0/val/
mv *.pkl ../../../data_for_training/iter_0/train/
cd ../../../..
# NOTE: At this point you can delete the Summary/unlabeled_data/iter_0 directory if you want since we will only use
    # Summary/data_for_training/iter_0 from now on
# Step 4: Train the summarization model
python train_value_model.py --dataset=summary --objective=summarization --data_dir=Summary/data_for_training/iter_0/ \
                            --output_dir=Summary/training_output/iter_0/summarization
# Step 5: 
python train_value_model.py --dataset=summary --objective=faithful --data_dir=Summary/data_for_training/iter_0/ \
                            --output_dir=Summary/training_output/iter_0/faithful
# At this point, you can take the adapter_config.json and adapter_model.safetensors files from the output directories
# and place them in the value_models/iter_0/<objective> directories. This needs to be done before training the next iteration.