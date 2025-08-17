#!/bin/bash

# This script should be run from within the base directory of the MAVIS code

# This script is used to train the iteration 0 value models for helpfulness, harmlessness, and humor
# It is assumed that you have already done dataset preprocessing and run SFT to get the reference policy, as detailed in the README

# Create directory to store data and training results
mkdir -p Anthropic/unlabeled_data/iter_0
mkdir -p Anthropic/data_for_training/iter_0/train
mkdir -p Anthropic/data_for_training/iter_0/val
mkdir -p Anthropic/training_output/iter_0/

# Step 1: Data collection under the reference policy
python get_data_base.py 0 5000 --dataset=anthropic --output_dir=Anthropic/unlabeled_data/iter_0/ 
# Step 2: Data labeling
python label_tree.py Anthropic/unlabeled_data/iter_0/ all --dataset=anthropic --check_trees
# Step 3: Split data into training and validation sets and move to separate directory
mv Anthropic/unlabeled_data/iter_0/all_tokens.hdf5 Anthropic/data_for_training/iter_0/all_tokens.hdf5
# Take the first 300 files for validation and the rest for training
cd Anthropic/unlabeled_data/iter_0/all_labeled/
ls -d "$PWD/"* -1b | head -n 300 | tr '\n' ' ' | xargs mv -t ../../../data_for_training/iter_0/val/
mv *.pkl ../../../data_for_training/iter_0/train/
cd ../../../..
# NOTE: At this point you can delete the Anthropic/unlabeled_data/iter_0 directory if you want since we will only use
    # Anthropic/data_for_training/iter_0 from now on
# Step 4: Train the helpfulness model
python train_value_model.py --dataset=anthropic --objective=help --data_dir=Anthropic/data_for_training/iter_0/ \
                            --output_dir=Anthropic/training_output/iter_0/help --batch_size=16 --lr=2e-5
# Step 5: Train the harmlessness model
python train_value_model.py --dataset=anthropic --objective=harm --data_dir=Anthropic/data_for_training/iter_0/ \
                            --output_dir=Anthropic/training_output/iter_0/harm --batch_size=16 --lr=2e-5
# Step 6: Train the humor model
python train_value_model.py --dataset=anthropic --objective=humor --data_dir=Anthropic/data_for_training/iter_0/ \
                            --output_dir=Anthropic/training_output/iter_0/humor --batch_size=16 --lr=2e-5
# At this point, you can take the adapter_config.json and adapter_model.safetensors files from the output directories
# and place them in the value_models/iter_0/<objective> directories. This needs to be done before training the next iteration.