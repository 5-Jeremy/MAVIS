#!/bin/bash

# Script arguments:
#     $1: The value model iteration numbers (one for each objective).
#     $2: The value of beta to use.

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <iteration_numbers> <beta>"
    echo "Example: $0 0,1 5"
    echo "In the above example, the summarization model would be iteration 0 and the faithfulness model would be iteration 1."
else
    mkdir -p mavis_logs/summ_faith/
    python mavis.py --dataset=summary --obj_weights=0.0,1.0 --value_model_iter=$1 --beta=$2 --normalize_values > mavis_logs/summ_faith/0.0_output.txt
    python mavis.py --dataset=summary --obj_weights=0.2,0.8 --value_model_iter=$1 --beta=$2 --normalize_values > mavis_logs/summ_faith/0.2_output.txt
    python mavis.py --dataset=summary --obj_weights=0.4,0.6 --value_model_iter=$1 --beta=$2 --normalize_values > mavis_logs/summ_faith/0.4_output.txt
    python mavis.py --dataset=summary --obj_weights=0.6,0.4 --value_model_iter=$1 --beta=$2 --normalize_values > mavis_logs/summ_faith/0.6_output.txt
    python mavis.py --dataset=summary --obj_weights=0.8,0.2 --value_model_iter=$1 --beta=$2 --normalize_values > mavis_logs/summ_faith/0.8_output.txt
    python mavis.py --dataset=summary --obj_weights=1.0,0.0 --value_model_iter=$1 --beta=$2 --normalize_values > mavis_logs/summ_faith/1.0_output.txt
fi