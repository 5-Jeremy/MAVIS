# Environment Setup
The environment we used when running our experiments can be recreated using the included requirements.txt file. The required Python version is 3.10.18.

# GPU Requirements
We recommend using NVIDIA A100 80GB GPUs or other GPUs with a similar amount of memory for these experiments. This is more of a requirement when training if you intend to use the same batch size which we used, but for inference GPUs with less memory will likely suffice.

# Dataset Preprocessing
To create a CSV file with the preprocessed data for a specific split of one of the datasets, run 
```bash
python dataset_preprocess.py <anthropic|summary> --split=<split>
```
where the first argument indicates the dataset and `<split>` is either `train` or `test`.

# Creating the reference model with SFT
In the Finetuning/SFT directory, run the following:
   ```bash
   python sft.py --save_directory <DIR> --wandb_name <NAME> --exp_type <assistant|summary>
   ```
   - `<DIR>`: Path to save the LoRA adapter (e.g., `checkpoints/sft_lora`).
   - `<NAME>`: Run name for logging to Weights & Biases (e.g., `sft_run_1`).
   - `<assistant|summary>`: Type of experiment (e.g., `assistant` for Anthropic HH-RLHF or `summary` for OpenAI Summarize from Feedback).

   ```bash
   python merge_sft_lora.py --lora_name <DIR/NAME>
   ```
   - `<DIR/NAME>`: Full path combining the save directory and run name (e.g., `checkpoints/sft_lora/sft_run_1`).
The merged SFT model should be copied to MAVIS/sft_model/{dataset}

# Training Value Models
We have included scripts with the commands needed to collect training data and train each value model. Note that running these scripts will perform all data collection sequentially on one GPU, which will take much more time than running it in parallel. Thus, when multiple GPUs are available we recommend manually running the commands and using the --device argument to specify different GPUs for the data collection processes so that they can run at the same time. You may have to divide up the prompt ranges manually (e.g. "get_data_base.py 0 5000" would become "get_data_base.py 0 2500 --output_dir={dir1}" and "get_data_base.py 2500 2500 --output_dir={dir2}" - note how unique output directories must be specified to avoid two processes writing to the same HDF5 file concurrently). This will result in different sequences being generated compared to generating them all with one command, but the final training outcome should not be too different. Since each process running a data collection script will create a separate HDF5 file, it is necessary to combine them into a single file for use during training. This can be done using a function in utils/hdf5_utils.py as follows:
```bash
python
>>> from utils.hdf5_utils import *
>>> merge_hdf5_files([<data_dir_1>/all_tokens.hdf5, ..., <data_dir_N>/all_tokens.hdf5], <output_dir>/all_tokens.hdf5)
```
This will create a file called all_tokens.hdf5 inside output_dir which will contain the contents of all of the HDF5 files included in the list passed as the first argument to the function.

Before training the iteration 0 value models, make sure you have preprocessed the data and run SFT to get the reference model as explained in the above steps. To train the iteration 0 value models for all objectives under a given dataset, simply run scripts/{dataset}/train_iter_0.sh (Or if you want to parallelize data collection, run the commands manually). Afterwards, to train future iterations you can use the get_data.sh script to generate and label the data, then create the training/validation splits from the labeled data (following lines 18-23 in train_iter_0.sh) and run the training script as follows:
```bash
python train_value_model.py --dataset <anthropic|summary> --objective <OBJECTIVE> --data_dir <DATA_DIR> --init_checkpoint <INIT_DIR> --output_dir <OUTPUT_DIR> --num_epochs=_ --batch_size=_ --lr=_ --weight_decay=_ --KL_penalty <ETA> --no_warmup
```
- `<OBJECTIVE>`: The abbreviated name for the objective (help for helpfulness, harm for harmlessness, faithful for faithfulness; no abbreviation is used for humor or summarization)
- `<DATA_DIR>`: The directory containing the data from training; should contain an all_tokens.hdf5 file, a train subdirectory, and a val subdirectory
- `<INIT_DIR>`: The directory containing the adapter_config.json and adapter_model.safetensors files for the value model you want to initialize from
- `<OUTPUT_DIR>`: The directory where the training checkpoints, training logs, and final model will be saved
- `<ETA>`: The weight applied to the KL penalty term for the soft value function (see equations 1 and 2 in the main paper)

# Setting up for MAVIS Evaluation
If you have a trained value model for each objective and you wish to evaluate their performance, you must take the files adapter_config.json and adapter_model.safetensors created during training and place them in the directory value_models/iter_#/{objective}/ where the iteration number can be chosen to reflect how many times the value model has undergone training, and {objective} is the abbreviated name of the objective for that value model (see above).

# Running MAVIS Evaluation
The basic command to evaluate the performance of MAVIS is to run 
```bash
python mavis.py --dataset <anthropic|summary> --obj_weights <WEIGHTS> --value_model_dir <VM_DIR> --value_model_iter <VM_ITER> --beta <BETA> [--track_KL]
```
- `<WEIGHTS>`: A comma-separated list of floating-point values that sum to one. If the weight of an objective is set to 0.0, no value model will be loaded for that objective but rewards for that objective will still be tracked. If you do not want to track rewards for an objective, leave its spot in the list empty (e.g. 0.2,,0.8). The order of objectives is [helpfulness,harmlessness,humor] for the anthropic dataset and [summarization,faithfulness] for the summary dataset.
- `<VM_DIR>`: The path to a directory containing value models for the objectives which will need them. The structure of the directory should be as specified in "Setting up for MAVIS Evaluation".
- `<VM_ITER>`: A comma-separated list of non-negative integers for the objectives which need value models. This is used to determine which subdirectory within `<VM_DIR>` each value model should be loaded from. For objectives which do not need a value model, leave their position empty (e.g. 1,,0).
- `<BETA>`: Parameter which scales the normalized values, and hence determines how much they can alter the token distribution
- `track_KL`: If this flag is used, the KL divergence between the MAVIS policy and the reference policy (computed according to equation 6 in the main paper) will be recorded in the final statistics

Note that we have included scripts which will evaluate the entire pareto front by repeatedly calling this command, but these scripts assume that a single value of beta will be used across the entire pareto front; for more flexibility, run the commands manually

# PPO/MORLHF Training for Baselines
We use modified scripts from the official code for Rewards-In-Context [1] for training PPO models. Assuming you have created and merged the SFT model for the desired dataset, you can run PPO fine-tuning using the ppo.py script in Finetuning/PPO as follows:

```bash
python ppo.py --base_model <SFT_MODEL_PATH> --exp_type <assistant|summary> --alpha <ALPHA>
```
- `<SFT_MODEL_PATH>`: Path to the merged sft model for the desired dataset.
- `<assistant|summary>`: Type of experiment (e.g., `assistant` for Anthropic HH-RLHF or `summary` for OpenAI Summarize from Feedback).
- `<ALPHA>`: This is equivalent to lambda_1 in our formulation.

# Setting up for Baseline Evaluation
Models trained using PPO should be organized into the following directory trees (all within the MAVIS directory):

- morlhf
   - anthropic
      - harm_humor
         - morlhf_0.2
         - morlhf_0.4
         - morlhf_0.6
         - morlhf_0.8
      - help_harm
         - morlhf_0.2
         - &#x22EE;
      - help_humor
         - morlhf_0.2
         - &#x22EE;
      - single
         - help
         - harm
         - humor
   - summary
      - morlhf_0.2
      - &#x22EE;
      - single
         - summarization
         - faithful
- reward_soup
   - anthropic
      - harm_humor
         - reward_soup_0.2
         - reward_soup_0.4
         - reward_soup_0.6
         - reward_soup_0.8
      - help_harm
         - reward_soup_0.2
         - &#x22EE;
      - help_humor
         - reward_soup_0.2
         - &#x22EE;
   - summary
      - reward_soup_0.2
      - reward_soup_0.4
      - reward_soup_0.6
      - reward_soup_0.8

# Running Baseline Evaluation
There are three scripts used for evaluating the baselines. The first two, baseline_eval.py and baseline_eval_KL.py, are used to run MORLHF and Rewarded Soups. The third, mod_eval.py, is used to run MOD. Note that baseline_eval.py and mod_eval.py only evaluate the rewards (not the KL divergence) while baseline_eval_KL.py also evaluates the KL divergence by running a forward pass on the generated sequence through the reference model. All of these scripts use the same obj_weights argument to control the weight assigned to each objective. For MORLHF and Rewarded Soups, a different model is loaded depending on the objective weights. For MOD, the single-objective PPO model for each of the relevant objectives are all loaded. When running baseline_eval.py or baseline_eval_KL.py, the arguments --ppo and --morlhf determine which model to load when multiple objectives have weight assigned to them (if only one objective has weight, they have the same effect). The --ppo argument corresponds to Rewarded Soups, and the --morlhf argument corresponds to MORLHF. If neither argument is used, then the reference model will be loaded instead.

# Evaluation Logs
Whenever an evaluation script is run, it creates a new subdirectory in one of four main directories (either baseline_logs, baseline_bon_logs, mavis_logs, or mavis_bs_logs) depending on which script was run. Inside the subdirectory, you will find a YAML file listing the arguments that the script was run with, a CSV file with the average rewards and average generation time for each prompt, and (if the evaluation runs to completion) a text file with the overall statistics.

# Test-Time Search
The baseline methods can be run with best-of-N simply by adding `--num_samples=N` when running the evaluation script. To run MAVIS with beam search, you must use mavis_bs.py which accepts a similar set of arguments as mavis.py except it does not accept `--track_KL` and it has the additional arguments `Q`, `N`, and `num_samples`. `Q` determines the number of parallel sequences, `N` determines the number of final candidates which are returned to be ranked by the reward models (set to `Q` by default), and `num_samples` is a hyperparameter which determines how many candidates from each beam are compared with each other when deciding on the next set of beams (we did not observe improved performance from changing this parameter from its default value).

# Acknowledgements
The code we use for fine-tuning the generative model is heavily based on the official code for Rewards-in-Context.

[1] Yang, R.; Pan, X.; Luo, F.; Qiu, S.; Zhong, H.; Yu, D.; and Chen, J. 2024. Rewards-in-Context: Multi-objective Alignment of Foundation Models with Dynamic Preference Adjustment. arXiv:2402.10207.
