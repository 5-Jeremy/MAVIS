import torch
import numpy as np
from time import time_ns
import random
import pandas as pd
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def eval_loop(inputs, prompt_indices, num_trials, search_routine, reward_models, rm_device, get_rewards, obj_weights, 
              verbose=False, track_time=False, track_KL=False, file_name=None, seed=0):
    objectives = [obj for obj, weight in obj_weights.items() if weight is not None]
    # Dictionary to store trial-level results
    trial_results = {obj: {} for obj in objectives}
    
    # Initialize the trial dictionary
    for t in range(num_trials):
        for obj in objectives:
            trial_results[obj][t+1] = []

    if track_KL:
        per_sequence_KL = []

    objective_name_map = {
        "help": "helpfulness",
        "harm": "harmlessness",
        "humor": "humor",
        "summarization": "summarization",
        "faithful": "faithfulness"
    }
    if file_name is not None:
        objective_reward_headers = ", ".join([f"{objective_name_map[obj]} reward" for obj in objectives])
        with open(file_name, "w") as f:
            f.write(f"Prompt index, {objective_reward_headers}, Mean time\n")
            
    for indx in range(len(inputs)):
        set_seed(seed)
        print("****************************************************************************************")
        print("Prompt index: ", prompt_indices[indx])
        trial_rewards = []
        trial_times = []
        prompt = inputs[indx]
        if track_KL:
            trial_KL = []
        
        for t in range(num_trials):
            if verbose: print(f"Trial {t+1} ======================================================")
            start = time_ns()

            for rm in reward_models.values():
                rm.to("cpu")

            if track_KL:
                final_candidates, sequence_KL = search_routine(prompt)
                trial_KL.append(sequence_KL)
            else:
                final_candidates = search_routine(prompt)
            
            if track_time: print("Time taken for search: ", (time_ns() - start) / 1e9, " seconds")
            trial_times.append((time_ns() - start) / 1e9)

            for rm in reward_models.values():
                rm.to(rm_device)
                
            # If the algorithm returns multiple final candidates, then we need to evaluate them with the reward model and choose
            # the best one here. Otherwise, we assume the search algorithm itself chose the best candidate.
            assert type(final_candidates) == list, "final_candidates should be a list even if there is only one candidate"
            if isinstance(final_candidates[0], list):
                raise ValueError("Final candidates should be a list of strings, not a list of tokenized sequences")
            # Evaluate the final candidates with the reward model (they are all responses to the same prompt)
            rewards = get_rewards(final_candidates)
            ranking_rewards = torch.zeros(len(final_candidates))
            for i in range(len(final_candidates)):
                for k in objectives:
                    ranking_rewards[i] += (rewards[k][i] * obj_weights[k]).item()
            top_reward, top_index = torch.topk(ranking_rewards, 1)
            top_reward = top_reward.item()
            top_index = top_index.item()
            chosen_rewards = {}
            for k, v in rewards.items():
                chosen_rewards[k] = v[top_index].item()
            
            for rm in reward_models.values():
                rm.to("cpu")

            # At this point, the final list of candidates is stored in seq
            if verbose:
                print("Completion: ", final_candidates[top_index])
            
            # Store rewards for this trial
            trial_rewards.append(chosen_rewards)
            
            # Add rewards to per-trial tracking
            for obj in objectives:
                trial_results[obj][t+1].append(chosen_rewards[obj])

        # Calculate prompt-level metrics
        objective_means = {}
        for obj in objectives:
            objective_means[obj] = np.mean([r[obj] for r in trial_rewards])
            print(f"Mean {objective_name_map[obj]} reward over all trials: ", objective_means[obj])
        
        if track_KL:
            per_sequence_KL.append(trial_KL)

        if track_time: print("Mean time over all trials: ", np.mean(trial_times))
        
        if file_name is not None:
            objective_means_str = ", ".join([f"{objective_means[obj]}" for obj in objectives])
            with open(file_name, "a") as f:
                f.write(f"{prompt_indices[indx]}, {objective_means_str}, {np.mean(trial_times)}\n")

    # Close file if it was opened
    if file_name is not None:
        with open(file_name, "a") as f:
            f.close()
    
    # Calculate trial-level summary statistics
    trial_means = {obj: {} for obj in objectives}
    trial_stds = {obj: None for obj in objectives}
    
    for t in range(num_trials):
        for obj in objectives:
            if trial_results[obj][t+1]:
                trial_means[obj][t+1] = np.mean(trial_results[obj][t+1])
            else:
                trial_means[obj][t+1] = None
    
    print("****************************************************************************************")
    # Calculate standard deviations across trials
    for obj in objectives:
        if trial_means[obj]:
            trial_mean_values = list(trial_means[obj].values())
            trial_stds[obj] = np.std(trial_mean_values)
            print(f"Trial-level standard deviation for {objective_name_map[obj]}: {trial_stds[obj]:.4f}")
        else:
            trial_stds[obj] = None  
    
    # Now average over trials
    for obj in objectives:
        if trial_results[obj]:
            print(f"Average {objective_name_map[obj]} reward across trials: {np.mean(list(trial_means[obj].values())):.4f}")
    if track_KL:
        per_prompt_KL_means = [np.mean(seq_KL) for seq_KL in per_sequence_KL]
        per_trial_KL_means = [np.mean([trial_KL[t] for trial_KL in per_sequence_KL]) for t in range(num_trials)]
        print(f"Average KL divergence across trials: {np.mean(per_prompt_KL_means):.4f}")

    # Save the aggregated results to a txt file
    with open(file_name.replace(".csv", "_trial_results.txt"), "w") as f:
        f.write("Trial-level results:\n")
        for obj in objectives:
            f.write(f"{objective_name_map[obj]}:\n")
            for t in range(1, num_trials + 1):
                if t in trial_results[obj]:
                    f.write(f"  Trial {t}: {trial_results[obj][t]}\n")
                else:
                    f.write(f"  Trial {t}: N/A\n")
            f.write(f"  Mean: {np.mean(list(trial_means[obj].values()))}\n")
            f.write(f"  Std: {trial_stds[obj]}\n")
            f.write("\n")
        if track_KL:
            f.write(f"Overall average KL divergence: {np.mean(per_prompt_KL_means):.4f}\n")
            f.write(f"Standard deviation of KL divergence across trials: {np.std(per_trial_KL_means):.4f}\n")

def parse_objective_weights(lambda_str, dataset_valid_objectives):
    weights = [x.strip() for x in lambda_str.split(",")]
    num_weights = len(weights)
    assert num_weights <= len(dataset_valid_objectives), f"Expected at most {len(dataset_valid_objectives)} weights, got {num_weights}."
    # We only evaluate on the objectives for which obj_mask is True
    lambda_vec = np.zeros(len(dataset_valid_objectives), dtype=float)
    obj_mask = np.ones(len(dataset_valid_objectives), dtype=bool)
    for i, weight in enumerate(weights):
        if weight == "":
            obj_mask[i] = False  # Objective is not evaluated
        else:
            try:
                lambda_vec[i] = float(weight)
            except ValueError:
                raise ValueError(f"Invalid weight '{weight}' at position {i}. Weights must be float values.")
    if len(weights) < len(dataset_valid_objectives):
        # If fewer weights are provided than objectives, the rest should not be evaluated
        obj_mask[len(weights):] = False
    assert np.all(lambda_vec[obj_mask] >= 0), "All weights must be non-negative."
    assert np.all(lambda_vec[obj_mask] <= 1), "All weights must be less than or equal to 1."
    assert np.isclose(np.sum(lambda_vec[obj_mask]), 1.0), "Weights must sum to 1."
    # For objectives that are not evaluated, we set the weight to None rather than 0
    return {objective: weight.item() if obj_mask[indx] else None for indx, (objective, weight) in enumerate(zip(dataset_valid_objectives, lambda_vec))}

# If the value models you want to use come from different iterations, you need to format the value_model_iter argument
# as <obj 1 iter>,<obj 2 iter>,... so that it can be parsed by this function to get a mapping from objective names to 
# iterations
def parse_value_model_iter(iter_str, dataset_valid_objectives):
    if ',' not in iter_str:
        assert iter_str.isdigit()
        return {objective: iter_str for objective in dataset_valid_objectives}
    else:
        iter_values = [x.strip() for x in iter_str.split(',')]
        num_values = len(iter_values)
        assert num_values <= len(dataset_valid_objectives), f"Expected at most {len(dataset_valid_objectives)} weights, got {num_values}."
        # We only evaluate on the objectives for which obj_mask is True
        obj_mask = np.ones(len(dataset_valid_objectives), dtype=bool)
        for i, val in enumerate(iter_values):
            if val == "":
                obj_mask[i] = False  # Objective is not evaluated
            else:
                try:
                    int(val)
                except ValueError:
                    raise ValueError(f"Invalid iteration number '{val}' at position {i}. Must be a non-negative integer.")
        if len(iter_values) < len(dataset_valid_objectives):
            # If fewer weights are provided than objectives, the rest should not be evaluated
            obj_mask[len(iter_values):] = False
        return {objective: int(value) if obj_mask[indx] else None for indx, (objective, value) in enumerate(zip(dataset_valid_objectives, iter_values))}
        

def assign_gpus():
    if torch.cuda.device_count() >= 2:
        g_device = "cuda:0"
        rm_device = "cuda:1"
    elif torch.cuda.device_count() == 1:
        g_device = "cuda:0"
        rm_device = "cuda:0"
    else:
        raise ValueError("No GPU detected")
    print("Generative model device:", g_device)
    print("Reward model device:", rm_device)
    return g_device, rm_device