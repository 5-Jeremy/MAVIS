import numpy as np
import torch
import os
from datetime import datetime
import argparse

from transformers.utils import logging
logging.get_logger("transformers").setLevel(logging.ERROR)

from utils.loading_utils import load_for_anthropic, load_for_summary
from utils.search_utils import assign_gpus, eval_loop, parse_objective_weights
from utils.gen_utils import add_generation_args

A = argparse.ArgumentParser()
add_generation_args(A, {"data_split":"test"}) # dataset, data_split, temperature, topk, device, no_cache
A.add_argument("--obj_weights", type=str, default="1.0,,", help="Comma-separated list of weights for the three objectives (Help, Harm, Humor for the anthropic dataset). If you do not want to evaluate on an objective, leave it empty (e.g. 0.5,,0.5).")
A.add_argument("--num_prompts", type=int, default=100, help="Number of prompts to evaluate on; the prompts are selected in order starting from index 0")
A.add_argument("--num_trials", type=int, default=3)
A.add_argument("--track_time", action="store_true")
A.add_argument("--num_samples", type=int, default=1)
A.add_argument("--ppo", action="store_true")
A.add_argument("--morlhf", action="store_true", help="Use the MORLHF model instead of the reward soup mix")
A.add_argument("--seed", type=int, default=0, help="The random seed is set to this at the start of each prompt")
A.add_argument("--log_dir", type=str, default="baseline_logs/" + datetime.now().strftime("%m-%d_%H-%M"))
args = A.parse_args()

verbose = True
prompt_indices = list(range(args.num_prompts))

print("Dataset:", args.dataset)
print(f"Prompt indices: {prompt_indices[0]} to {prompt_indices[-1]}")
print("Number of samples (N):", args.num_samples)

g_device, rm_device = assign_gpus()

if args.dataset == "anthropic":
    dataset_valid_objectives = ["help", "harm", "humor"]
elif args.dataset == "summary":
    dataset_valid_objectives = ["summarization", "faithful"]
else:
    raise ValueError("Invalid dataset_type")

# Parse the given objective weights
obj_weights = parse_objective_weights(args.obj_weights, dataset_valid_objectives)

objectives = [obj for obj, weight in obj_weights.items() if weight is not None]

# Reward and value models start in the CPU to avoid OOM
if args.dataset == "anthropic":
    loaded_assets = load_for_anthropic(
        csv_path="datasets/anthropic/",
        prompt_indices=prompt_indices, rewards=objectives, split=args.data_split, 
        pre_tokenized=False, rm_device=rm_device, obj_weights=obj_weights, ppo=args.ppo,
        morlhf=args.morlhf,
    )
    ref_model = load_for_anthropic(include_inputs=False, include_rewards=False)["gen_model"].to(g_device)
elif args.dataset == "summary":
    loaded_assets = load_for_summary(
        csv_path="datasets/summary/",
        prompt_indices=prompt_indices, rewards=objectives, split=args.data_split, 
        pre_tokenized=False, rm_device=rm_device, obj_weights=obj_weights, ppo=args.ppo,
        morlhf=args.morlhf
    )
    ref_model = load_for_summary(include_inputs=False, include_rewards=False)["gen_model"].to(g_device)
else:
    raise ValueError("Invalid dataset_type")
generative_model = loaded_assets["gen_model"].to(g_device)
tokenizer = loaded_assets["gen_tokenizer"]
eos_token_id = loaded_assets["eos_token_id"]
reward_models = loaded_assets["ORM_model"]
get_rewards = loaded_assets["get_rewards"]
inputs = loaded_assets["prompts"]
max_completion_len = loaded_assets["max_completion_len"]

def get_best_of_n():
    # We need this variable to stay in scope inbetween calls to best_of_n
    sequence_kl_div_list = []
    def best_of_n(prompt_str):
        """
        Generate N trajectories using the generative model and return all of them.
        The selection of the best one will be handled by the eval_loop.
        """
        prompt = tokenizer(prompt_str, return_tensors="pt").to(g_device)
        num_prompt_tokens = prompt["input_ids"].shape[1]
        
        # Generate N samples
        with torch.no_grad():
            outputs = generative_model.generate(
                prompt["input_ids"],
                do_sample=True,
                top_k=args.topk, # Use 0 to disable top-k sampling
                temperature=args.temperature,
                max_new_tokens=max_completion_len,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=args.num_samples,
                return_dict_in_generate=True,
                output_logits=True,
                output_scores=True,
            )
            sequence = outputs.sequences
            baseline_logits = outputs.logits
            chosen_tokens = sequence[0, num_prompt_tokens:]
            
            # The scores will account for the fact that we are only sampling from the top-k tokens
            baseline_scores = outputs.scores
            baseline_scores_tensor = torch.stack(baseline_scores, dim=1)  # Shape: (batch_size, num_new_tokens, vocab_size)
            baseline_score_probs = torch.softmax(baseline_scores_tensor, dim=-1)

            ref_logits = ref_model(sequence, return_dict=True).logits
            # ref_probs will have an extra probability for what comes after the last token, so we have to shift the indices we extract
            ref_probs = torch.softmax(ref_logits[:, num_prompt_tokens-1:-1, :]/args.temperature, dim=-1)

        # generate() returns logits as a tuple of T tensors, one for each generated token
        # We compute the log probability ratio for each token and sum them up to get the sequence log probability ratio
        sequence_logprob_ratio = 0
        for i in range(len(baseline_logits)):
            token_prob_ratio = baseline_score_probs[:,i, chosen_tokens[i].item()] / ref_probs[:,i, chosen_tokens[i].item()]
            sequence_logprob_ratio += torch.log(token_prob_ratio)

        sequence_kl_div_list.append(sequence_logprob_ratio.item())
        print(f"Running average KL divergence: {np.mean(sequence_kl_div_list)}")

        # Decode the generated sequences
        candidates = tokenizer.batch_decode(sequence, skip_special_tokens=True)

        return candidates, sequence_logprob_ratio.item()
    return best_of_n

log_dir = os.path.join(args.log_dir, "_".join(objectives), datetime.now().strftime("%m-%d_%H-%M"))

log_dir = args.log_dir
if args.ppo:
    file_name = os.path.join(log_dir, f"results_ppo_{args.obj_weights}.csv")
elif args.morlhf:
    file_name = os.path.join(log_dir, f"results_morlhf_{args.obj_weights}.csv")
else:
    file_name = os.path.join(log_dir, "results_base.csv")
os.makedirs(log_dir, exist_ok=True)

if __name__ == "__main__":
    eval_loop(inputs, prompt_indices, args.num_trials, get_best_of_n(), reward_models, rm_device, get_rewards, obj_weights=obj_weights,
              track_time=args.track_time, track_KL=True, file_name=file_name, seed=args.seed)