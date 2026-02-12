"""
WARNING: This script assumes that the distilled value model will have 3 outputs if it is for the anthropic dataset. For the 
summary dataset, it assumes 2 outputs.
"""

import numpy as np
import torch
import os, random, yaml
from time import time_ns
from datetime import datetime
import argparse

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

from transformers.utils import logging
logging.get_logger("transformers").setLevel(logging.ERROR)

from utils.loading_utils import load_for_anthropic, load_for_summary, load_value_model
from utils.search_utils import assign_gpus, eval_loop, parse_objective_weights, parse_value_model_iter
from utils.gen_utils import ValueGeneratorMultiHead, ValueGeneratorMultiHeadBatched, get_mavis_alg, add_generation_args, add_mavis_args

A = argparse.ArgumentParser()
# TODO: Make beta a required argument
add_generation_args(A, {"data_split":"test"}) # dataset, data_split, temperature, topk, device, no_cache
add_mavis_args(A, {}) # value_model_dir, value_model_iter, beta, allow_eos_on_first_token
A.add_argument("--num_prompts", type=int, default=100, help="Number of prompts to evaluate on; the prompts are selected in order starting from index 0")
A.add_argument("--obj_weights", type=str, default="1.0,0.0,0.0", help="Comma-separated list of weights for the three objectives (Help, Harm, Humor for the anthropic dataset). If you do not want to evaluate on an objective, leave it empty (e.g. 0.5,,0.5).")
A.add_argument("--num_trials", type=int, default=3)
A.add_argument("--track_time", action="store_true")
A.add_argument("--seed", type=int, default=0, help="The random seed is set to this at the start of each prompt")
A.add_argument("--log_dir", type=str, default="Eval_Logs/mavis_logs/")
A.add_argument("--track_KL", action="store_true", help="If set, the average KL divergence over all sequences generated will be tracked and included in the results.")
A.add_argument("--batched", action="store_true", help="If set, will generate all trials in a single batch rather than sequentially.")
args = A.parse_args()
assert not args.no_cache, "The no_cache argument will not be supported until the ValueGenerator class is updated."

verbose = True
prompt_indices = list(range(args.num_prompts))

print("Dataset:", args.dataset)
print(f"Prompt range: {prompt_indices[0]} to {prompt_indices[-1]}")

g_device, rm_device = assign_gpus(devices=[args.device])

if args.dataset == "anthropic":
    dataset_valid_objectives = ["help", "harm", "humor"]
elif args.dataset == "summary":
    dataset_valid_objectives = ["summarization", "faithful"]
else:
    raise ValueError("Invalid dataset_type")

# Parse the given objective weights and value model iteration numbers
obj_weights = parse_objective_weights(args.obj_weights, dataset_valid_objectives)
iter_nums = parse_value_model_iter(args.value_model_iter, dataset_valid_objectives)

objectives = [obj for obj, weight in obj_weights.items() if weight is not None]

value_models = {}
if args.dataset == "anthropic":
    loaded_assets = load_for_anthropic(
        csv_path="datasets/anthropic/",
        prompt_indices=prompt_indices, rewards=objectives, split=args.data_split, 
        pre_tokenized=False, rm_device=rm_device, obj_weights=obj_weights, base_model_type=args.base_model_type
    )
elif args.dataset == "summary":
    loaded_assets = load_for_summary(
        csv_path="datasets/summary/",
        prompt_indices=prompt_indices, rewards=objectives, split=args.data_split, 
        pre_tokenized=False, rm_device=rm_device, obj_weights=obj_weights, base_model_type=args.base_model_type
    )
else:
    raise ValueError("Invalid dataset type")
generative_model = loaded_assets["gen_model"].to(g_device)
tokenizer = loaded_assets["gen_tokenizer"]
eos_token_id = loaded_assets["eos_token_id"]
reward_model = loaded_assets["ORM_model"]
get_rewards = loaded_assets["get_rewards"]
inputs = loaded_assets["prompts"]
max_completion_len = loaded_assets["max_completion_len"]

output_name_map = {0: "help", 1: "harm", 2: "humor"} if args.dataset == "anthropic" else {0: "summarization", 1: "faithful"}
num_objectives = len(output_name_map)
value_model = load_value_model(checkpoint=args.value_model_dir, torch_dtype=torch.float32, device=g_device, 
                                                tokenizer=tokenizer, num_objectives=num_objectives)
value_model.eval()
value_model.config.use_cache = not args.no_cache
value_tokenizer = tokenizer

if args.batched:
    value_generator = ValueGeneratorMultiHeadBatched(value_model, output_name_map, value_tokenizer, g_device, obj_weights, dtype=torch.float32)
    num_parallel = args.num_trials
else:
    value_generator = ValueGeneratorMultiHead(value_model, output_name_map, value_tokenizer, g_device, obj_weights, dtype=torch.float32)
    num_parallel = 1

search_alg = get_mavis_alg(tokenizer, generative_model, value_generator, max_completion_len, args=args, device=g_device,
                           return_strings=True, track_KL=args.track_KL, num_parallel=num_parallel,
                           force_no_eos_on_first_token=(not args.allow_eos_on_first_token))

log_dir = os.path.join(args.log_dir, "_".join(objectives), datetime.now().strftime("%m-%d_%H-%M"))
file_name = os.path.join(log_dir, f"mavis_{args.obj_weights}.csv")
os.makedirs(log_dir, exist_ok=True)

# Save the arguments to a YAML file
with open(os.path.join(log_dir, f"mavis_{args.obj_weights}_args.yaml"), "w") as f:
    yaml.dump(vars(args), f)

if __name__ == "__main__":
    eval_loop(inputs, prompt_indices, args.num_trials, search_alg, reward_model, rm_device, get_rewards, obj_weights=obj_weights,
              verbose=True, track_KL=args.track_KL, file_name=file_name, batched=args.batched, save_json=(args.num_trials==1))
    print("**********************************")
    print(f"Results saved to {file_name}")
    print("**********************************")