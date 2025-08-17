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
from utils.gen_utils import ValueGenerator, ValueGeneratorWithCache, ValueGeneratorBatched, get_mavis_alg, add_generation_args, add_mavis_args

A = argparse.ArgumentParser()
# TODO: Make beta a required argument
add_generation_args(A, {"data_split":"test"}) # dataset, data_split, temperature, topk, device, no_cache
add_mavis_args(A, {"normalize_values":True}) # value_model_dir, value_model_iter, beta, normalize_values
A.add_argument("--num_prompts", type=int, default=100, help="Number of prompts to evaluate on; the prompts are selected in order starting from index 0")
A.add_argument("--obj_weights", type=str, default="1.0,0.0,0.0", help="Comma-separated list of weights for the three objectives (Help, Harm, Humor for the anthropic dataset). If you do not want to evaluate on an objective, leave it empty (e.g. 0.5,,0.5).")
A.add_argument("--num_trials", type=int, default=3)
A.add_argument("--track_time", action="store_true")
A.add_argument("--seed", type=int, default=0, help="The random seed is set to this at the start of each prompt")
A.add_argument("--log_dir", type=str, default="mavis_logs/")
A.add_argument("--track_KL", action="store_true", help="If set, the average KL divergence over all sequences generated will be tracked and included in the results.")
args = A.parse_args()
assert not args.no_cache, "The no_cache argument will not be supported until the ValueGenerator class is updated."

verbose = True
prompt_indices = list(range(args.num_prompts))

print("Dataset:", args.dataset)
print(f"Prompt range: {prompt_indices[0]} to {prompt_indices[-1]}")

g_device, rm_device = assign_gpus()

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
value_model_types = [obj for obj, weight in obj_weights.items() if weight is not None and weight > 0.0]

value_models = {}
if args.dataset == "anthropic":
    loaded_assets = load_for_anthropic(
        csv_path="datasets/anthropic/",
        prompt_indices=prompt_indices, rewards=objectives, split=args.data_split, 
        pre_tokenized=False, rm_device=rm_device, obj_weights=obj_weights
    )
elif args.dataset == "summary":
    loaded_assets = load_for_summary(
        csv_path="datasets/summary/",
        prompt_indices=prompt_indices, rewards=objectives, split=args.data_split, 
        pre_tokenized=False, rm_device=rm_device, obj_weights=obj_weights
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
for objective in value_model_types:
    subdir = f"iter_{iter_nums[objective]}"
    print(f"Loading {objective} value model from {os.path.join(args.value_model_dir, subdir)}")
    value_models[objective] = load_value_model(checkpoint=os.path.join(args.value_model_dir, subdir, objective), torch_dtype=torch.float16, device=g_device, 
                                                tokenizer=tokenizer)
    value_models[objective].eval()
    value_models[objective].config.use_cache = not args.no_cache
value_tokenizer = tokenizer

if not args.no_cache:
    value_generator = ValueGeneratorWithCache(value_models, value_tokenizer, g_device, obj_weights)
else:
    value_generator = ValueGenerator(value_models, value_tokenizer, tokenizer, g_device, args.mix_ratio)

search_alg = get_mavis_alg(tokenizer, generative_model, value_generator, max_completion_len, args=args, device=g_device,
                           return_strings=True, track_KL=args.track_KL, do_normalization=args.normalize_values)

log_dir = os.path.join(args.log_dir, "_".join(objectives), datetime.now().strftime("%m-%d_%H-%M"))
file_name = os.path.join(log_dir, f"mavis_{args.obj_weights}.csv")
os.makedirs(log_dir, exist_ok=True)

# Save the arguments to a YAML file
with open(os.path.join(log_dir, f"mavis_{args.obj_weights}_args.yaml"), "w") as f:
    yaml.dump(vars(args), f)

if __name__ == "__main__":
    eval_loop(inputs, prompt_indices, args.num_trials, search_alg, reward_model, rm_device, get_rewards, obj_weights=obj_weights,
              verbose=True, track_time=args.track_time, track_KL=args.track_KL, file_name=file_name,)
    print("**********************************")
    print(f"Results saved to {file_name}")
    print("**********************************")